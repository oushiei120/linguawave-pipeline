#!/usr/bin/env python3
"""Shared building blocks for the KOTO cloud pipeline.

Single source of truth
----------------------
The library state lives on Tencent Cloud COS at ``state/library.json`` (gzip).
It holds per-episode metadata only; subtitles, grammar and translations live in
content-addressed files ``episodes/{id}.{sha8}.json`` (gzip, immutable).
``index.json`` is generated from the state after every published episode, so a
cancelled CI run never leaves git and COS disagreeing.

Every publish path (``cloud_transcribe.py`` and ``rebuild_library.py``) goes
through :func:`publish_library`, which runs :func:`validate_episode` on every
episode it has content for.
"""
import difflib
import gzip
import hashlib
import json
import os
import re
import subprocess
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

env_file = REPO_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

COS_SECRET_ID = os.environ.get("TENCENT_COS_SECRET_ID")
COS_SECRET_KEY = os.environ.get("TENCENT_COS_SECRET_KEY")
COS_REGION = os.environ.get("TENCENT_COS_REGION") or "ap-shanghai"
COS_BUCKET = os.environ.get("TENCENT_COS_BUCKET") or "koto-audio-1314383219"
COS_BASE_URL = f"https://{COS_BUCKET}.cos.{COS_REGION}.myqcloud.com"

# Private objects (ACL private, read only with the COS key): the pipeline's
# state and the blocked-word list. Everything else in the bucket is public-read
# because the app reads it without logging in.
STATE_KEY = "private/state/library.json"
LEGACY_STATE_KEY = "state/library.json"
POLICY_KEY = "private/content_policy.json"
INDEX_KEY = "index.json"
CONTENT_VERSION = 4
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

PUNCT_RE = re.compile(r"[\s、。！？!?,.，〜～・…\-「」『』（）()\"'“”]+")


def clean_text(s: str) -> str:
    return PUNCT_RE.sub("", s or "")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Process / HTTP helpers
# ---------------------------------------------------------------------------

def run(cmd: list, timeout: int = None, check: bool = False) -> subprocess.CompletedProcess:
    """subprocess.run that never crashes on non-UTF-8 output (ID3 tags etc.)."""
    res = subprocess.run(cmd, capture_output=True, timeout=timeout)
    res.stdout = res.stdout.decode("utf-8", errors="replace")
    res.stderr = res.stderr.decode("utf-8", errors="replace")
    if check and res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({res.returncode}): {res.stderr.strip()[-500:]}")
    return res


def ffprobe_duration(path_or_url: str) -> float:
    res = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path_or_url)], timeout=120)
    try:
        return float(res.stdout.strip())
    except ValueError:
        raise IOError(f"ffprobe could not read duration: {res.stderr.strip()[-300:]}")


def http_get(url: str, timeout: int = 60) -> bytes:
    """GET that transparently handles gzip Content-Encoding (urllib does not)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding", "").lower() == "gzip" or data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return data


def http_get_json(url: str, timeout: int = 60):
    return json.loads(http_get(url, timeout).decode("utf-8"))


def http_head(url: str, timeout: int = 15) -> tuple:
    """Returns (status_code, content_length or None, error string)."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            clen = resp.headers.get("Content-Length")
            return resp.getcode(), int(clen) if clen and clen.isdigit() else None, ""
    except urllib.error.HTTPError as e:
        return e.code, None, f"HTTP {e.code}"
    except Exception as e:  # network errors
        return 0, None, str(e)


# ---------------------------------------------------------------------------
# COS storage
# ---------------------------------------------------------------------------

class Storage:
    """COS-backed storage. ``dry_run_dir`` writes to disk instead of uploading."""

    def __init__(self, dry_run_dir: Path = None):
        self.dry_run_dir = Path(dry_run_dir) if dry_run_dir else None
        self._client = None
        self.head_cache = {}

    @property
    def client(self):
        if self._client is None:
            if not (COS_SECRET_ID and COS_SECRET_KEY):
                raise RuntimeError("TENCENT_COS_SECRET_ID / TENCENT_COS_SECRET_KEY missing from environment.")
            from qcloud_cos import CosConfig, CosS3Client
            self._client = CosS3Client(CosConfig(Region=COS_REGION, SecretId=COS_SECRET_ID,
                                                 SecretKey=COS_SECRET_KEY, Timeout=120))
        return self._client

    def url(self, key: str) -> str:
        return f"{COS_BASE_URL}/{key}"

    def put_bytes(self, key: str, body: bytes, content_type: str, cache_control: str, gzip_encoded: bool = False,
                  private: bool = False):
        if self.dry_run_dir:
            p = self.dry_run_dir / key
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
            return
        kwargs = {"ContentType": content_type, "CacheControl": cache_control}
        if gzip_encoded:
            kwargs["ContentEncoding"] = "gzip"
        if private:
            kwargs["ACL"] = "private"
        last = None
        for attempt in range(3):
            try:
                self.client.put_object(Bucket=COS_BUCKET, Body=body, Key=key, **kwargs)
                self.head_cache.pop(self.url(key), None)
                return
            except Exception as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"COS upload of {key} failed: {last}")

    def put_json(self, key: str, obj, cache_control: str, private: bool = False) -> bytes:
        raw = canonical_json(obj)
        self.put_bytes(key, gzip.compress(raw, mtime=0), "application/json; charset=utf-8",
                       cache_control, gzip_encoded=True, private=private)
        return raw

    def get_private_json(self, key: str):
        """Authenticated read of a private object. None only when it truly does
        not exist; any other problem (no key, no permission, network) raises,
        so the pipeline never mistakes "cannot read" for "empty"."""
        if self.dry_run_dir and (self.dry_run_dir / key).exists():
            data = (self.dry_run_dir / key).read_bytes()
            return json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)
        from qcloud_cos.cos_exception import CosServiceError
        try:
            resp = self.client.get_object(Bucket=COS_BUCKET, Key=key)
        except CosServiceError as e:
            if e.get_error_code() == "NoSuchKey":
                return None
            raise
        data = resp["Body"].get_raw_stream().read()
        return json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)

    def delete(self, key: str):
        if self.dry_run_dir:
            print(f"[KOTO] (试运行) 将删除 {key}")
            return
        last = None
        for attempt in range(3):
            try:
                self.client.delete_object(Bucket=COS_BUCKET, Key=key)
                self.head_cache.pop(self.url(key), None)
                return
            except Exception as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"COS delete of {key} failed: {last}")

    def upload_file(self, key: str, path: Path, content_type: str, cache_control: str):
        if self.dry_run_dir:
            return
        last = None
        for attempt in range(3):
            try:
                self.client.upload_file(Bucket=COS_BUCKET, LocalFilePath=str(path), Key=key, PartSize=2,
                                        MAXThread=2, ContentType=content_type, CacheControl=cache_control)
                return
            except Exception as e:
                if "UnavailableForLegalReasons" in str(e):
                    raise PermanentFailure(f"COS rejected {key}: UnavailableForLegalReasons (content moderation)")
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"COS upload of {key} failed: {last}")

    def get_json(self, key: str):
        """Returns parsed JSON or None if the object does not exist."""
        if self.dry_run_dir and (self.dry_run_dir / key).exists():
            data = (self.dry_run_dir / key).read_bytes()
            return json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)
        try:
            return http_get_json(self.url(key))
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            raise

    def head(self, url: str) -> tuple:
        if self.dry_run_dir and url.startswith(COS_BASE_URL):
            local = self.dry_run_dir / url[len(COS_BASE_URL) + 1:]
            if local.exists():
                return 200, None, ""
        if url not in self.head_cache:
            res = http_head(url)
            for _ in range(2):  # a network blip (TLS timeout) must not pull an episode
                if res[0] != 0:
                    break
                time.sleep(3)
                res = http_head(url)
            self.head_cache[url] = res
        return self.head_cache[url]


class PermanentFailure(Exception):
    """An episode that must not be retried (e.g. COS content moderation)."""


class ServiceStop(Exception):
    """An upstream account problem (quota, billing, auth). Stop the run; it is
    not the episode's fault, so it must never count as an episode failure."""


class RateLimitExhausted(ServiceStop):
    """Groq quota is used up for now; stop starting new episodes."""


class AccountProblem(ServiceStop):
    """HTTP 401/402/403 from an API: key revoked or balance exhausted."""


def canonical_json(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Library state
# ---------------------------------------------------------------------------

ENV_ERROR_PREFIXES = ("No module named", "cannot import name")


def empty_state() -> dict:
    return {"version": 1, "updated_at": "", "episodes": {}, "failures": {}, "quarantine": []}


def load_state(storage: Storage) -> dict:
    """Load state from COS; bootstrap from the live index.json if none exists yet."""
    state = storage.get_private_json(STATE_KEY)
    if state is None:
        # one-time move from the old public location
        state = storage.get_private_json(LEGACY_STATE_KEY)
        if state is not None:
            print("[KOTO] 从旧位置迁移题库状态到私有位置。")
    if state:
        for k, v in empty_state().items():
            state.setdefault(k, v)
        # failures caused by our own environment (e.g. a missing Python module) are
        # not the episode's fault; forget them so the episodes are retried
        env_failures = [k for k, v in state["failures"].items()
                        if str(v.get("last_error", "")).startswith(ENV_ERROR_PREFIXES)]
        for k in env_failures:
            del state["failures"][k]
        if env_failures:
            print(f"[KOTO] 清除了 {len(env_failures)} 条由运行环境错误造成的失败记录。")
        print(f"[KOTO] 已从腾讯云读取题库状态：{len(state['episodes'])} 集。")
        return state
    state = empty_state()
    index = storage.get_json(INDEX_KEY) or {}
    for ep in index.get("episodes", []):
        meta = dict(ep)
        meta["legacy"] = True  # published before the v4 gate; rebuild_library.py upgrades it
        state["episodes"][ep["id"]] = meta
    print(f"[KOTO] 腾讯云上还没有状态文件，已从线上 index.json 建立：{len(state['episodes'])} 集（标记为待重建）。")
    return state


def is_published(meta: dict) -> bool:
    return not meta.get("quarantined") and not meta.get("legacy")


def published_count(state: dict, source: dict) -> int:
    """Episodes of this station that are actually in the app (pulled ones excluded)."""
    return sum(1 for m in state["episodes"].values() if is_published(m)
               and (m.get("source_id") == source.get("id") or m.get("source_name") == source.get("name")))


def save_state(storage: Storage, state: dict):
    state["updated_at"] = now_iso()
    storage.put_json(STATE_KEY, state, cache_control="no-cache", private=True)


def record_failure(state: dict, ep_id: str, title: str, station: str, error: str, permanent: bool = False):
    entry = state["failures"].setdefault(ep_id, {"count": 0, "failures": []})
    entry["count"] = 3 if permanent else entry.get("count", 0) + 1
    entry.update(title=title, station=station, last_error=str(error)[:500], last_attempt=now_iso())
    entry["failures"] = (entry.get("failures", []) + [{"time": entry["last_attempt"], "error": str(error)[:500]}])[-5:]


def is_permanently_failed(state: dict, ep_id: str) -> bool:
    return state["failures"].get(ep_id, {}).get("count", 0) >= 3


# ---------------------------------------------------------------------------
# Audio chunking
# ---------------------------------------------------------------------------

def mean_volume_db(audio_path: Path) -> float:
    res = run(["ffmpeg", "-hide_banner", "-i", str(audio_path), "-af", "volumedetect", "-f", "null", "-"])
    m = re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB", res.stderr)
    return float(m.group(1)) if m else -20.0


def detect_silences(audio_path: Path, noise_db: float = None, min_dur: float = 0.2) -> list:
    """Pauses in speech. The threshold follows the episode's own loudness, so
    shows with background music under the talk still yield split points.
    Measured on ゆる言語学ラジオ #478 (38 min, music bed): a fixed -30 dB gave 29
    forced cuts; 10 dB below the mean volume gives 2, and 0 on a clean show."""
    if noise_db is None:
        noise_db = min(-20.0, max(-50.0, mean_volume_db(audio_path) - 10.0))
    res = run(["ffmpeg", "-hide_banner", "-i", str(audio_path), "-af",
               f"silencedetect=noise={noise_db:.1f}dB:d={min_dur}", "-f", "null", "-"])
    silences, start = [], None
    for line in res.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[0-9.]+)", line)
        if m:
            start = max(0.0, float(m.group(1)))
            continue
        m = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m and start is not None:
            silences.append((start, float(m.group(1))))
            start = None
    return silences


HARD_CUT_OVERLAP = 2.0


def plan_chunks(silences: list, total_duration: float, target: float = 90.0) -> tuple:
    """Split points in the middle of pauses near every `target` seconds.

    Returns (chunks, hard_cuts). Each chunk is a dict with the audio range to
    send (`start`/`end`) and the range whose words and sentences it owns
    (`keep_start`/`keep_end`). A split made with no pause nearby ("hard cut")
    sends 2 s of overlap on both sides so the word at the cut is heard whole by
    one of the two chunks; ownership is decided by midpoint.
    """
    if total_duration <= target * 1.25:
        return [{"start": 0.0, "end": total_duration, "keep_start": 0.0, "keep_end": total_duration}], 0
    mids = [((s + e) / 2.0, e - s) for s, e in silences]
    splits, hard, cur = [0.0], [False], target
    while cur < total_duration - 20.0:
        near = [(m, d) for m, d in mids if cur - 20.0 <= m <= cur + 20.0 and m > splits[-1] + 15.0]
        if not near:
            wide = [(m, d) for m, d in mids if cur - 35.0 <= m <= cur + 35.0 and m > splits[-1] + 15.0]
            near = sorted(wide, key=lambda x: -x[1])[:1]  # widest pause when the window had to grow
        if near:
            best, is_hard = min(near, key=lambda x: abs(x[0] - cur))[0], False
        else:
            best, is_hard = cur, True
        splits.append(round(best, 3))
        hard.append(is_hard)
        cur = best + target
    splits.append(total_duration)
    hard.append(False)
    chunks = []
    for i, (a, b) in enumerate(zip(splits, splits[1:])):
        if b <= a + 0.5:
            continue
        chunks.append({
            "start": max(0.0, a - (HARD_CUT_OVERLAP if hard[i] else 0.0)),
            "end": min(total_duration, b + (HARD_CUT_OVERLAP if hard[i + 1] else 0.0)),
            "keep_start": a, "keep_end": b,
        })
    return chunks, sum(hard)


def owned(item: dict, chunk: dict) -> bool:
    mid = (float(item["start"]) + float(item["end"])) / 2.0
    return chunk["keep_start"] <= mid < chunk["keep_end"] or (
        chunk["keep_end"] >= chunk["end"] and mid >= chunk["keep_start"])


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

HALLUCINATION_KEYWORDS = [
    "ご視聴ありがとうございました", "チャンネル登録", "高評価", "いいねとチャンネル登録",
    "ご視聴いただき", "お聞きいただきありがとうございました", "Thank you for watching",
    "Subtitles by", "Translated by", "Amara.org", "句読点",
]


def is_hallucination_or_loop(text: str, prev_norm_texts: list = None) -> bool:
    norm = clean_text(text)
    if not norm or len(norm) <= 1:
        return True
    for kw in HALLUCINATION_KEYWORDS:
        if kw in norm and len(norm) <= len(kw) + 12:
            return True
    if re.search(r"(.{5,30}?)\1{2,}", norm):
        return True
    if prev_norm_texts and len(norm) >= 4 and sum(1 for p in prev_norm_texts[-4:] if p == norm) >= 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Word timing
# ---------------------------------------------------------------------------

def clamp_word(w: dict) -> dict:
    """Whisper often stretches the first word after a pause back to where the
    pause began. Cap a word's duration by its length and keep its end."""
    n = max(1, len(clean_text(w.get("word", ""))))
    start, end = float(w["start"]), float(w["end"])
    max_dur = max(0.8, 0.35 * n)
    if end - start > max_dur:
        start = end - min(end - start, 0.15 * n + 0.2)
    return {"word": w.get("word", ""), "start": round(start, 2), "end": round(end, 2)}


def dedupe_words(words: list) -> list:
    seen, out = set(), []
    for w in sorted(words, key=lambda w: (float(w["start"]), float(w["end"]))):
        key = (round(float(w["start"]), 1), clean_text(w.get("word", "")))
        if key in seen or not clean_text(w.get("word", "")):
            continue
        seen.add(key)
        out.append(w)
    return out


def finalize_timing(segments: list, reorder: bool = True) -> list:
    """Clamp words, derive segment bounds from its own words, then make the
    segment list strictly ordered and non-overlapping.

    With reorder=False the existing order and ids are kept (translations and
    grammar of published episodes are keyed by id); a sentence that would start
    before its predecessor is pushed to the predecessor's end instead."""
    for s in segments:
        words = sorted((clamp_word(w) for w in s.get("words", [])), key=lambda w: (w["start"], w["end"]))
        s["words"] = words
        if words:
            s["start"], s["end"] = words[0]["start"], max(w["end"] for w in words)
            s["timing"] = "word"
        else:
            s["timing"] = "segment"
        s["start"], s["end"] = round(float(s["start"]), 2), round(float(s["end"]), 2)
        if s["end"] <= s["start"]:
            s["end"] = round(s["start"] + max(0.3, 0.12 * len(clean_text(s["text"]))), 2)
    if reorder:
        segments.sort(key=lambda s: s["start"])
    for a, b in zip(segments, segments[1:]):
        if b["start"] < a["start"] + 0.05:
            # only reachable with reorder=False: keep order, move b after a
            b["start"] = round(a["end"], 2)
            b["words"] = [w for w in b["words"] if w["start"] >= b["start"]]
            if b["end"] <= b["start"]:
                b["end"] = round(b["start"] + max(0.3, 0.12 * len(clean_text(b["text"]))), 2)
        if b["start"] < a["end"]:
            a["end"] = round(max(a["start"] + 0.05, b["start"]), 2)
            a["words"] = [w for w in a["words"] if w["start"] < a["end"]]
            for w in a["words"]:
                w["end"] = min(w["end"], a["end"])
    if reorder:
        for i, s in enumerate(segments):
            s["id"] = i
    return segments


def assign_words(segments: list, words: list, tolerance: float = 0.15) -> list:
    """Give every word to exactly one segment: the one whose Groq time span
    contains the word's midpoint. Words that fall in no kept segment (dropped
    hallucinations, duplicated gap-fill words) are discarded."""
    segments = sorted(segments, key=lambda s: float(s["start"]))
    words = dedupe_words(words)
    for s in segments:
        s["words"] = []
    j = 0
    for w in words:
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        while j < len(segments) and float(segments[j]["end"]) + tolerance < mid:
            j += 1
        cands = [k for k in (j - 1, j, j + 1) if 0 <= k < len(segments)
                 and float(segments[k]["start"]) - tolerance <= mid <= float(segments[k]["end"]) + tolerance]
        if not cands:
            continue
        inside = [k for k in cands if float(segments[k]["start"]) <= mid <= float(segments[k]["end"])]
        k = (inside or cands)[0]
        segments[k]["words"].append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
    return finalize_timing(segments)


def repair_legacy_words(segments: list) -> list:
    """Fix subtitles produced by the old ±0.3 s window alignment, where a word at
    a sentence boundary was copied into both neighbours. Order and ids are kept."""
    segments = sorted(segments, key=lambda s: s["id"])
    for a, b in zip(segments, segments[1:]):
        wa, wb = a.get("words") or [], b.get("words") or []
        keys_b = {(round(float(w["start"]), 2), w["word"]) for w in wb}
        shared = [w for w in wa if (round(float(w["start"]), 2), w["word"]) in keys_b]
        for w in shared:
            cw = clean_text(w["word"])
            belongs_to_next = clean_text(b["text"]).startswith(cw) and not clean_text(a["text"]).endswith(cw)
            key = (round(float(w["start"]), 2), w["word"])
            if belongs_to_next:
                a["words"] = [x for x in a["words"] if (round(float(x["start"]), 2), x["word"]) != key]
            else:
                b["words"] = [x for x in b["words"] if (round(float(x["start"]), 2), x["word"]) != key]
    seen = set()
    for s in segments:
        s.pop("display_text", None)
        kept = []
        for w in s.get("words") or []:
            key = (round(float(w["start"]), 2), w["word"])
            if key not in seen:
                seen.add(key)
                kept.append(w)
        s["words"] = kept
    return finalize_timing(segments, reorder=False)


def timing_report(segments: list) -> dict:
    """Numbers the publish gate checks."""
    seen, shared, overlaps, inverted, long_first, outside = set(), 0, 0, 0, 0, 0
    for s in segments:
        if s["end"] <= s["start"]:
            inverted += 1
        ws = s.get("words") or []
        if ws:
            w0 = ws[0]
            if (w0["end"] - w0["start"]) / max(1, len(clean_text(w0["word"]))) > 1.0:
                long_first += 1
        for w in ws:
            key = (round(w["start"], 2), w["word"])
            if key in seen:
                shared += 1
            seen.add(key)
            if w["start"] < s["start"] - 0.01 or w["end"] > s["end"] + 0.01:
                outside += 1
    for a, b in zip(segments, segments[1:]):
        if b["start"] < a["end"] - 0.01:
            overlaps += 1
    n = max(1, len(segments))
    return {"inverted": inverted, "overlaps": overlaps, "shared_words": shared,
            "words_outside": outside, "long_first_word_ratio": round(long_first / n, 4)}


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def deepseek_json(prompt: str, api_key: str, timeout: int = 120) -> dict:
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
               "response_format": {"type": "json_object"}, "temperature": 0.2}
    req = urllib.request.Request(DEEPSEEK_URL, data=json.dumps(payload).encode("utf-8"), headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        if e.code in (401, 402, 403):
            raise AccountProblem(f"DeepSeek HTTP {e.code}"
                                 + ("（账户余额不足，请充值）" if e.code == 402 else "（API Key 无效或无权限）"))
        if e.code == 429:
            raise RateLimitExhausted("DeepSeek HTTP 429")
        raise
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", content)
    return json.loads(content, strict=False)


class TranslationError(Exception):
    pass


def _translate_batch(batch: list, before: list, after: list, title: str, source: str, api_key: str) -> dict:
    lines = "\n".join(f"[{i + 1}] {s['text']}" for i, s in enumerate(batch))
    ctx_b = "\n".join(s["text"] for s in before) or "（无）"
    ctx_a = "\n".join(s["text"] for s in after) or "（无）"
    prompt = f"""你是资深中日同传译员。把下面编号的日语播客句子逐句译成自然的中文。
节目：{source}《{title}》

上文（只作语境，不要翻译）：
{ctx_b}

需要翻译的句子（共 {len(batch)} 句，编号 1 到 {len(batch)}）：
{lines}

下文（只作语境，不要翻译）：
{ctx_a}

要求：每个编号都要有译文，不要合并、拆分或重新编号；口头语气词也要译出。
只输出 JSON，键是编号字符串，值是译文，例如 {{"1": "……", "2": "……"}}，必须正好包含 "1" 到 "{len(batch)}" 这 {len(batch)} 个键。"""
    res = deepseek_json(prompt, api_key)
    if isinstance(res.get("translations"), dict):
        res = res["translations"]
    expected = {str(i + 1) for i in range(len(batch))}
    got = {str(k).strip("[] ") : v for k, v in res.items()}
    if set(got) != expected:
        raise TranslationError(f"expected keys 1..{len(batch)}, got {sorted(got)[:5]}… ({len(got)} keys)")
    out = {}
    for i, s in enumerate(batch):
        t = str(got[str(i + 1)] or "").strip()
        if not t:
            raise TranslationError(f"empty translation for line {i + 1}")
        out[str(s["id"])] = t
    return out


def translate_segments(segments: list, title: str, source: str, api_key: str, only_ids: set = None,
                       batch_size: int = 40) -> dict:
    """Translate segments (or just `only_ids`). Every requested id gets a
    translation or TranslationError is raised — never a partial result."""
    by_pos = {s["id"]: i for i, s in enumerate(segments)}
    targets = [s for s in segments if only_ids is None or s["id"] in only_ids]
    out = {}

    def do(batch: list, depth: int = 0):
        first, last = by_pos[batch[0]["id"]], by_pos[batch[-1]["id"]]
        before = segments[max(0, first - 3):first]
        after = segments[last + 1:last + 4]
        err = None
        for attempt in range(3):
            try:
                out.update(_translate_batch(batch, before, after, title, source, api_key))
                return
            except ServiceStop:
                raise
            except Exception as e:
                err = e
                time.sleep(2 * (attempt + 1))
        if len(batch) > 5 and depth < 3:
            mid = len(batch) // 2
            do(batch[:mid], depth + 1)
            do(batch[mid:], depth + 1)
            return
        raise TranslationError(f"batch {batch[0]['id']}~{batch[-1]['id']} failed: {err}")

    # contiguous runs only, so context lines are real neighbours
    run_, runs = [], []
    for s in targets:
        if run_ and by_pos[s["id"]] != by_pos[run_[-1]["id"]] + 1:
            runs.append(run_)
            run_ = []
        run_.append(s)
    if run_:
        runs.append(run_)
    for r in runs:
        for i in range(0, len(r), batch_size):
            do(r[i:i + batch_size])
    return out


def sample_for_overview(segments: list, max_chars: int = 9000) -> list:
    total = sum(len(s["text"]) + 6 for s in segments)
    if total <= max_chars:
        return segments
    stride = total / max_chars
    picked, acc, nxt = [], 0.0, 0.0
    for s in segments:
        if acc >= nxt:
            picked.append(s)
            nxt += stride * (len(s["text"]) + 6)
        acc += len(s["text"]) + 6
    return picked


def best_segment_for_text(segments: list, text: str) -> tuple:
    target = clean_text(text)
    if not target:
        return None, 0.0
    best, best_r = None, 0.0
    for s in segments:
        c = clean_text(s["text"])
        if not c or abs(len(c) - len(target)) > max(len(c), len(target)):
            continue
        r = difflib.SequenceMatcher(None, target, c).ratio()
        if r > best_r:
            best, best_r = s, r
    return best, best_r


def repair_cards(cards: list, segments: list) -> list:
    """Point every card at the sentence whose text it quotes; drop cards that
    match nothing. Start/end/japanese always come from the sentence itself."""
    by_id = {s["id"]: s for s in segments}
    out, used = [], set()
    for c in cards or []:
        if not isinstance(c, dict):
            continue
        quote = c.get("ja_original") or c.get("japanese") or ""
        seg = by_id.get(c.get("segment_id"))
        ok = seg is not None and (not quote or difflib.SequenceMatcher(
            None, clean_text(quote), clean_text(seg["text"])).ratio() >= 0.5)
        if not ok:
            seg, r = best_segment_for_text(segments, quote)
            if seg is None or r < 0.6:
                continue
        if seg["id"] in used:
            continue
        used.add(seg["id"])
        card = {k: c.get(k, "") for k in ("target_pattern", "target_grammar", "zh_prompt", "shadowing_tip")}
        card.update(segment_id=seg["id"], ja_original=seg["text"], japanese=seg["text"],
                    start=seg["start"], end=seg["end"])
        out.append(card)
    out.sort(key=lambda c: c["start"])
    return out


def generate_overview(segments: list, title: str, source: str, api_key: str) -> dict:
    sample = sample_for_overview(segments)
    text = "\n".join(f"{s['id']}: {s['text']}" for s in sample)
    prompt = f"""你是精通中日双语的口译教学专家。下面是一期日本播客的逐句文字稿（从全集均匀抽取，行首数字是句子编号）。
播客：{source}
标题：{title}

{text}

请输出 JSON：
{{"summary": "80-120字中文概要，覆盖全集而不只是开头",
  "key_points": ["日语表达 —— 中文释义", ...3~5条],
  "practice_cards": [{{"segment_id": 句子编号(必须是上面出现过的编号), "ja_original": "该编号句子的原文(逐字照抄)", "target_pattern": "句型", "target_grammar": "25字以内解析", "zh_prompt": "中文视译提示句", "shadowing_tip": "跟读建议"}}, ...10~15张，从全集前中后段都要选]}}"""
    last_err = None
    for _ in range(2):
        try:
            res = deepseek_json(prompt, api_key)
            cards = repair_cards(res.get("practice_cards", []), segments)
            summary = (res.get("summary") or "").strip()
            if summary and len(cards) >= 5:
                return {"summary": summary, "key_points": res.get("key_points", [])[:5], "practice_cards": cards}
            last_err = f"summary={bool(summary)} cards={len(cards)}"
        except ServiceStop:
            raise
        except Exception as e:
            last_err = e
    raise TranslationError(f"overview failed: {last_err}")


# ---------------------------------------------------------------------------
# Content policy (blocked terms, allowed stations)
# ---------------------------------------------------------------------------

# The blocked-word list is kept out of the (public) repository: it lives in the
# private COS object POLICY_KEY. Edit it with scripts/policy_tool.py.
POLICY_FILE = REPO_ROOT / "content_policy.json"  # local working copy, git-ignored
CATALOG_FILE = REPO_ROOT / "source_catalog.json"
_POLICY_CACHE = {}
_STRIP_RE = re.compile(r"[\s\u3000・·•、。，,.！!？?「」『』（）()\[\]【】\-—_~〜～:：;；'\"“”‘’/]+")


def normalize_for_policy(text: str) -> str:
    """NFKC (full/half width), lower-case, no spaces or punctuation, so that
    「習 近平」「ﾃﾛ事件」「xi jinping」 still match."""
    return _STRIP_RE.sub("", unicodedata.normalize("NFKC", text or "").lower())


def load_policy(path: Path = None) -> dict:
    """Blocked terms + allowed stations. From `path` when given (tests, local
    edits), otherwise from the private COS object; raises if it cannot be read,
    so nothing is ever published without the word list."""
    if path is not None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        if "raw" not in _POLICY_CACHE:
            fetched = Storage().get_private_json(POLICY_KEY)
            if not fetched or not fetched.get("blocked_terms"):
                raise RuntimeError(f"屏蔽词表 {POLICY_KEY} 读取失败或为空，拒绝在没有词表的情况下运行。")
            _POLICY_CACHE["raw"] = fetched
        raw = _POLICY_CACHE["raw"]
    terms = []
    for category, words in raw.get("blocked_terms", {}).items():
        for w in words:
            n = normalize_for_policy(w)
            if n:
                terms.append((n, w, category))
    allowed = [normalize_for_policy(a) for a in raw.get("allowed_phrases", []) if normalize_for_policy(a)]
    catalog_ids = sorted(c["id"] for c in json.loads(CATALOG_FILE.read_text(encoding="utf-8")))
    digest = hashlib.sha256(json.dumps([sorted(set(t[0] for t in terms)), allowed, catalog_ids],
                                       ensure_ascii=False).encode()).hexdigest()[:12]
    return {"terms": terms, "allowed": allowed, "catalog_ids": set(catalog_ids), "hash": digest}


def policy_texts(meta: dict, content: dict = None) -> list:
    """(where, text) pairs that must be free of blocked terms."""
    out = [("标题", meta.get("title", "")), ("节目简介", meta.get("description", "")),
           ("概要", meta.get("summary", ""))]
    out += [("要点", k) for k in meta.get("key_points", []) or []]
    if content:
        segs = content.get("transcript", {}).get("raw", {}).get("segments", [])
        out += [(f"字幕#{s.get('id')}", s.get("text", "")) for s in segs]
        # also the whole transcript run together, so a term split across two lines matches
        out.append(("字幕全文", "".join(s.get("text", "") for s in segs)))
        trans = content.get("translation", {}).get("segments", {})
        out += [(f"翻译#{k}", v) for k, v in trans.items()]
        for c in content.get("interpretation", {}).get("cards", []) or []:
            out += [("卡片", " ".join(str(c.get(k, "")) for k in
                                     ("ja_original", "zh_prompt", "target_pattern", "target_grammar", "shadowing_tip")))]
    return out


def policy_hits(policy: dict, meta: dict, content: dict = None, limit: int = 8) -> list:
    """Blocked terms found, as "词（分类）@位置：上下文" strings (empty = clean)."""
    hits, seen = [], set()
    for where, text in policy_texts(meta, content):
        norm = normalize_for_policy(text)
        if not norm:
            continue
        for a in policy["allowed"]:
            norm = norm.replace(a, "")
        for n, word, category in policy["terms"]:
            if n in norm and word not in seen:
                seen.add(word)
                i = norm.find(n)
                hits.append(f"{word}（{category}）@{where}：…{norm[max(0, i - 12):i + len(n) + 12]}…")
                if len(hits) >= limit:
                    return hits
    return hits


def purge_episode(storage: Storage, state: dict, ep_id: str, reason: str, details: list):
    """Remove an episode for good: delete its audio and content files from COS,
    drop it from the library, and record it so it is never re-added."""
    meta = state["episodes"].pop(ep_id, None) or {}
    for url in (meta.get("audio_url", ""), meta.get("content_url", "")):
        if url.startswith(COS_BASE_URL + "/"):
            storage.delete(url[len(COS_BASE_URL) + 1:])
    state.setdefault("removed", {})[ep_id] = {
        "title": meta.get("title", ""), "station": meta.get("source_name", ""),
        "reason": reason, "details": details[:8], "removed_at": now_iso()}
    record_failure(state, ep_id, meta.get("title", ""), meta.get("source_name", ""),
                   f"{reason}: {'; '.join(details[:3])}", permanent=True)
    print(f"[KOTO] 🗑 已下架并删除 {ep_id}「{meta.get('title', '')[:24]}」：{reason} {details[:3]}")


def policy_sweep(storage: Storage, state: dict, policy: dict) -> list:
    """Re-check every library episode whose last check used a different word
    list or station list. Returns the purged episode ids."""
    purged = []
    for ep_id, meta in list(state["episodes"].items()):
        if meta.get("policy_hash") == policy["hash"]:
            continue
        if meta.get("source_id") not in policy["catalog_ids"]:
            purge_episode(storage, state, ep_id, "STATION_NOT_ALLOWED", [f"电台 {meta.get('source_name')} 不在电台目录中"])
            purged.append(ep_id)
            continue
        content = None
        if meta.get("content_url"):
            try:
                content = http_get_json(meta["content_url"], timeout=180)
            except Exception as e:
                print(f"[KOTO] 复查 {ep_id} 时读取字幕失败，下次再查：{e}")
                continue
        hits = policy_hits(policy, meta, content)
        if hits:
            purge_episode(storage, state, ep_id, "CONTENT_POLICY", hits)
            purged.append(ep_id)
        else:
            meta["policy_hash"] = policy["hash"]
    print(f"[KOTO] 屏蔽词复查完成：下架 {len(purged)} 集。")
    return purged


# ---------------------------------------------------------------------------
# Content files & publish gate
# ---------------------------------------------------------------------------

def build_content(ep_id: str, segments: list, grammar: list, translations: dict, cards: list, dropped: list) -> dict:
    return {
        "id": ep_id,
        "content_version": CONTENT_VERSION,
        "transcript": {
            "content_version": CONTENT_VERSION,
            "raw": {"segments": segments},
            "grammar": {"version": 2, "segments": grammar},
            # grammar is analysed on `text`, the same field the app displays and times
            "grammar_texts": {str(s["id"]): s["text"] for s in segments},
            "dropped": dropped or [],
        },
        "translation": {"version": 3, "segments": translations},
        "interpretation": {"cards": cards},
    }


def content_sha(content: dict) -> str:
    return hashlib.sha256(canonical_json(content)).hexdigest()[:12]


def episode_stats(segments: list, translations: dict, audio_duration: float, hard_cuts: int = 0) -> dict:
    last_end = max((s["end"] for s in segments), default=0.0)
    rep = timing_report(segments)
    return {
        "sentences_count": len(segments),
        "translation_coverage": round(sum(1 for s in segments if translations.get(str(s["id"]))) / max(1, len(segments)), 4),
        "coverage": round(last_end / audio_duration, 4) if audio_duration else 0.0,
        "hard_cuts": hard_cuts,
        **rep,
    }


AUDIO_URL_RE = re.compile(r"/audio/([0-9a-f]{16})-([0-9a-f]{8})\.mp3$")


def validate_episode(meta: dict, content: dict, storage: Storage, policy: dict = None) -> list:
    """Return the list of reasons this episode must not be published (empty = OK)."""
    r = []
    policy = policy or load_policy()
    if meta.get("source_id") not in policy["catalog_ids"]:
        r.append(f"STATION_NOT_ALLOWED {meta.get('source_id')}")
    hits = policy_hits(policy, meta, content)
    if hits:
        r.append("CONTENT_POLICY " + "; ".join(hits))
    ep_id = meta.get("id", "")
    if meta.get("content_version", 0) < CONTENT_VERSION:
        r.append(f"CONTENT_VERSION {meta.get('content_version')} < {CONTENT_VERSION}")
    sha = meta.get("audio_sha256") or ""
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        r.append("AUDIO_SHA256_INVALID")
    nbytes, dur = int(meta.get("audio_bytes") or 0), float(meta.get("audio_duration") or 0)
    if nbytes <= 0:
        r.append("AUDIO_BYTES_MISSING")
    if dur <= 0:
        r.append("AUDIO_DURATION_MISSING")
    m = AUDIO_URL_RE.search(meta.get("audio_url", ""))
    if not m or m.group(1) != ep_id or not sha.startswith(m.group(2)):
        r.append("AUDIO_URL_NOT_VERSIONED")
    else:
        code, clen, err = storage.head(meta["audio_url"])
        if code != 200:
            r.append(f"AUDIO_HEAD_FAILED {err or code}")
        elif clen is not None and clen != nbytes:
            r.append(f"AUDIO_BYTES_MISMATCH {clen} != {nbytes}")
    if not meta.get("stats"):
        r.append("STATS_MISSING")
    if not (meta.get("summary") or "").strip():
        r.append("SUMMARY_EMPTY")

    tr = content.get("transcript", {})
    segs = tr.get("raw", {}).get("segments", [])
    if not segs:
        return r + ["NO_SEGMENTS"]
    if [s.get("id") for s in segs] != list(range(len(segs))):
        r.append("SEGMENT_IDS_NOT_SEQUENTIAL")
    rep = timing_report(segs)
    for key in ("inverted", "overlaps", "shared_words", "words_outside"):
        if rep[key]:
            r.append(f"TIMING_{key.upper()} {rep[key]}")
    if rep["long_first_word_ratio"] > 0.01:
        r.append(f"TIMING_LONG_FIRST_WORD {rep['long_first_word_ratio']:.1%}")
    last_end = segs[-1]["end"]
    if dur and not (abs(dur - last_end) <= 60 or last_end / dur >= 0.9):
        r.append(f"SUBTITLES_END_EARLY {last_end:.0f}s / {dur:.0f}s")
    if any("句読点" in s.get("text", "") for s in segs):
        r.append("PROMPT_LEAK")

    trans = content.get("translation", {}).get("segments", {})
    ids = {str(s["id"]) for s in segs}
    if set(trans) - ids:
        r.append(f"TRANSLATION_ORPHAN_KEYS {len(set(trans) - ids)}")
    missing = [i for i in ids if not (trans.get(i) or "").strip()]
    if missing:
        r.append(f"TRANSLATION_MISSING {len(missing)}")

    grammar = tr.get("grammar", {}).get("segments", [])
    if not grammar:
        r.append("GRAMMAR_EMPTY")
    gtexts = tr.get("grammar_texts", {})
    if any(gtexts.get(str(s["id"])) != s["text"] for s in segs):
        r.append("GRAMMAR_TEXTS_MISMATCH")

    cards = content.get("interpretation", {}).get("cards", [])
    by_id = {s["id"]: s for s in segs}
    bad_cards = [c for c in cards if c.get("segment_id") not in by_id
                 or c.get("ja_original") != by_id[c["segment_id"]]["text"]]
    if bad_cards:
        r.append(f"CARDS_POINT_TO_WRONG_SENTENCE {len(bad_cards)}")
    if len(cards) < 5:
        r.append(f"TOO_FEW_CARDS {len(cards)}")
    return r


def publish_library(storage: Storage, state: dict, new_content: dict, allow_shrink: bool = False,
                    expected_removals: int = 0) -> dict:
    """Validate, upload content files, rebuild index.json, save state.

    `new_content` maps episode id -> content dict produced in this run. Those
    are fully validated; previously published episodes are re-checked with HEAD
    requests only. Returns {"published": [...], "quarantined": [...]}.
    """
    quarantined, published = [], []
    for ep_id, content in new_content.items():
        meta = state["episodes"][ep_id]
        sha = content_sha(content)
        meta["content_sha"] = sha
        meta["content_url"] = storage.url(f"episodes/{ep_id}.{sha}.json")
        policy = load_policy()
        reasons = validate_episode(meta, content, storage, policy)
        if reasons:
            meta["quarantined"] = reasons
            quarantined.append({"id": ep_id, "title": meta.get("title", ""), "reasons": reasons})
            print(f"[KOTO] ⚠️ {ep_id}「{meta.get('title', '')[:24]}」未通过门槛：{reasons}")
            continue
        key = f"episodes/{ep_id}.{sha}.json"
        if storage.head(storage.url(key))[0] != 200:
            storage.put_json(key, content, cache_control="public, max-age=31536000, immutable")
        meta.pop("quarantined", None)
        meta.pop("legacy", None)
        meta["validated_at"] = now_iso()
        meta["policy_hash"] = policy["hash"]

    catalog_ids = load_policy()["catalog_ids"]
    index_eps = []
    for ep_id, meta in state["episodes"].items():
        if meta.get("source_id") not in catalog_ids:
            continue  # station removed from the catalog: never shown
        if ep_id in new_content:
            if meta.get("quarantined"):
                continue
        else:
            # Published earlier, or pulled earlier because its files did not resolve:
            # re-check on every publish. A pulled episode is marked `quarantined`, so it
            # stops counting toward its station's minimum, and comes back by itself
            # once its files resolve again.
            pulled = meta.get("quarantined") or []
            if pulled and not str(pulled[0]).startswith("FILES_UNREACHABLE"):
                continue
            ok_audio = storage.head(meta.get("audio_url", ""))[0] == 200
            ok_content = storage.head(meta.get("content_url", ""))[0] == 200
            if not (ok_audio and ok_content):
                reason = f"FILES_UNREACHABLE audio={ok_audio} content={ok_content}"
                meta["quarantined"] = [reason]
                quarantined.append({"id": ep_id, "title": meta.get("title", ""), "reasons": [reason]})
                continue
            meta.pop("quarantined", None)
        entry = {k: v for k, v in meta.items() if k not in ("quarantined", "validated_at")}
        entry["duration"] = float(meta.get("audio_duration") or meta.get("duration") or 0)
        index_eps.append(entry)
        published.append(ep_id)

    index_eps.sort(key=lambda e: e.get("published", ""), reverse=True)
    live = storage.get_json(INDEX_KEY) or {}
    live_count = len(live.get("episodes", []))
    if live_count and len(index_eps) < int((live_count - expected_removals) * 0.8) and not allow_shrink:
        raise RuntimeError(f"拒绝发布：线上 {live_count} 集，本次只有 {len(index_eps)} 集合格（降幅超过 20%）。"
                           f"确认要缩减时加 --allow-shrink。")

    state["quarantine"] = quarantined
    index = {"version": 3, "count": len(index_eps), "exported_at": now_iso(), "episodes": index_eps}
    storage.put_json(INDEX_KEY, index, cache_control="no-cache")
    save_state(storage, state)
    print(f"[KOTO] ⚡ 已发布 index.json：{len(index_eps)} 集（隔离 {len(quarantined)} 集）。")
    write_step_summary(quarantined, len(index_eps))
    return {"published": published, "quarantined": quarantined}


def write_step_summary(quarantined: list, count: int):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n### 发布门槛\n- 已发布：{count} 集\n- 隔离：{len(quarantined)} 集\n\n")
            if quarantined:
                f.write("| ID | 标题 | 原因 |\n|---|---|---|\n")
                for q in quarantined:
                    f.write(f"| `{q['id']}` | {q['title'][:30]} | {'; '.join(q['reasons'])} |\n")
    except OSError:
        pass
