#!/usr/bin/env python3
"""KOTO cloud worker: RSS -> CBR audio on COS -> Groq Whisper -> word timing ->
DeepSeek translation/overview -> GiNZA bunsetsu -> publish gate -> index.json.

State lives on COS (see koto_lib). Each episode is published as soon as it is
done, so cancelling the job loses at most the episode in progress.
"""
import argparse
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

socket.setdefaulttimeout(60)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import koto_lib as K  # noqa: E402

UA = K.UA
ITUNES_NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}


def load_curated_sources() -> list:
    items = json.loads((K.REPO_ROOT / "source_catalog.json").read_text(encoding="utf-8"))
    print(f"[CloudWorker] 从 source_catalog.json 加载了 {len(items)} 个电台。")
    return items


# Some hosts (e.g. Substack for ブックカタリスト) answer 403 to browser-like
# requests from data-center IPs but serve podcast clients; try those next.
PODCAST_CLIENT_UAS = [UA, "AppleCoreMedia/1.0.0.21E236 (iPhone; U; CPU OS 17_4 like Mac OS X; ja_jp)",
                      "Podcasts/1.0", "Overcast/3.0 (+http://overcast.fm/; Apple Watch podcast app)"]


def open_with_fallback_ua(url: str, timeout: int):
    last = None
    for ua in PODCAST_CLIENT_UAS:
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (401, 403, 429):
                raise
            last = e
    raise last


def fetch_feed_xml(url: str, timeout: int = 20, attempts: int = 4) -> ET.Element:
    last = None
    for i in range(attempts):
        try:
            with open_with_fallback_ua(url, timeout) as resp:
                return ET.fromstring(resp.read())
        except urllib.error.HTTPError:
            raise  # a real answer from the server; retrying will not change it
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e  # TLS EOF / reset / timeout: transient, try again
            time.sleep(2 * (i + 1))
    raise last


def download_file(url: str, dest: Path, timeout: int = 900, max_retries: int = 3):
    last = None
    for attempt in range(max_retries):
        try:
            started = time.time()
            with open_with_fallback_ua(url, 60) as resp:
                clen = resp.headers.get("Content-Length")
                expected = int(clen) if clen and clen.isdigit() else None
                with open(dest, "wb") as f:
                    while True:
                        if time.time() - started > timeout:
                            raise TimeoutError(f"download exceeded {timeout}s")
                        chunk = resp.read(1 << 19)
                        if not chunk:
                            break
                        f.write(chunk)
            if expected and dest.stat().st_size != expected:
                raise IOError(f"Content-Length mismatch: expected {expected}, got {dest.stat().st_size}")
            K.ffprobe_duration(dest)
            return
        except Exception as e:
            last = e
            dest.unlink(missing_ok=True)
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
    raise IOError(f"download failed after {max_retries} attempts: {last}")


def convert_to_cbr_mp3(src: Path, dst: Path) -> dict:
    K.run(["ffmpeg", "-y", "-i", str(src), "-vn", "-map_metadata", "-1", "-ac", "1",
           "-c:a", "libmp3lame", "-b:a", "64k", str(dst)], check=True)
    data = dst.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    return {"path": dst, "sha256": sha, "sha8": sha[:8], "duration": K.ffprobe_duration(dst), "bytes": len(data)}


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

class GroqClient:
    """Groq whisper-large-v3, paced to stay under the per-minute request limit.

    Last runs hit 429 with Retry-After: 3 on almost every chunk (a requests-per-
    minute limit) and burned the whole wait budget in 5 s steps. Requests are
    now spaced by `min_interval`; short waits (<= 15 s) are normal pacing and do
    not count toward `max_wait_total`, which only guards against long waits
    (hourly / daily audio quota).
    """

    URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, api_key: str, max_wait_total: float = 1200.0, min_interval: float = 3.2):
        self.api_key = api_key
        self.waited = 0.0
        self.max_wait_total = max_wait_total
        self.min_interval = min_interval
        self._last_request = 0.0

    def _body(self, path: Path) -> tuple:
        boundary = "----KOTO" + hashlib.md5(str(time.time()).encode()).hexdigest()
        body = bytearray()
        for name, value in (("model", "whisper-large-v3"), ("language", "ja"), ("response_format", "verbose_json"),
                            ("temperature", "0.0"), ("timestamp_granularities[]", "segment"),
                            ("timestamp_granularities[]", "word")):
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                 f"Content-Type: audio/mpeg\r\n\r\n").encode()
        body += path.read_bytes() + b"\r\n" + f"--{boundary}--\r\n".encode()
        return bytes(body), boundary

    def transcribe(self, path: Path, offset: float) -> tuple:
        body, boundary = self._body(path)
        headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": UA,
                   "Content-Type": f"multipart/form-data; boundary={boundary}"}
        consecutive_429 = 0
        for attempt in range(10):
            gap = self.min_interval - (time.time() - self._last_request)
            if gap > 0:
                time.sleep(gap)
            self._last_request = time.time()
            req = urllib.request.Request(self.URL, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code in (401, 402, 403):
                    raise K.AccountProblem(f"Groq HTTP {e.code}（API Key 无效、无权限或欠费）")
                if e.code != 429:
                    if e.code >= 500 and attempt < 4:
                        time.sleep(5 * (attempt + 1))
                        continue
                    raise
                consecutive_429 += 1
                ra = (e.headers.get("retry-after") or "").strip().rstrip("s")
                try:
                    wait = float(ra)
                except ValueError:
                    wait = 15.0 * consecutive_429
                wait = max(wait, 4.0 * consecutive_429)
                long_wait = wait > 15
                if wait > 600 or consecutive_429 > 8 or (long_wait and self.waited + wait > self.max_wait_total):
                    raise K.RateLimitExhausted(f"Groq 429: retry-after={ra or '?'}, long waits {self.waited:.0f}s this run")
                if long_wait:
                    print(f"[CloudWorker] Groq 限流，等待 {wait:.0f} 秒（本次长等待累计 {self.waited:.0f} 秒）")
                    self.waited += wait
                time.sleep(wait)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt >= 4:
                    raise
                time.sleep(5 * (attempt + 1))
        else:
            raise K.RateLimitExhausted("Groq request did not succeed after 10 attempts")

        segs = [{"start": round(float(s.get("start", 0)) + offset, 2), "end": round(float(s.get("end", 0)) + offset, 2),
                 "text": (s.get("text") or "").strip(), "no_speech_prob": float(s.get("no_speech_prob", 0.0))}
                for s in data.get("segments", [])]
        words = [{"word": w.get("word", ""), "start": round(float(w.get("start", 0)) + offset, 2),
                  "end": round(float(w.get("end", 0)) + offset, 2)} for w in data.get("words", [])]
        return segs, words


def cut_chunk(audio: Path, start: float, end: float, dest: Path):
    # re-encode (not -c copy) so the chunk starts exactly at `start`, with no
    # MP3 bit-reservoir garbage in the first frame
    K.run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(audio),
           "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "48k", str(dest)], check=True)


def filter_segments(segments: list) -> tuple:
    kept, dropped, prev = [], [], []
    for s in sorted(segments, key=lambda s: s["start"]):
        n = len(K.clean_text(s["text"]))
        if s.get("no_speech_prob", 0.0) > 0.88 and n <= 3:
            dropped.append({"segment": s, "reason": f"no_speech_prob {s['no_speech_prob']:.2f}"})
        elif K.is_hallucination_or_loop(s["text"], prev):
            dropped.append({"segment": s, "reason": "hallucination_or_loop"})
        else:
            prev.append(K.clean_text(s["text"]))
            kept.append(s)
    return kept, dropped


def transcribe(audio: Path, groq: GroqClient, duration: float) -> tuple:
    silences = K.detect_silences(audio)
    chunks, hard_cuts = K.plan_chunks(silences, duration)
    print(f"[CloudWorker] Groq 转录：{len(chunks)} 段（无停顿硬切 {hard_cuts} 处，硬切处前后重叠 {K.HARD_CUT_OVERLAP:.0f} 秒）")
    raw, words = [], []
    with tempfile.TemporaryDirectory() as d:
        for i, c in enumerate(chunks):
            p = Path(d) / f"c{i}.mp3"
            cut_chunk(audio, c["start"], c["end"], p)
            s, w = groq.transcribe(p, offset=c["start"])
            # each sentence/word belongs to exactly one chunk (by midpoint), so the
            # overlap around a hard cut never produces duplicates
            raw += [x for x in s if K.owned(x, c)]
            words += [x for x in w if K.owned(x, c)]
    kept, dropped = filter_segments(raw)
    kept, words = fill_gaps(kept, words, audio, groq, duration)
    segments = K.assign_words([{"start": s["start"], "end": s["end"], "text": s["text"]} for s in kept], words)
    return segments, dropped, hard_cuts


def fill_gaps(segments: list, words: list, audio: Path, groq: GroqClient, duration: float, min_gap: float = 5.0):
    """Re-transcribe gaps > min_gap between kept segments. Words of the first
    pass inside a re-transcribed gap are removed so nothing is duplicated."""
    if not segments:
        return segments, words
    gaps = []
    if segments[0]["start"] >= min_gap:
        gaps.append((0.0, segments[0]["start"]))
    for a, b in zip(segments, segments[1:]):
        if b["start"] - a["end"] >= min_gap:
            gaps.append((a["end"], b["start"]))
    if duration - segments[-1]["end"] >= min_gap + 3:
        gaps.append((segments[-1]["end"], duration))
    if not gaps:
        return segments, words
    print(f"[CloudWorker] 补转 {len(gaps)} 处超过 {min_gap:.0f} 秒的空白")
    prev_norm = [K.clean_text(s["text"]) for s in segments]
    with tempfile.TemporaryDirectory() as d:
        for i, (a, b) in enumerate(gaps):
            if b - a < 1.0:
                continue
            p = Path(d) / f"g{i}.mp3"
            cut_chunk(audio, a, b, p)
            try:
                gs, gw = groq.transcribe(p, offset=a)
            except K.ServiceStop:
                raise
            except Exception as e:
                print(f"[CloudWorker] 补转跳过 {a:.1f}~{b:.1f}s：{e}")
                continue
            gs, _ = filter_segments(gs)
            gs = [s for s in gs if not K.is_hallucination_or_loop(s["text"], prev_norm)]
            if not gs:
                continue
            words = [w for w in words if not (a <= (w["start"] + w["end"]) / 2 <= b)] + gw
            segments += gs
    segments.sort(key=lambda s: s["start"])
    return segments, words


def analyze_bunsetsu(segments: list) -> list:
    import bunsetsu
    if hasattr(bunsetsu, "get_nlp") and bunsetsu.get_nlp() is None:
        raise RuntimeError("GiNZA 模型未加载，拒绝用规则兜底生成文节")
    return bunsetsu.split_segments(segments)


def preflight():
    """Fail before spending any Groq/DeepSeek quota if a later step cannot run."""
    try:
        import bunsetsu
        nlp = bunsetsu.get_nlp()
        if nlp is None:
            raise RuntimeError("GiNZA 模型未加载")
        bunsetsu.split_segments([{"id": 0, "start": 0.0, "end": 1.0, "text": "今日はいい天気です。", "words": []}])
        K.run(["ffmpeg", "-version"], check=True)
    except Exception as e:
        sys.exit(f"启动自检失败，未消耗任何额度：{type(e).__name__}: {e}")


def parse_itunes_duration(text: str) -> float:
    """'01:02:03', '62:03' or '3723' -> seconds; unknown -> inf."""
    text = (text or "").strip()
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return float("inf")
    if not parts:
        return float("inf")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec if sec > 0 else float("inf")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_episode(item_meta: dict, source: dict, storage: K.Storage, groq: GroqClient, deepseek_key: str,
                    tmp: Path) -> tuple:
    ep_id, title, station = item_meta["id"], item_meta["title"], source.get("name", "")
    orig, cbr = tmp / f"orig_{ep_id}", tmp / f"cbr_{ep_id}.mp3"
    try:
        print(f"[CloudWorker] 下载：{item_meta['source_audio_url']}")
        download_file(item_meta["source_audio_url"], orig)
        audio = convert_to_cbr_mp3(orig, cbr)
        orig.unlink(missing_ok=True)

        # upload first: a moderation rejection must not cost Groq quota
        key = f"audio/{ep_id}-{audio['sha8']}.mp3"
        storage.upload_file(key, cbr, "audio/mpeg", "public, max-age=31536000, immutable")
        audio_url = storage.url(key)

        segments, dropped, hard_cuts = transcribe(cbr, groq, audio["duration"])
        if not segments:
            raise ValueError("转录结果为空")
        last_end = segments[-1]["end"]
        if abs(audio["duration"] - last_end) > 60 and last_end / audio["duration"] < 0.9:
            raise ValueError(f"字幕只覆盖到 {last_end:.0f}s / {audio['duration']:.0f}s")

        translations = K.translate_segments(segments, title, station, deepseek_key)
        overview = K.generate_overview(segments, title, station, deepseek_key)
        grammar = analyze_bunsetsu(segments)
    finally:
        orig.unlink(missing_ok=True)
        cbr.unlink(missing_ok=True)

    content = K.build_content(ep_id, segments, grammar, translations, overview["practice_cards"], dropped)
    meta = {
        "id": ep_id, "title": title, "source_name": station, "source_id": source.get("id", ""),
        "audio_url": audio_url, "audio_sha256": audio["sha256"], "audio_duration": round(audio["duration"], 3),
        "audio_bytes": audio["bytes"], "duration": round(audio["duration"], 3),
        "published": item_meta["published"], "day": item_meta["day"], "image_url": item_meta["image_url"],
        "description": item_meta.get("description", "")[:2000],
        "summary": overview["summary"], "key_points": overview["key_points"],
        "practice_cards": overview["practice_cards"], "content_version": K.CONTENT_VERSION,
        "stats": K.episode_stats(segments, translations, audio["duration"], hard_cuts),
    }
    return meta, content


def iter_candidates(source: dict, cutoff: datetime, now: datetime):
    # catalog entries may list `fallback_urls` (e.g. a Substack publication's own
    # /feed when api.substack.com refuses GitHub's servers)
    root, last = None, None
    for url in [source["url"]] + list(source.get("fallback_urls") or []):
        try:
            root = fetch_feed_xml(url, timeout=15)
            break
        except Exception as e:
            last = e
    if root is None:
        raise last
    channel = root.find("channel")
    channel_img = ""
    if channel is not None:
        node = channel.find("itunes:image", ITUNES_NS)
        channel_img = (node.get("href") if node is not None else "") or (channel.findtext("image/url") or "")

    def pub(item):
        try:
            d = parsedate_to_datetime(item.findtext("pubDate") or "")
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    items = sorted(root.findall("./channel/item"), key=lambda i: pub(i) or datetime.min.replace(tzinfo=timezone.utc),
                   reverse=True)
    for item in items:
        enc = item.find("enclosure")
        url = enc.get("url") if enc is not None else None
        if not url:
            continue
        d = pub(item) or now
        if d < cutoff:
            break
        img = item.find("itunes:image", ITUNES_NS)
        yield {
            "rss_duration": parse_itunes_duration(item.findtext("itunes:duration", default="", namespaces=ITUNES_NS)),
            "id": hashlib.sha256((item.findtext("guid") or url).encode("utf-8")).hexdigest()[:16],
            "title": item.findtext("title") or "无标题",
            "description": re.sub(r"<[^>]+>", " ", item.findtext("itunes:summary", default="", namespaces=ITUNES_NS)
                                  or item.findtext("description") or "")[:4000],
            "source_audio_url": url,
            "published": d.isoformat(),
            "day": d.strftime("%Y-%m-%d"),
            "image_url": ((img.get("href") if img is not None else "") or channel_img),
        }


def run_candidate(cand, source, ctx) -> str:
    """Transcribe, gate and publish one episode. Returns "ok", "fail" or "stop"."""
    state, storage, groq, deepseek_key, tmp, args, reports = ctx
    ep_id = cand["id"]
    print(f"\n[CloudWorker] 新单集：「{cand['title']}」({cand['day']}, {source.get('name')})")
    try:
        meta, content = process_episode(cand, source, storage, groq, deepseek_key, tmp)
    except K.ServiceStop as e:
        # quota / billing / auth problem: not this episode's fault, retry next run
        reports.append((source.get("name"), cand["title"], f"⏸ {e}，下次再处理"))
        ctx_stop[0] = f"服务暂不可用，本次运行停止：{e}"
        return "stop"
    except ImportError as e:
        reports.append((source.get("name"), cand["title"], f"⏸ {e}"))
        ctx_stop[0] = f"运行环境错误：{e}"
        return "stop"
    except K.PermanentFailure as e:
        K.record_failure(state, ep_id, cand["title"], source.get("name"), e, permanent=True)
        K.save_state(storage, state)
        reports.append((source.get("name"), cand["title"], f"✗ 永久失败：{e}"))
        return "fail"
    except Exception as e:
        K.record_failure(state, ep_id, cand["title"], source.get("name"), e)
        K.save_state(storage, state)
        reports.append((source.get("name"), cand["title"], f"✗ 失败：{e}"))
        print(f"[CloudWorker] ❌ {e}")
        return "fail"

    state["episodes"][ep_id] = meta
    result = K.publish_library(storage, state, {ep_id: content}, allow_shrink=args.allow_shrink)
    if ep_id in result["published"]:
        st = meta["stats"]
        reports.append((source.get("name"), cand["title"],
                        f"✓ {meta['audio_duration'] / 60:.1f} 分钟，{st['sentences_count']} 句，翻译 {st['translation_coverage']:.0%}"))
        return "ok"
    reasons = state["episodes"][ep_id].get("quarantined", [])
    del state["episodes"][ep_id]
    K.record_failure(state, ep_id, cand["title"], source.get("name"), "; ".join(reasons))
    K.save_state(storage, state)
    reports.append((source.get("name"), cand["title"], f"✗ 未过门槛：{'; '.join(reasons)}"))
    return "fail"


ctx_stop = [""]


def main():
    ap = argparse.ArgumentParser(description="KOTO 云端转录")
    ap.add_argument("--min-per-station", type=int, default=2,
                    help="保底：每个电台在 App 里至少有这么多集完整单集")
    ap.add_argument("--latest-days", type=float, default=14,
                    help="每日追新：电台最新一期在这么多天内发布的才追（避免给停更电台补旧集）")
    ap.add_argument("--max-episodes", type=int, default=60, help="单次运行最多处理的单集数")
    ap.add_argument("--baseline-candidates", type=int, default=15,
                    help="保底阶段每台在最近这么多集里挑（最短优先），挑不满再往后找")
    ap.add_argument("--episode-id", default="")
    ap.add_argument("--station-id", default="")
    ap.add_argument("--allow-shrink", action="store_true")
    ap.add_argument("--max-runtime-minutes", type=float, default=300,
                    help="停止开始新单集的时间（workflow 超时是 350 分钟）")
    ap.add_argument("--max-groq-wait-minutes", type=float, default=20,
                    help="长时间限流（小时/日额度）最多等待多久，超过就结束本次运行")
    ap.add_argument("--groq-min-interval", type=float, default=3.2,
                    help="两次 Groq 请求之间的最小间隔（秒），免费档每分钟 20 次")
    # accepted for old workflow files; no longer used
    ap.add_argument("--per-station", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--days", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not groq_key or not deepseek_key:
        sys.exit("GROQ_API_KEY 和 DEEPSEEK_API_KEY 都必须设置：没有翻译和逐词时间的单集不会被发布。")

    preflight()
    started = time.time()
    storage = K.Storage()
    state = K.load_state(storage)
    groq = GroqClient(groq_key, max_wait_total=args.max_groq_wait_minutes * 60,
                      min_interval=args.groq_min_interval)
    now = datetime.now(timezone.utc)
    reports = []
    processed = 0

    # Re-check every live episode against the current word list and station list;
    # anything that no longer passes is removed from the app and from COS.
    policy = K.load_policy()
    purged = K.policy_sweep(storage, state, policy)
    if purged:
        K.publish_library(storage, state, {}, allow_shrink=args.allow_shrink, expected_removals=len(purged))
        for ep_id in purged:
            r = state.get("removed", {}).get(ep_id, {})
            reports.append((r.get("station", ""), r.get("title", ep_id), f"🗑 下架：{r.get('reason')} {(r.get('details') or [''])[0]}"))
    else:
        K.save_state(storage, state)

    sources = [s for s in load_curated_sources() if s.get("url")]
    if args.station_id:
        sources = [s for s in sources if s.get("id") == args.station_id]

    feeds = {}

    def feed(source):
        """All RSS episodes of a station, newest first (cached for this run)."""
        if source["id"] not in feeds:
            try:
                feeds[source["id"]] = list(iter_candidates(source, datetime.min.replace(tzinfo=timezone.utc), now))
            except Exception as e:
                print(f"[CloudWorker] RSS 获取失败：{source.get('name')}：{e}")
                reports.append((source.get("name"), "-", f"✗ RSS 获取失败：{e}"))
                feeds[source["id"]] = []
        return feeds[source["id"]]

    tried = set()  # an episode that failed this run is not retried until the next run

    def blocked_by_title(cand, source) -> bool:
        hits = K.policy_hits(policy, {"title": cand["title"], "description": cand.get("description", "")})
        if hits:
            K.record_failure(state, cand["id"], cand["title"], source.get("name"),
                             "CONTENT_POLICY(标题/简介): " + "; ".join(hits), permanent=True)
            reports.append((source.get("name"), cand["title"], f"⛔ 屏蔽词，未下载：{hits[0]}"))
            return True
        return False

    def usable(cand):
        if args.episode_id and cand["id"] != args.episode_id:
            return False
        return (cand["id"] not in tried and cand["id"] not in state["episodes"]
                and not K.is_permanently_failed(state, cand["id"]))

    def budget_left():
        if ctx_stop[0]:
            return False
        if processed >= args.max_episodes:
            ctx_stop[0] = f"已处理 {processed} 集，达到单次上限"
            return False
        if (time.time() - started) / 60 > args.max_runtime_minutes:
            ctx_stop[0] = f"已运行 {args.max_runtime_minutes:.0f} 分钟，停止开始新单集"
            return False
        return True

    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = (state, storage, groq, deepseek_key, Path(tmp_dir), args, reports)

        # Phase 1 - baseline: every station must have `min_per_station` complete
        # episodes in the app. Stations with the fewest go first; within a station
        # the shortest of its recent episodes first (same learning value, less quota),
        # moving further back in the feed until the minimum is met.
        for source in sorted(sources, key=lambda s: K.published_count(state, s)):
            while budget_left() and K.published_count(state, source) < args.min_per_station:
                pool = [c for c in feed(source) if usable(c)]
                if not pool:
                    break
                recent = pool[:args.baseline_candidates]
                cand = min(recent, key=lambda c: c["rss_duration"])
                tried.add(cand["id"])
                if blocked_by_title(cand, source):
                    continue
                if run_candidate(cand, source, ctx) == "ok":
                    processed += 1
            if not budget_left():
                break

        # Phase 2 - daily latest: each station's newest episode, if published within
        # `latest_days` and not in the library yet. No per-station cap.
        latest = []
        scan = state.setdefault("last_scan", {})
        for source in sources:
            items = feed(source)
            # what this run saw, so the acceptance check judges against the same
            # snapshot instead of episodes published while the run was going
            scan[source["id"]] = {"scanned_at": K.now_iso(), "ok": bool(items),
                                  "newest_id": items[0]["id"] if items else "",
                                  "newest_title": items[0]["title"] if items else "",
                                  "newest_published": items[0]["published"] if items else ""}
            if items:
                newest = items[0]
                age_days = (now - datetime.fromisoformat(newest["published"])).total_seconds() / 86400
                if age_days <= args.latest_days and usable(newest):
                    latest.append((newest, source))
        latest.sort(key=lambda x: x[0]["published"], reverse=True)
        for cand, source in latest:
            if not budget_left():
                break
            tried.add(cand["id"])
            if blocked_by_title(cand, source):
                continue
            if run_candidate(cand, source, ctx) == "ok":
                processed += 1

    K.save_state(storage, state)
    stop_reason = ctx_stop[0]
    if stop_reason:
        print(f"[CloudWorker] {stop_reason}")
        if "HTTP 40" in stop_reason or "环境错误" in stop_reason:
            print(f"::error::{stop_reason}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("### KOTO 云端转录\n\n| 电台 | 标题 | 结果 |\n|---|---|---|\n")
            for station, title, res in reports:
                f.write(f"| {station} | {title[:40]} | {res} |\n")
            if stop_reason:
                f.write(f"\n> {stop_reason}\n")
    catalog_ids = K.load_policy()["catalog_ids"]
    live = sum(1 for m in state["episodes"].values() if K.is_published(m) and m.get("source_id") in catalog_ids)
    K.write_step_summary(state.get("quarantine", []), live)
    print(f"[CloudWorker] 完成：新发布 {processed} 集。")


if __name__ == "__main__":
    main()

