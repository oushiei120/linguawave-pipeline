#!/usr/bin/env python3
"""Upgrade already-published episodes to the v4 format without calling Groq.

For each episode (default: every episode still marked legacy):
  * downloads the audio to measure its real sha256 / bytes / duration
  * repairs word timing from the saved Whisper words (no shared words, no
    first word swallowing the preceding pause, no overlaps)
  * re-translates only what is broken: missing ids, and the first batch when
    the last batch's translations were written over it (renumbered ids)
  * regenerates summary / key points / cards from the whole episode
  * keeps the GiNZA grammar (it was analysed on `text`) and fixes grammar_texts
Episodes that still fail the gate stay as they are (legacy) and are reported.

  python scripts/rebuild_library.py --dry-run out/      # no uploads, no DeepSeek
  python scripts/rebuild_library.py                       # needs COS + DeepSeek keys
"""
import argparse
import copy
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import koto_lib as K  # noqa: E402


def measure_audio(url: str) -> dict:
    h = hashlib.sha256()
    n = 0
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        req = urllib.request.Request(url, headers={"User-Agent": K.UA})
        with urllib.request.urlopen(req, timeout=120) as resp:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                f.write(chunk)
                n += len(chunk)
        f.flush()
        return {"sha256": h.hexdigest(), "bytes": n, "duration": K.ffprobe_duration(f.name)}


def ids_to_retranslate(segments: list, translations: dict, batch_size: int = 45) -> set:
    ids = [s["id"] for s in segments]
    missing = {i for i in ids if not (translations.get(str(i)) or "").strip()}
    need = set(missing)
    tail = len(ids) % batch_size
    # old bug: a short last batch came back numbered 0..k and overwrote the first sentences
    if tail and missing and set(ids[-tail:]) <= missing:
        need |= set(ids[:batch_size])
    return need


def rebuild_one(meta: dict, storage: K.Storage, deepseek_key: str, skip_ai: bool) -> tuple:
    ep_id = meta["id"]
    content = K.http_get_json(meta["content_url"], timeout=180)
    tr = content.get("transcript", {})
    old = tr.get("raw", {}).get("segments", [])
    if not old:
        raise ValueError("no segments")
    if [s.get("id") for s in old] != list(range(len(old))):
        raise ValueError("published segment ids are not 0..n-1; translations/grammar cannot be trusted")
    # ids and order are preserved, so translations and grammar keyed by id stay valid
    segments = K.repair_legacy_words(copy.deepcopy(old))

    translations = dict(content.get("translation", {}).get("segments", {}))
    need = ids_to_retranslate(segments, translations)
    cards = content.get("interpretation", {}).get("cards", [])
    summary, key_points = meta.get("summary", ""), meta.get("key_points", [])
    if skip_ai:
        cards = K.repair_cards(cards, segments)
    else:
        if need:
            print(f"    重新翻译 {len(need)} 句")
            translations.update(K.translate_segments(segments, meta["title"], meta["source_name"], deepseek_key,
                                                     only_ids=need))
        ov = K.generate_overview(segments, meta["title"], meta["source_name"], deepseek_key)
        summary, key_points, cards = ov["summary"], ov["key_points"], ov["practice_cards"]

    audio = measure_audio(meta["audio_url"])
    grammar = tr.get("grammar", {}).get("segments", [])
    new_content = K.build_content(ep_id, segments, grammar, translations, cards, tr.get("dropped", []))
    new_meta = {k: v for k, v in meta.items() if k not in ("legacy", "content_url", "content_sha", "quarantined")}
    new_meta.update(
        audio_sha256=audio["sha256"], audio_bytes=audio["bytes"], audio_duration=round(audio["duration"], 3),
        duration=round(audio["duration"], 3), summary=summary, key_points=key_points, practice_cards=cards,
        content_version=K.CONTENT_VERSION,
        stats=K.episode_stats(segments, translations, audio["duration"]),
    )
    return new_meta, new_content, {"retranslated": len(need), "skipped_ai": skip_ai}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", metavar="DIR", help="write files to DIR instead of COS")
    ap.add_argument("--ids", default="", help="comma-separated episode ids (default: all legacy)")
    ap.add_argument("--skip-ai", action="store_true", help="do not call DeepSeek (dry-run inspection)")
    ap.add_argument("--allow-shrink", action="store_true")
    args = ap.parse_args()

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_key and not args.skip_ai:
        sys.exit("DEEPSEEK_API_KEY missing (or pass --skip-ai)")

    storage = K.Storage(Path(args.dry_run) if args.dry_run else None)
    state = K.load_state(storage)
    wanted = [i for i in args.ids.split(",") if i] or [i for i, m in state["episodes"].items() if m.get("legacy")]
    print(f"[Rebuild] 待重建 {len(wanted)} 集")

    upgraded, failed, stop = {}, [], ""
    for ep_id in wanted:
        meta = state["episodes"].get(ep_id)
        if not meta:
            print(f"[Rebuild] 跳过未知 id {ep_id}")
            continue
        print(f"[Rebuild] {ep_id}「{meta.get('title', '')[:30]}」")
        try:
            new_meta, content, info = rebuild_one(meta, storage, deepseek_key, args.skip_ai)
        except K.ServiceStop as e:
            stop = str(e)
            print(f"    ⏸ 停止重建：{e}")
            break
        except Exception as e:
            failed.append((ep_id, meta.get("title", ""), [f"REBUILD_ERROR {e}"]))
            print(f"    ❌ {e}")
            continue
        # validate before touching state: a failing rebuild must not take a live episode down
        probe = dict(new_meta, content_url="", content_sha=K.content_sha(content))
        reasons = K.validate_episode(probe, content, storage)
        st = new_meta["stats"]
        print(f"    时间：首词过长 {st['long_first_word_ratio']:.2%}，重叠 {st['overlaps']}，共用词 {st['shared_words']}；"
              f"翻译 {st['translation_coverage']:.1%}；卡片 {len(content['interpretation']['cards'])}")
        if reasons:
            failed.append((ep_id, meta.get("title", ""), reasons))
            print(f"    ⚠️ 未过门槛，保留线上旧版：{reasons}")
            continue
        state["episodes"][ep_id] = new_meta
        upgraded[ep_id] = content

    if upgraded:
        K.publish_library(storage, state, upgraded, allow_shrink=args.allow_shrink, summary=True)
    print(f"\n[Rebuild] 升级 {len(upgraded)} 集，保留旧版 {len(failed)} 集")
    for ep_id, title, reasons in failed:
        print(f"  - {ep_id} {title[:30]}: {'; '.join(reasons)}")
    if stop:
        print(f"::error::重建中断：{stop}。线上单集保持原样，问题解决后下次定时任务会继续。")
        sys.exit(1)
    if failed and not args.skip_ai:
        sys.exit(1)


if __name__ == "__main__":
    main()

