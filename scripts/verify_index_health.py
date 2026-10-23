#!/usr/bin/env python3
"""Acceptance check for what the app actually shows (the live index.json).

Fails (exit 1) when any of the promises is broken:
  1. every episode in the app passes the publish gate (audio on our COS,
     complete subtitles, complete translation, grammar, clean timing, cards);
  2. every catalog station has at least --min-per-station episodes;
  3. every station's newest RSS episode published within --latest-days is in
     the app (unless --skip-latest).

  python scripts/verify_index_health.py
"""
import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import koto_lib as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=f"{K.COS_BASE_URL}/{K.INDEX_KEY}")
    ap.add_argument("--min-per-station", type=int, default=2)
    ap.add_argument("--latest-days", type=float, default=14)
    ap.add_argument("--skip-latest", action="store_true")
    args = ap.parse_args()

    index = K.http_get_json(args.index) if args.index.startswith("http") else \
        json.loads(Path(args.index).read_text(encoding="utf-8"))
    episodes = index.get("episodes", [])
    storage = K.Storage()
    problems = []

    # 1. every episode passes the gate
    bad = 0
    for ep in episodes:
        try:
            reasons = K.validate_episode(ep, K.http_get_json(ep.get("content_url", ""), timeout=180), storage)
        except Exception as e:
            reasons = [f"CONTENT_FETCH_FAILED {e}"]
        if reasons:
            bad += 1
            problems.append(f"不合格单集 {ep['id']}「{ep.get('title', '')[:30]}」：{'; '.join(reasons)}")
    print(f"[验收] App 内 {len(episodes)} 集，不合格 {bad} 集")

    # 2. station minimum
    catalog = json.loads((K.REPO_ROOT / "source_catalog.json").read_text(encoding="utf-8"))
    by_station = collections.Counter(e.get("source_id") for e in episodes)
    short = [(s, by_station.get(s["id"], 0)) for s in catalog if by_station.get(s["id"], 0) < args.min_per_station]
    for s, n in short:
        problems.append(f"电台不足 {args.min_per_station} 集：{s['name']}（{s['id']}）只有 {n} 集")
    print(f"[验收] {len(catalog)} 个电台，不足 {args.min_per_station} 集的 {len(short)} 个")

    # 3. newest episode of every station, judged against what the run itself saw
    #    when it scanned the feeds (state["last_scan"]). Episodes published while
    #    the run was going are the next run's job, not a failure of this one.
    if not args.skip_latest:
        state = storage.get_private_json(K.STATE_KEY) or K.empty_state()
        scan = state.get("last_scan", {})
        live_ids = {e["id"] for e in episodes}
        now = datetime.now(timezone.utc)
        missing = 0
        for s in catalog:
            seen = scan.get(s["id"])
            if not seen or not seen.get("ok"):
                problems.append(f"本次运行没能读到 RSS：{s['name']}")
                continue
            if (now - datetime.fromisoformat(seen["scanned_at"])).total_seconds() > 2 * 86400:
                problems.append(f"RSS 扫描记录超过 2 天：{s['name']}（上次 {seen['scanned_at'][:16]}）")
                continue
            ep_id = seen["newest_id"]
            failure = state["failures"].get(ep_id, {})
            if "CONTENT_POLICY" in str(failure.get("last_error", "")) or ep_id in state.get("removed", {}):
                continue  # intentionally excluded by the blocked-word list
            age = (now - datetime.fromisoformat(seen["newest_published"])).total_seconds() / 86400
            if age <= args.latest_days and ep_id not in live_ids:
                missing += 1
                why = failure.get("last_error", "未处理（额度或运行时间用完）")
                problems.append(f"缺最新一期：{s['name']}「{seen['newest_title'][:30]}」：{why[:150]}")
        print(f"[验收] 扫描时各台的最新一期（{args.latest_days:.0f} 天内）缺 {missing} 个")

    if problems:
        print("\n".join("  ✗ " + p for p in problems))
        print(f"::error::验收未通过：{len(problems)} 项问题（详见日志）")
        sys.exit(1)
    print("[验收] 全部通过")


if __name__ == "__main__":
    main()

