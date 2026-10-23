"""Scheduler promises: every station reaches the minimum, and each station's
newest episode is added daily, with no cap and no stale backfill."""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import cloud_transcribe as C  # noqa: E402
import koto_lib as K  # noqa: E402

NOW = datetime.now(timezone.utc)


def ep(eid, days_ago, minutes):
    d = NOW - timedelta(days=days_ago)
    return {"id": eid, "title": eid, "source_audio_url": f"https://x/{eid}.mp3", "published": d.isoformat(),
            "day": d.strftime("%Y-%m-%d"), "image_url": "", "rss_duration": minutes * 60}


class SchedulerTest(unittest.TestCase):
    def run_main(self, sources, feeds, state, failing=()):
        published = []

        def fake_process(cand, source, *a, **k):
            if cand["id"] in failing:
                raise ValueError("boom")
            return {"id": cand["id"], "source_id": source["id"], "source_name": source["name"],
                    "audio_duration": 60, "stats": {"sentences_count": 1, "translation_coverage": 1.0}}, {}

        def fake_publish(storage, st, new, allow_shrink=False, **kw):
            published.extend(new)
            return {"published": list(new), "quarantined": []}

        # GITHUB_STEP_SUMMARY emptied: tests must not write into the real run summary
        with mock.patch.dict("os.environ", {"GROQ_API_KEY": "g", "DEEPSEEK_API_KEY": "d", "GITHUB_STEP_SUMMARY": ""}), \
                mock.patch.object(sys, "argv", ["x"]), \
                mock.patch.object(C, "preflight"), \
                mock.patch.object(C, "load_curated_sources", return_value=sources), \
                mock.patch.object(C, "iter_candidates", side_effect=lambda s, *_: feeds[s["id"]]), \
                mock.patch.object(C, "process_episode", side_effect=fake_process), \
                mock.patch.object(K, "Storage"), \
                mock.patch.object(K, "load_state", return_value=state), \
                mock.patch.object(K, "save_state"), \
                mock.patch.object(K, "publish_library", side_effect=fake_publish), \
                mock.patch.object(K, "policy_sweep", return_value=[]), \
                mock.patch.object(K, "load_policy", return_value={"terms": [], "allowed": [], "hash": "h",
                                                                   "catalog_ids": {x["id"] for x in sources}}):
            C.ctx_stop[0] = ""
            C.main()
        return published

    def state_with(self, eps):
        st = K.empty_state()
        st["episodes"] = {e["id"]: dict(e) for e in eps}
        return st

    def test_empty_station_gets_two_shortest_recent(self):
        src = [{"id": "a", "name": "A", "url": "u"}]
        feeds = {"a": [ep("a1", 40, 90), ep("a2", 50, 20), ep("a3", 60, 30), ep("a4", 70, 10)]}
        got = self.run_main(src, feeds, self.state_with([]))
        self.assertEqual(sorted(got), ["a2", "a4"])  # newest a1 is 40 days old, so no daily add

    def test_newest_is_added_daily_even_when_station_has_many(self):
        src = [{"id": "b", "name": "B", "url": "u"}]
        have = [{"id": f"old{i}", "source_id": "b"} for i in range(5)]
        feeds = {"b": [ep("b-new", 1, 120)] + [ep(f"old{i}", 10 + i, 30) for i in range(5)]}
        self.assertEqual(self.run_main(src, feeds, self.state_with(have)), ["b-new"])

    def test_scan_snapshot_is_recorded_for_the_acceptance_check(self):
        src = [{"id": "f", "name": "F", "url": "u"}]
        st = self.state_with([{"id": "f1", "source_id": "f"}, {"id": "f2", "source_id": "f"}])
        self.run_main(src, {"f": [ep("f1", 1, 10), ep("f2", 2, 10)]}, st)
        self.assertEqual(st["last_scan"]["f"]["newest_id"], "f1")
        self.assertTrue(st["last_scan"]["f"]["ok"])

    def test_stale_newest_is_not_backfilled(self):
        src = [{"id": "c", "name": "C", "url": "u"}]
        have = [{"id": "c1", "source_id": "c"}, {"id": "c2", "source_id": "c"}]
        feeds = {"c": [ep("c3", 30, 20), ep("c1", 40, 20), ep("c2", 50, 20)]}
        self.assertEqual(self.run_main(src, feeds, self.state_with(have)), [])

    def test_failure_moves_on_to_next_candidate(self):
        src = [{"id": "d", "name": "D", "url": "u"}]
        feeds = {"d": [ep("d1", 40, 10), ep("d2", 41, 20), ep("d3", 42, 30)]}
        got = self.run_main(src, feeds, self.state_with([]), failing={"d1"})
        self.assertEqual(sorted(got), ["d2", "d3"])

    def test_pulled_episodes_do_not_count(self):
        src = [{"id": "e", "name": "E", "url": "u"}]
        have = [{"id": "e1", "source_id": "e"},
                {"id": "e2", "source_id": "e", "quarantined": ["FILES_UNREACHABLE audio=False content=True"]}]
        feeds = {"e": [ep("e1", 40, 10), ep("e2", 41, 10), ep("e3", 42, 10)]}
        self.assertEqual(self.run_main(src, feeds, self.state_with(have)), ["e3"])


if __name__ == "__main__":
    unittest.main()

