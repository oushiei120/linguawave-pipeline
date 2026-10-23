"""Blocked-word list: any hit keeps an episode out of the app, and a live
episode that starts hitting (word list or station list changed) is deleted."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import koto_lib as K  # noqa: E402


def policy(terms, allowed=(), catalog=("ok-station",)):
    d = Path(tempfile.mkdtemp())
    (d / "p.json").write_text(json.dumps({"blocked_terms": {"测试": list(terms)},
                                          "allowed_phrases": list(allowed)}, ensure_ascii=False))
    (d / "c.json").write_text(json.dumps([{"id": c} for c in catalog]))
    with mock.patch.object(K, "CATALOG_FILE", d / "c.json"):
        return K.load_policy(d / "p.json")


def content(lines, translations=None):
    return {"transcript": {"raw": {"segments": [{"id": i, "text": t} for i, t in enumerate(lines)]}},
            "translation": {"segments": translations or {}}, "interpretation": {"cards": []}}


class PolicyTest(unittest.TestCase):
    def test_width_spacing_and_case_variants_match(self):
        p = policy(["テロ事件", "Xi Jinping"])
        self.assertTrue(K.policy_hits(p, {"title": "ﾃﾛ 事件について"}))
        self.assertTrue(K.policy_hits(p, {"title": "xi-jinping said"}))

    def test_latin_terms_match_whole_words_only(self):
        p = policy(["ISIS"])
        self.assertFalse(K.policy_hits(p, {"description": "Hi, this is a Japanese podcast"}))
        self.assertTrue(K.policy_hits(p, {"description": "news about ISIS today"}))
        self.assertTrue(K.policy_hits(p, {"description": "ＩＳＩＳ"}))  # full width

    def test_term_split_across_two_subtitle_lines(self):
        p = policy(["習近平"])
        self.assertTrue(K.policy_hits(p, {"title": "雑談"}, content(["昨日は習", "近平の話を"])))

    def test_translation_and_cards_are_checked(self):
        p = policy(["共产党"])
        self.assertTrue(K.policy_hits(p, {"title": "x"}, content(["こんにちは"], {"0": "关于共产党的话题"})))

    def test_allowed_phrase_is_not_a_hit(self):
        p = policy(["大麻"], allowed=["大麻比古神社"])
        self.assertFalse(K.policy_hits(p, {"title": "大麻比古神社に行った"}))
        self.assertTrue(K.policy_hits(p, {"title": "大麻の話"}))

    def test_clean_episode_has_no_hits(self):
        p = policy(["習近平"])
        self.assertFalse(K.policy_hits(p, {"title": "ラーメンの話"}, content(["美味しかった"], {"0": "很好吃"})))

    def test_sweep_deletes_files_and_blocks_readding(self):
        p = policy(["共産党"], catalog=("ok-station",))
        st = K.empty_state()
        base = K.COS_BASE_URL
        st["episodes"] = {
            "bad": {"id": "bad", "title": "共産党の話", "source_id": "ok-station",
                    "audio_url": f"{base}/audio/bad-1.mp3", "content_url": f"{base}/episodes/bad.x.json"},
            "gone": {"id": "gone", "title": "ニュース", "source_id": "removed-station",
                     "audio_url": f"{base}/audio/gone-1.mp3", "content_url": f"{base}/episodes/gone.x.json"},
            "fine": {"id": "fine", "title": "雑談", "source_id": "ok-station",
                     "audio_url": f"{base}/audio/fine-1.mp3", "content_url": f"{base}/episodes/fine.x.json"},
        }
        storage = mock.Mock()
        with mock.patch.object(K, "http_get_json", return_value=content(["普通の話"])):
            purged = K.policy_sweep(storage, st, p)
        self.assertEqual(sorted(purged), ["bad", "gone"])
        deleted = sorted(c.args[0] for c in storage.delete.call_args_list)
        self.assertEqual(deleted, ["audio/bad-1.mp3", "audio/gone-1.mp3", "episodes/bad.x.json", "episodes/gone.x.json"])
        self.assertEqual(list(st["episodes"]), ["fine"])
        self.assertTrue(K.is_permanently_failed(st, "bad"))
        self.assertEqual(st["episodes"]["fine"]["policy_hash"], p["hash"])

    def test_sweep_skips_episodes_already_checked_with_same_list(self):
        p = policy(["共産党"])
        st = K.empty_state()
        st["episodes"] = {"a": {"id": "a", "title": "共産党", "source_id": "ok-station", "policy_hash": p["hash"]}}
        self.assertEqual(K.policy_sweep(mock.Mock(), st, p), [])

    def test_catalog_has_no_news_stations(self):
        p = policy(["x"], catalog=[c["id"] for c in json.loads(K.CATALOG_FILE.read_text(encoding="utf-8"))])
        self.assertFalse({"nhk", "tbs-news", "news-connect", "bbc-6-minute-english"} & p["catalog_ids"])

    def test_policy_refuses_to_run_without_word_list(self):
        K._POLICY_CACHE.clear()
        with mock.patch.object(K.Storage, "get_private_json", return_value=None):
            with self.assertRaises(RuntimeError):
                K.load_policy()

    def test_private_state_never_falls_back_on_read_errors(self):
        class Boom(Exception):
            pass
        storage = mock.Mock()
        storage.get_private_json.side_effect = Boom("403")
        with self.assertRaises(Boom):
            K.load_state(storage)


if __name__ == "__main__":
    unittest.main()

