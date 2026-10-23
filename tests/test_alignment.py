import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import koto_lib as K  # noqa: E402


def W(word, start, end):
    return {"word": word, "start": start, "end": end}


class AssignWordsTest(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"start": 0.5, "end": 2.5, "text": "おはようございます"},
            {"start": 2.8, "end": 4.5, "text": "今日もいい天気ですね"},
            {"start": 5.0, "end": 7.2, "text": "張り切っていきましょう"},
        ]
        self.words = [W("おはよう", 0.5, 1.5), W("ございます", 1.5, 2.5), W("今日", 2.8, 3.3), W("も", 3.3, 3.5),
                      W("いい", 3.5, 3.8), W("天気", 3.8, 4.2), W("ですね", 4.2, 4.5),
                      W("張り切って", 5.0, 5.8), W("いき", 5.8, 6.4), W("ましょう", 6.4, 7.2)]

    def test_each_word_belongs_to_one_sentence(self):
        segs = K.assign_words(copy.deepcopy(self.segments), self.words)
        self.assertEqual([len(s["words"]) for s in segs], [2, 5, 3])
        rep = K.timing_report(segs)
        self.assertEqual((rep["shared_words"], rep["overlaps"], rep["inverted"]), (0, 0, 0))

    def test_dropped_middle_sentence_does_not_shift_neighbours(self):
        segs = K.assign_words([copy.deepcopy(self.segments[0]), copy.deepcopy(self.segments[2])], self.words)
        self.assertEqual((segs[1]["start"], segs[1]["end"]), (5.0, 7.2))
        self.assertEqual(segs[1]["words"][0]["word"], "張り切って")
        self.assertEqual(segs[0]["end"], 2.5)  # words of the dropped sentence are discarded

    def test_first_word_after_pause_is_clamped(self):
        segs = [{"start": 30.0, "end": 39.1, "text": "理学研究科"}]
        words = [W("理", 30.0, 35.38), W("学", 35.38, 35.62), W("研", 35.62, 35.96)]
        out = K.assign_words(segs, words)
        self.assertGreater(out[0]["start"], 35.0)
        self.assertEqual(K.timing_report(out)["long_first_word_ratio"], 0)

    def test_duplicate_gap_fill_words_are_removed(self):
        segs = K.assign_words(copy.deepcopy(self.segments), self.words + [W("今日", 2.8, 3.3)])
        self.assertEqual(len(segs[1]["words"]), 5)


class LegacyRepairTest(unittest.TestCase):
    def test_shared_boundary_word_stays_with_the_sentence_that_ends_with_it(self):
        segs = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "そうですね", "words": [W("そう", 0.0, 1.0), W("ですね", 1.0, 2.0)]},
            {"id": 1, "start": 1.0, "end": 4.0, "text": "次の話です",
             "words": [W("ですね", 1.0, 2.0), W("次", 2.2, 2.6), W("の話です", 2.6, 4.0)]},
        ]
        out = K.repair_legacy_words(segs)
        self.assertEqual([s["id"] for s in out], [0, 1])
        self.assertEqual(out[1]["words"][0]["word"], "次")
        self.assertEqual(out[1]["start"], 2.2)
        self.assertEqual(K.timing_report(out)["shared_words"], 0)

    def test_ids_and_order_are_kept(self):
        segs = [{"id": 0, "start": 5.0, "end": 6.0, "text": "あ", "words": []},
                {"id": 1, "start": 4.0, "end": 7.0, "text": "いいえ", "words": []}]
        out = K.repair_legacy_words(segs)
        self.assertEqual([s["id"] for s in out], [0, 1])
        self.assertGreaterEqual(out[1]["start"], out[0]["end"])


class ChunkPlanTest(unittest.TestCase):
    def test_split_in_the_middle_of_silence(self):
        chunks, hard = K.plan_chunks([(49.0, 51.0), (99.0, 101.0)], 160.0, target=50.0)
        self.assertEqual((chunks[0]["start"], chunks[0]["end"]), (0.0, 50.0))
        self.assertEqual(chunks[1]["keep_end"], 100.0)
        self.assertEqual(hard, 0)

    def test_hard_cut_overlaps_and_ownership_is_unique(self):
        chunks, hard = K.plan_chunks([], 200.0, target=90.0)
        self.assertGreater(hard, 0)
        a, b = chunks[0], chunks[1]
        self.assertEqual(a["keep_end"], b["keep_start"])
        self.assertGreater(a["end"], a["keep_end"])      # audio extends past the cut
        self.assertLess(b["start"], b["keep_start"])
        word = {"start": a["keep_end"] - 0.2, "end": a["keep_end"] + 0.4}  # straddles the cut
        self.assertEqual(sum(K.owned(word, c) for c in chunks), 1)

    def test_last_word_of_episode_is_owned(self):
        chunks, _ = K.plan_chunks([], 60.0)
        self.assertTrue(K.owned({"start": 59.5, "end": 60.0}, chunks[-1]))


class TranslationTest(unittest.TestCase):
    def segs(self, n):
        return [{"id": i, "text": f"文{i}", "start": i, "end": i + 0.5} for i in range(n)]

    def test_renumbered_reply_is_rejected_and_never_overwrites_other_ids(self):
        calls = []

        def fake(prompt, key, timeout=120):
            calls.append(prompt)
            n = prompt.count("\n[")
            if len(calls) == 1:
                return {"0": "错", "1": "错"}  # wrong numbering, like the old bug
            return {str(i + 1): f"译{i + 1}" for i in range(n)}

        with mock.patch.object(K, "deepseek_json", side_effect=fake), mock.patch.object(K.time, "sleep"):
            out = K.translate_segments(self.segs(3), "t", "s", "k", only_ids={1, 2})
        self.assertEqual(set(out), {"1", "2"})
        self.assertNotIn("0", out)

    def test_billing_error_stops_immediately(self):
        fake = mock.Mock(side_effect=K.AccountProblem("DeepSeek HTTP 402"))
        with mock.patch.object(K, "deepseek_json", fake), mock.patch.object(K.time, "sleep"):
            with self.assertRaises(K.ServiceStop):
                K.translate_segments(self.segs(40), "t", "s", "k")
            with self.assertRaises(K.ServiceStop):
                K.generate_overview(self.segs(40), "t", "s", "k")
        self.assertEqual(fake.call_count, 2)  # one call each: no retries, no batch splitting

    def test_persistent_failure_raises(self):
        with mock.patch.object(K, "deepseek_json", return_value={"x": 1}), mock.patch.object(K.time, "sleep"):
            with self.assertRaises(K.TranslationError):
                K.translate_segments(self.segs(4), "t", "s", "k")


class CardRepairTest(unittest.TestCase):
    def test_card_is_moved_to_the_sentence_it_quotes(self):
        segs = [{"id": 0, "text": "こんにちは", "start": 0, "end": 1},
                {"id": 1, "text": "地球は物理で解けるのか", "start": 1, "end": 3}]
        cards = K.repair_cards([{"segment_id": 0, "ja_original": "地球は物理で解けるのか"},
                                {"segment_id": 1, "ja_original": "まったく関係ない文章です"}], segs)
        self.assertEqual(len(cards), 1)
        self.assertEqual((cards[0]["segment_id"], cards[0]["start"]), (1, 1))


if __name__ == "__main__":
    unittest.main()

