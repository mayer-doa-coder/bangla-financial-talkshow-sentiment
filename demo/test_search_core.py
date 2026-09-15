from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from search_core import CorpusIndex, normalize_text


class SearchCoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        bundle = root / "member_a"
        (bundle / "data/fused").mkdir(parents=True)
        (bundle / "data/raw").mkdir(parents=True)
        with (bundle / "data/raw/registry.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["episode_id", "duration_sec", "programme", "channel", "local_path"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "episode_id": "ep001",
                    "duration_sec": "60",
                    "programme": "RTV Business Talk",
                    "channel": "RTV Business",
                    "local_path": "",
                }
            )
        payload = {
            "episode_id": "ep001",
            "diarization_provider": "pyannote",
            "asr_model": "test-whisper",
            "stats": {"n_words": 12, "unassigned_words": 0},
            "utterances": [
                {
                    "utterance_id": "ep001_utt00000",
                    "speaker_id": 0,
                    "start_sec": 0,
                    "end_sec": 10,
                    "text_verbatim": "ব্যাঙ্ক খাত থেকে সরকার ঋণ নিয়েছে",
                    "text_normalized": "ব্যাঙ্ক খাত থেকে সরকার ঋণ নিয়েছে",
                    "words": [{"word": "ব্যাঙ্ক", "prob": 0.9}],
                },
                {
                    "utterance_id": "ep001_utt00001",
                    "speaker_id": 1,
                    "start_sec": 11,
                    "end_sec": 20,
                    "text_verbatim": "রেমিট্যান্স বৈদেশিক মুদ্রার প্রধান উৎস",
                    "text_normalized": "রেমিট্যান্স বৈদেশিক মুদ্রার প্রধান উৎস",
                    "words": [{"word": "রেমিট্যান্স", "prob": 0.8}],
                },
            ],
        }
        (bundle / "data/fused/ep001.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        self.index = CorpusIndex(root)

    def tearDown(self):
        self.temp.cleanup()

    def test_normalizes_bank_spelling(self):
        self.assertEqual(normalize_text("ব্যাঙ্ক"), normalize_text("ব্যাংক"))

    def test_normalizes_financial_search_aliases(self):
        self.assertEqual(normalize_text("রেমিটেন্স"), normalize_text("রেমিট্যান্স"))
        self.assertEqual(normalize_text("inflation"), normalize_text("মূল্যস্ফীতি"))

    def test_discovers_namespaced_episode(self):
        self.assertEqual(self.index.global_episodes, ["member_a::ep001"])
        self.assertEqual(self.index.corpus_summary()["episodes"], 1)

    def test_duplicate_member_episode_ids_stay_separate(self):
        root = Path(self.temp.name)
        shutil.copytree(root / "member_a", root / "member_b")
        combined = CorpusIndex(root)
        self.assertEqual(
            combined.global_episodes,
            ["member_a::ep001", "member_b::ep001"],
        )
        hits = combined.search(
            "ব্যাংক", query_type="Word", method="Exact", minimum_score=0.5
        )
        self.assertEqual({hit.collection_id for hit in hits}, {"member_a", "member_b"})

    def test_exact_word_search_returns_speaker(self):
        hits = self.index.search("ব্যাংক", query_type="Word", method="Exact", minimum_score=0.5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].speaker_id, 0)
        self.assertEqual(hits[0].global_episode_id, "member_a::ep001")

    def test_episode_and_speaker_scope(self):
        hits = self.index.search(
            "রেমিট্যান্স",
            query_type="Word",
            method="Exact",
            episode="member_a::ep001",
            speaker="1",
            minimum_score=0.5,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].speaker_id, 1)

    def test_fuzzy_phrase_search(self):
        hits = self.index.search(
            "ব্যাংক খাতের সংকট",
            query_type="Sentence",
            method="Fuzzy",
            minimum_score=0.45,
        )
        self.assertTrue(hits)
        self.assertEqual(hits[0].speaker_id, 0)

    def test_roll_number_layout_links_owner_audio_and_reports_individual_work(self):
        source = Path(self.temp.name) / "member_a"
        team_root = Path(self.temp.name) / "roll_layout"
        roll_root = team_root / "2107006"
        shutil.copytree(source, roll_root / "2107006_results")
        (roll_root / "01.mp3").write_bytes(b"test-audio-placeholder")

        pending_roll = team_root / "2107009"
        pending_roll.mkdir(parents=True)
        (pending_roll / "06.mp3").write_bytes(b"pending-audio-placeholder")

        index = CorpusIndex(team_root)

        self.assertEqual(index.collections, ["2107006"])
        self.assertEqual(index.global_episodes, ["2107006::ep001"])
        self.assertTrue(index.episodes["2107006::ep001"].audio_path.endswith("01.mp3"))
        self.assertEqual(index.corpus_summary()["contributors"], 2)
        self.assertEqual(index.corpus_summary()["source_audio_files"], 2)

        rows = {row["Roll / member"]: row for row in index.contributor_rows()}
        self.assertEqual(rows["2107006"]["Processed episodes"], 1)
        self.assertEqual(rows["2107006"]["Audio linked"], 1)
        self.assertEqual(rows["2107009"]["Processed episodes"], 0)
        self.assertEqual(rows["2107009"]["Bundle"], "Not available")


if __name__ == "__main__":
    unittest.main()
