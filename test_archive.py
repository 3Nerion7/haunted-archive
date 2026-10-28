"""Mechanical regressions; fake embeddings here are not inference-quality evidence."""
import copy
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

from archive_game import ROOT, Archive, ArchiveError, citation, conflicts, digest, load_corpus


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("corpus.json", "EVALUATION-PLAN.json", "PRE-INFERENCE-FREEZE.json", "MODEL-MANIFEST.json"):
            shutil.copyfile(ROOT / name, self.root / name)
        self.archive = Archive(self.root)
        for target in ("socket.socket.connect", "socket.create_connection"):
            p = patch(target, side_effect=AssertionError("No network in this test"))
            p.start()
            self.addCleanup(p.stop)

    def test_literal_citations_and_provenance(self):
        result = self.archive.rank("north lantern midnight", "lexical")
        self.assertTrue(result["hits"])
        sources = {r["id"]: r for r in self.archive.records}
        for hit in result["hits"]:
            text = sources[hit["source_id"]]["text"]
            self.assertEqual(hit["quote"], text[hit["start"]:hit["end"]])
            self.assertEqual(hit["text_sha256"], digest(text.encode()))
            self.assertIn("fictional", hit["provenance"])

    def test_unknown_information_abstains_without_generated_text(self):
        result = self.archive.rank("xylophone quasar hyperdrive", "lexical")
        self.assertTrue(result["abstained"])
        self.assertEqual(result["hits"], [])
        self.assertNotIn("answer", result)

    def test_source_tamper_at_start_is_rejected(self):
        (self.root / "corpus.json").write_text("[]", encoding="utf-8")
        with self.assertRaises(ArchiveError):
            Archive(self.root)

    def test_mid_session_source_tamper_is_rejected_before_model(self):
        (self.root / "corpus.json").write_text("[]", encoding="utf-8")
        with self.assertRaises(ArchiveError):
            self.archive.rank("lantern")

    def test_policy_change_is_not_silent_threshold_tuning(self):
        (self.root / "EVALUATION-PLAN.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ArchiveError):
            Archive(self.root)

    def test_invalid_claim_quote_and_duplicate_source_rejected(self):
        for transform in (lambda r: r[0]["claims"][0].update(quote="invented statement"),
                          lambda r: r.append(copy.deepcopy(r[0]))):
            records = copy.deepcopy(self.archive.records)
            transform(records)
            data = json.dumps(records).encode()
            with self.assertRaises(ArchiveError):
                load_corpus(data, digest(data))

    def test_authored_conflict_is_not_a_model_truth_decision(self):
        selected = self.archive.records[:2]
        result = conflicts(selected)
        self.assertEqual(len(result), 1)
        self.assertIn("AUTHOR_SUPPLIED", result[0]["basis"])
        self.assertEqual(set(result[0]["values"]), {"dark", "shining"})
        self.assertEqual(conflicts([selected[0], selected[0]]), [])

    def test_notebook_has_two_reader_chosen_sources_and_unverified_hypothesis(self):
        mocked_ranking = self.archive.rank("lantern", "lexical")
        hypothesis = "FICTIONAL TEST INPUT: I have not decided which account to trust."
        with patch.object(self.archive, "rank", return_value=mocked_ranking):
            note = self.archive.notebook("lantern", hypothesis, ["harbor-a", "harbor-b"])
        self.assertEqual(note["hypothesis"], hypothesis)
        self.assertEqual(note["personal_review"], "NOT_CLAIMED")
        self.assertEqual(note["supports_hypothesis"], "NOT_AUTOMATICALLY_DETERMINED")
        self.assertEqual(len(note["selected_evidence"]), 2)
        self.assertEqual(len(note["authored_conflicts"]), 1)
        self.assertNotIn("answer", note)

    def test_invalid_notebook_selections_rejected_before_inference(self):
        for selected in (["harbor-a"], ["harbor-a", "harbor-a"], ["unknown", "voices"]):
            with self.assertRaises(ArchiveError):
                self.archive.notebook("lantern", "fictional hypothesis", selected)

    def test_semantic_threshold_can_abstain_with_controlled_test_vectors(self):
        import numpy as np
        n = len(self.archive.records)
        self.archive.vectors = np.eye(n)
        self.archive.embed = lambda texts: np.zeros((len(texts), n))
        result = self.archive.rank("mechanical zero-vector fixture")
        self.assertTrue(result["abstained"])
        self.assertEqual(result["threshold"], 0.40)

    def test_empty_or_oversize_query_rejected_before_model(self):
        for query in ("", "  ", "x" * 1001):
            with self.assertRaises(ArchiveError):
                self.archive.rank(query)

    def test_manifest_tamper_rejected_before_loading_weights(self):
        (self.root / "MODEL-MANIFEST.json").write_text("{}", encoding="utf-8")
        with patch.dict("os.environ", {"HAUNTED_MODEL_DIR": str(self.root)}):
            with self.assertRaises(ArchiveError):
                self.archive.setup_model()


if __name__ == "__main__":
    unittest.main(verbosity=2)
