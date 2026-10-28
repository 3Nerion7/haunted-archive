"""Real local CPU inference invariants, not a new held-out quality benchmark."""
import json
import math
import os
import unittest
from unittest.mock import patch

from archive_game import Archive, digest


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("HAUNTED_MODEL_DIR"):
            raise unittest.SkipTest("Set existing local HAUNTED_MODEL_DIR for real-inference checks")
        for target in ("socket.socket.connect", "socket.create_connection"):
            p = patch(target, side_effect=AssertionError("Network forbidden"))
            p.start()
            cls.addClassCleanup(p.stop)
        cls.archive = Archive()

    def test_real_mean_pooling_padding_and_normalization(self):
        import numpy as np
        text = self.archive.records[0]["text"]
        longer = text + " " + self.archive.records[1]["text"]
        alone = self.archive.embed([text])
        padded = self.archive.embed([text, longer])
        self.assertEqual(alone.shape, (1, 384))
        self.assertEqual(padded.shape, (2, 384))
        self.assertTrue(np.isfinite(padded).all())
        self.assertTrue(np.allclose(np.linalg.norm(padded, axis=1), 1, atol=1e-5))
        self.assertTrue(np.allclose(alone[0], padded[0], atol=2e-5))

    def test_real_retrieval_quotes_and_frozen_out_of_corpus_policy(self):
        by_id = {r["id"]: r for r in self.archive.records}
        for case in self.archive.plan["cases"]:
            result = self.archive.rank(case["query"])
            self.assertNotIn("answer", result)
            self.assertIn("NOT_PROBABILITY", result["score_kind"])
            for hit in result["hits"]:
                text = by_id[hit["source_id"]]["text"]
                self.assertEqual(text[hit["start"]:hit["end"]], hit["quote"])
                self.assertEqual(digest(text.encode()), hit["text_sha256"])
                self.assertTrue(math.isfinite(hit["score"]))
            if case["id"] == "out-of-corpus":
                self.assertTrue(result["abstained"])
                self.assertEqual(result["hits"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
