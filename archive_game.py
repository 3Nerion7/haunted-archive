"""Fictional clue retrieval + a reader's evidence notebook; never a truth generator."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
STOP = set("a an and are at did do does for how in is it of on or s the to was what where which who with".split())


class ArchiveError(ValueError):
    pass


def check(ok, message):
    if not ok:
        raise ArchiveError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_corpus(data, expected_hash):
    check(digest(data) == expected_hash, "Corpus changed after pre-inference freeze")
    records = json.loads(data)
    check(isinstance(records, list) and 1 <= len(records) <= 20, "Corpus size invalid")
    ids = set()
    for r in records:
        check(isinstance(r, dict) and isinstance(r.get("id"), str) and r["id"] not in ids, "Duplicate/invalid source")
        ids.add(r["id"])
        check(all(isinstance(r.get(k), str) and r[k].strip() for k in ("title", "text", "provenance")), "Source/provenance missing")
        check(len(r["text"]) <= 2000 and isinstance(r.get("claims"), list), "Source limit/claims invalid")
        for c in r["claims"]:
            check(all(isinstance(c.get(k), str) and c[k] for k in ("topic", "value", "quote")), "Annotation invalid")
            check(r["text"].count(c["quote"]) == 1, "Authored claim must quote one exact source span")
    return records


def citation(record, score=None):
    text = record["text"]
    return {"source_id": record["id"], "title": record["title"], "provenance": record["provenance"],
            "quote": text, "start": 0, "end": len(text), "text_sha256": digest(text.encode()), "score": score}


def conflicts(records):
    found = []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            for x in a["claims"]:
                for y in b["claims"]:
                    if x["topic"] == y["topic"] and x["value"] != y["value"]:
                        found.append({"topic": x["topic"], "sources": [a["id"], b["id"]],
                                      "values": [x["value"], y["value"]],
                                      "basis": "AUTHOR_SUPPLIED_FICTIONAL_ANNOTATIONS_NOT_MODEL_INFERENCE"})
    return found


def lexical_scores(query, texts):
    tokens = lambda text: [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]
    docs = [Counter(tokens(text)) for text in texts]
    df = Counter(word for doc in docs for word in doc)
    weights = {w: 1 + math.log((len(docs) + 1) / (n + 1)) for w, n in df.items()}
    vector = lambda words: {w: n * weights[w] for w, n in Counter(words).items() if w in weights}
    q = vector(tokens(query))
    norm = lambda v: math.sqrt(sum(n * n for n in v.values()))
    return [sum(q.get(w, 0) * n * weights[w] for w, n in doc.items()) /
            (norm(q) * norm(vector(doc.elements())) or 1) for doc in docs]


class Archive:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.freeze = json.loads((self.root / "PRE-INFERENCE-FREEZE.json").read_text(encoding="utf-8"))
        plan_bytes = (self.root / "EVALUATION-PLAN.json").read_bytes()
        check(digest(plan_bytes) == self.freeze["files"]["EVALUATION-PLAN.json"], "Policy changed after freeze")
        self.plan = json.loads(plan_bytes)
        self.records = load_corpus((self.root / "corpus.json").read_bytes(), self.freeze["files"]["corpus.json"])
        self.session = self.tokenizer = self.vectors = None

    def verify_corpus(self):
        check(digest((self.root / "corpus.json").read_bytes()) == self.freeze["files"]["corpus.json"], "Corpus changed during session")

    def setup_model(self):
        model_dir = os.environ.get("HAUNTED_MODEL_DIR")
        check(model_dir, "Set HAUNTED_MODEL_DIR to the existing local model; no download supported")
        manifest_bytes = (self.root / "MODEL-MANIFEST.json").read_bytes()
        check(digest(manifest_bytes) == self.freeze["files"]["MODEL-MANIFEST.json"], "Model manifest changed")
        for item in json.loads(manifest_bytes)["files"]:
            data = (Path(model_dir) / item["file"]).read_bytes()
            check(len(data) == item["bytes"] and digest(data) == item["sha256"], "Model artifact mismatch: " + item["file"])
        from tokenizers import Tokenizer
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.session = ort.InferenceSession(str(Path(model_dir) / "onnx/model.onnx"), options, providers=["CPUExecutionProvider"])

    def embed(self, texts):
        import numpy as np
        if self.session is None:
            self.setup_model()
        encoded = [self.tokenizer.encode(text) for text in texts]
        check(all(0 < len(e.ids) <= 256 for e in encoded), "Text exceeds model token limit; no silent truncation")
        width, pad = max(len(e.ids) for e in encoded), self.tokenizer.token_to_id("[PAD]")
        arrays = {"input_ids": [e.ids + [pad] * (width - len(e.ids)) for e in encoded],
                  "attention_mask": [[1] * len(e.ids) + [0] * (width - len(e.ids)) for e in encoded],
                  "token_type_ids": [e.type_ids + [0] * (width - len(e.ids)) for e in encoded]}
        inputs = {item.name: np.asarray(arrays[item.name], dtype=np.int64) for item in self.session.get_inputs()}
        hidden = self.session.run(None, inputs)[0]
        mask = np.asarray(arrays["attention_mask"], dtype=np.float32)[..., None]
        check(hidden.ndim == 3 and hidden.shape[:2] == mask.shape[:2], "Unexpected model output")
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        lengths = np.linalg.norm(pooled, axis=1, keepdims=True)
        check(np.isfinite(pooled).all() and (lengths > 0).all(), "Nonfinite/zero embedding")
        return pooled / lengths

    def rank(self, query, mode="semantic"):
        self.verify_corpus()
        check(isinstance(query, str) and query.strip() and len(query) <= 1000, "Query required, <=1000 characters")
        check(mode in ("semantic", "lexical"), "Unknown retrieval mode")
        texts = [r["text"] for r in self.records]
        if mode == "semantic":
            if self.vectors is None:
                self.vectors = self.embed(texts)
            scores = (self.vectors @ self.embed([query])[0]).tolist()
            threshold = self.plan["semantic_policy"]["threshold"]
        else:
            scores, threshold = lexical_scores(query, texts), 0
        order = sorted(range(len(texts)), key=lambda i: (-scores[i], self.records[i]["id"]))
        chosen = [i for i in order if scores[i] >= threshold and (mode != "lexical" or scores[i] > 0)][:3]
        hits = [citation(self.records[i], float(scores[i])) for i in chosen]
        return {"mode": mode, "query": query, "abstained": not hits, "threshold": threshold,
                "score_kind": "COSINE_SIMILARITY_NOT_PROBABILITY_OR_TRUTH", "hits": hits,
                "all_scores": [{"source_id": self.records[i]["id"], "score": float(scores[i])} for i in order],
                "authored_conflicts": conflicts([self.records[i] for i in chosen]),
                "scope": "FICTIONAL_CLUES_ONLY_NO_GENERATED_FACTS"}

    def notebook(self, query, hypothesis, source_ids):
        self.verify_corpus()
        check(isinstance(hypothesis, str) and hypothesis.strip() and len(hypothesis) <= 1000, "Reader hypothesis required")
        check(len(source_ids) == 2 and len(set(source_ids)) == 2, "Choose exactly two different sources")
        by_id = {r["id"]: r for r in self.records}
        check(all(s in by_id for s in source_ids), "Unknown selected source")
        retrieval = self.rank(query)
        selected = [by_id[s] for s in source_ids]
        return {"kind": "FICTIONAL_READER_NOTEBOOK", "hypothesis": hypothesis,
                "hypothesis_origin": "READER_INPUT_NOT_VERIFIED", "personal_review": "NOT_CLAIMED",
                "selected_evidence": [citation(r) for r in selected],
                "authored_conflicts": conflicts(selected), "retrieval": retrieval,
                "supports_hypothesis": "NOT_AUTOMATICALLY_DETERMINED",
                "corpus_sha256": self.freeze["files"]["corpus.json"]}


def main():
    parser = argparse.ArgumentParser(description="Haunted Archive: fictional evidence, not automatic truth")
    commands = parser.add_subparsers(dest="command", required=True)
    rank = commands.add_parser("rank")
    rank.add_argument("query")
    rank.add_argument("--mode", choices=("semantic", "lexical"), default="semantic")
    note = commands.add_parser("note")
    note.add_argument("query"); note.add_argument("hypothesis")
    note.add_argument("sources", nargs=2); note.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        archive = Archive()
        if args.command == "rank":
            result = archive.rank(args.query, args.mode)
        else:
            result = archive.notebook(args.query, args.hypothesis, args.sources)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ArchiveError, OSError, KeyError, ImportError, json.JSONDecodeError) as exc:
        print("LOCAL ERROR: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
