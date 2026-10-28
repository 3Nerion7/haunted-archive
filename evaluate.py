"""Run the frozen cases, retain every result and compare a simple lexical baseline."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch

from archive_game import ROOT, Archive, digest


def evaluate():
    archive = Archive()
    outcomes = []
    with patch("socket.socket.connect", side_effect=RuntimeError("Network forbidden")), \
         patch("socket.create_connection", side_effect=RuntimeError("Network forbidden")):
        for case in archive.plan["cases"]:
            modes = {}
            for mode in ("semantic", "lexical"):
                start = time.perf_counter()
                result = archive.rank(case["query"], mode)
                ids = [h["source_id"] for h in result["hits"]]
                checks = {"abstention": result["abstained"] == case["expected_abstain"],
                          "conflict": bool(result["authored_conflicts"]) == case["expected_conflict"]}
                if case.get("expected_any_source"):
                    checks["top_one_expected"] = bool(ids) and ids[0] in case["expected_any_source"]
                if case.get("expected_all_sources"):
                    checks["all_sources_present"] = set(case["expected_all_sources"]).issubset(ids)
                modes[mode] = {"result": result, "seconds": time.perf_counter() - start,
                               "predeclared_checks": checks, "all_checks_met": all(checks.values())}
            outcomes.append({"case": case, "modes": modes})
        note_case = archive.plan["notebook_test"]
        notebook = archive.notebook(archive.plan["cases"][2]["query"], note_case["hypothesis"],
                                    note_case["selected_sources"])
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "pre_inference_freeze_sha256": digest((ROOT / "PRE-INFERENCE-FREEZE.json").read_bytes()),
              "source_sha256": digest((ROOT / "archive_game.py").read_bytes()), "cases": outcomes,
              "notebook_example": notebook, "real_inference": True, "API_calls": 0,
              "guard": "Python socket guard only; not an OS/native-code network sandbox",
              "interpretation": "Four fictional smoke cases, no calibrated accuracy or probability."
                                " First semantic case includes setup/hash checks. No superiority presumed."}
    destination = ROOT / "EVALUATION-RESULTS.json"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"cases": [{"id": x["case"]["id"], "semantic": x["modes"]["semantic"]["all_checks_met"],
                                "lexical": x["modes"]["lexical"]["all_checks_met"]} for x in outcomes],
                      "report": str(destination)}, indent=2))


if __name__ == "__main__":
    evaluate()
