# Haunted Archive — a fictional clue notebook

A limited Python CLI prototype: ask a question, inspect original fictional
passages, then choose two passages and record an unverified hypothesis.
MiniLM ranks passages; the program does not generate an answer, determine
truth, or decide whether evidence supports a hypothesis. Conflict flags compare
author-written fictional annotations; they are not model contradiction detection.
The five passages describe no real people, customer data, or personal memories.
Prepared with AI assistance. Original application code/corpus/tests: MIT;
third-party model and dependencies retain their own licenses (see notices).

## Reproduce with an already configured environment

This repository contains no interpreter, dependencies, weights, credentials or installer.
It does not automatically install or download anything. An outsider without the
following prerequisites cannot run semantic retrieval immediately.

The recorded Windows 11 CPU run used Python 3.12.14, NumPy 2.5.4,
ONNX Runtime 1.31.0 and Tokenizers 0.23.3. Other combinations are not validated.
Have those dependencies installed beforehand and an existing local copy of
sentence-transformers/all-MiniLM-L6-v2 at revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41. MODEL-MANIFEST.json lists all nine
required files, source URLs, byte sizes and SHA256 hashes. The application
checks all of them before inference. Preserve the model's Apache-2.0 terms
if obtaining or redistributing it separately.

From the extracted directory in PowerShell (replace the placeholder):

```powershell
$env:HAUNTED_MODEL_DIR = 'D:/existing-model-directory'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
python -B -m unittest discover -v
python -B archive_game.py rank 'Was the north lantern lit at midnight on Night Three?'
python -B archive_game.py rank 'Where are the recorded memories?' --mode lexical
python -B archive_game.py note 'Was the north lantern lit?' 'FICTIONAL EXAMPLE: I have not decided which account to trust.' harbor-a harbor-b --output my-fictional-notebook.json
```

`python` must refer to the configured interpreter. The notebook output path must
not already exist; its parent directory must exist. Notebook output is ordinary
local JSON, not an uploaded claim. No wallet, payment, API key or cloud is used.
Lexical rank works without model dependencies; the full suite needs NumPy.
Without HAUNTED_MODEL_DIR, two real-inference tests skip explicitly; a skipped
run does not demonstrate inference. The included new receipt records 14 passed,
0 skipped: 12 mechanical checks and 2 real local CPU inference invariants.

## Inspect the actual limited interaction

`demo/TRANSCRIPT.txt` contains three real command outputs from this package:
retrieve the conflicting lantern accounts, ask for missing wattage, and write a
fictional notebook selecting harbor-a/harbor-b. Only the interpreter alias and
the description of the environment omit private absolute paths; JSON outputs
are actual stdout. `demo/example-notebook.json` is the actual saved notebook.
All questions/hypotheses/passages are synthetic fixtures.

The source, original tests, corpus, policy and model manifest are byte-identical
to the frozen local190 prototype. No threshold or evaluation case was changed.
The original four smoke cases were frozen before first inference; they are not
a held-out accuracy study or proof of novelty, superiority, or commercial value.

## Limits that matter

- Similarity is not probability, proof, truth, answerability, or guaranteed relevance.
- The lantern query also retrieves an irrelevant greenhouse passage above the
  fixed 0.40 threshold (about 0.428). This false positive is retained openly.
- Exact wattage is absent from the corpus, yet related lantern passages are
  returned rather than abstaining. No wattage answer is generated. Low-similarity
  abstention is not an answerability detector.
- Quotation hashes and offsets verify source bytes, not a story's factual truth.
- A hypothesis is unverified reader input; neither human review nor support is claimed.
- Queries are limited to 1000 characters and 256 model tokens; oversized inputs
  fail rather than being silently truncated. A corpus/policy/hash change fails closed.
- Python socket patches in tests are not operating-system network isolation.

## Project and contest status

This source package is prepared for public release under MIT. The runnable
interaction remains the limited CLI described above; there is no web interface
or hosted model. No claim of a prize, customer adoption or revenue is made.

The transcript is a text record of real local executions, not a video. A separate
video replay re-presents recorded command outputs and is labelled as a replay
rather than a new live run. It is not a simulated application interface. The
edited video's duration is not process latency. See [the video](docs/demo.mp4),
[poster](docs/demo-poster.png), [run receipt](docs/demo-run.json) and
[caption script](docs/demo-script.txt) for the demonstration and its provenance.

To optionally rebuild the edited replay, use a fresh output directory and an
environment that already has Pillow, ffmpeg (with libx264), and an installed
Segoe UI or DejaVu Sans font. These are additional rendering prerequisites;
the script does not install or download them and performs no model inference:

```text
python tools/render_demo.py --receipt docs/demo-run.json --output replay-docs --frames replay-frames
```

ffmpeg must be on PATH, or supply its existing executable with `--ffmpeg`.
The script checks recorded stdout hashes and writes a captioned replay, poster
and script from that receipt. Generated frames and local concat files are
working files, not part of this repository's release package.

This project is being prepared for the Hacktoberfest Open-Source AI Challenge
Week 4, Ghost in the Machine. A public repository or an article draft does not
by itself establish an accepted contest entry. Publication/submission and genuine
human author review are separate steps. No Colombian payment or prize is promised.
