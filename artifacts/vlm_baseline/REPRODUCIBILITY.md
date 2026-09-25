# VLM baseline — reproducibility

Separate, read-only experiment. It does not touch the CNN demo model, `demo_app.py`,
the original manifest, or any existing experiment artifact. Everything it writes
lands in `artifacts/vlm_baseline/`.

## Status

**Development inference in progress.** `qwen2.5vl:3b` has been pulled into the local
Ollama store (3.2 GB, on D:) and loads successfully — 2.9 GB resident, 100% CPU,
context 4096. The 136-image development run is currently executing with `prompt_v1`
at temperature 0 and is not yet complete, so **no development metrics exist yet**.

`environment_audit.json` records the environment as surveyed *before* that pull, when
no vision model was present; its `can_run_a_vlm_right_now: false` verdict has since
been resolved by pulling the model. The Groq finding in it still stands — that key
exposes no vision model.

The pipeline was proven end to end with the dry-run backend first. The locked
27-image final test has not been run.

## Data splits

| split | images | source | use |
|---|---|---|---|
| `dev` | 136 | `cleanliness_manifest.csv` (CLEAN + NOT_CLEAN; the 5 UNCERTAIN stay excluded) | prompt development, model selection, all reported metrics |
| `final_test` | 27 | A11/A12, from `artifacts/combined/combined_manifest_v2.csv` | **locked** — untouched until the configuration is frozen |

The runner refuses `--split final_test` unless `--i-have-locked-the-config` is
passed. That refusal was tested and works.

## Fixed prompt

`prompt_v1.txt` — `PROMPT_VERSION: v1`. It encodes the labelling policy: belongings
are not automatically a problem; localized untidiness stays CLEAN; maintenance
defects are never cleanliness issues; kitchens are judged on the visible kitchen
area; UNCERTAIN when too little of the room is visible; no ownership language and
the word "tenant" is forbidden in the evidence.

Output is one JSON object per image:

```json
{"predicted_label": "CLEAN|NOT_CLEAN|UNCERTAIN",
 "predicted_reason": "ITEMS_NOT_PROPERLY_ARRANGED|VISIBLE_DIRT_WASTE_DEBRIS|BOTH|NO_SPECIFIC_VISIBLE_ISSUE",
 "evidence_reason": "...", "maintenance_note": "...", "uncertainty_flag": true|false}
```

`vlm_runner.py` validates every field, forces `NO_SPECIFIC_VISIBLE_ISSUE` when the
label is CLEAN or UNCERTAIN, rejects the word "tenant", caps evidence length, and
records each violation in a `schema_problems` column rather than silently fixing it.

## Preprocessing

EXIF orientation corrected, longest edge scaled to 1024 px, JPEG quality 88,
base64. Original image files are opened read-only and never rewritten.

## How to run

```bash
# 1. plumbing check - calls no model, produces no predictions
python prep/vlm_runner.py --backend dryrun --limit 5

# 2. real development run (after a vision model is available)
ollama pull moondream                # or qwen2.5vl:3b / llava-phi3
python prep/vlm_runner.py --backend ollama --model moondream --split dev

# 3. score it
python prep/evaluate_vlm.py --predictions artifacts/vlm_baseline/vlm_predictions_dev_ollama_moondream.csv
```

Determinism: `temperature = 0`. Local models are otherwise reproducible on the same
build; an API model is not guaranteed to be stable across provider updates, so the
model id and run date are recorded in each `*_runconfig.json`.

## Outputs per run

```
vlm_predictions_<split>_<tag>.csv / .jsonl   per-image records with the fixed schema
vlm_predictions_<split>_<tag>_runconfig.json backend, model, prompt version, timings
vlm_predictions_<split>_<tag>_metrics.json   accuracy, balanced accuracy, macro-F1,
                                             per-class, subtype recall, per-room,
                                             per-property, schema problems
vlm_predictions_<split>_<tag>_errors.csv     every disagreement with the human label
```

## Requirements for the two candidate backends

**Ollama, local (no data leaves the machine)**

| model | download | RAM needed | fits here? |
|---|---|---|---|
| `moondream` (~2B) | ~1.7 GB | ~2–3 GB | tight but likely |
| `llava-phi3` | ~2.9 GB | ~4 GB | needs memory freed |
| `qwen2.5vl:3b` | ~3.2 GB | ~4–5 GB | **yes — pulled and running** |
| `llama3.2-vision:11b` | ~7.9 GB | ~10 GB | no |

CPU-only on this machine (Intel HD 4400 cannot accelerate inference).

Measured throughput for `qwen2.5vl:3b` at 1024 px max edge: **~209 s per image**
(n=11 consecutive images, range 200–241 s), which puts the 136-image development run
at roughly **8 hours**. While the model is resident, free RAM falls to about 0.4 GB of
7.9 GB total; per-image time stayed stable at that pressure, with no call failures.

Note when reading timings: Ollama caches the prompt/image prefix, so re-running an
image that was already sent returns in ~20 s. Only first-pass timings are meaningful.

**Groq API** — ruled out. The key in this environment exposes no vision model.

## Honesty note

VLM `evidence_reason` text is a *model-generated observation*, not verified truth.
It is stored alongside the human label for comparison and must never be presented
as confirmed fact.
