# -*- coding: utf-8 -*-
r"""VLM baseline runner - development set only, read-only on everything existing.

Backends
  dryrun : no model is called. Emits UNCERTAIN rows so the plumbing can be proven
           end to end. NEVER a cleanliness result - clearly marked.
  ollama : local Ollama server, vision model of your choice (no data leaves the machine)
  groq   : Groq OpenAI-compatible API (IMAGES LEAVE THIS MACHINE - approval required)

Hard safety rails
  * --split dev  is the ONLY split that runs by default (the original 136 images).
  * --split final_test refuses unless --i-have-locked-the-config is passed.
  * Nothing existing is written: output goes to artifacts/vlm_baseline/ only.

Usage
  python prep\vlm_runner.py --backend dryrun --limit 5
  python prep\vlm_runner.py --backend ollama --model qwen2.5vl:3b
  python prep\vlm_runner.py --backend groq   --model meta-llama/llama-4-scout-17b-16e-instruct
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from PIL import Image, ImageOps

from dataset_prep import MANIFEST, PROJECT_ROOT, load_manifest

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "vlm_baseline")
PROMPT_PATH = os.path.join(OUT_DIR, "prompt_v1.txt")
COMBINED = os.path.join(PROJECT_ROOT, "artifacts", "combined", "combined_manifest_v2.csv")
LOCKED_PROPERTIES = {"A11", "A12"}

LABELS = {"CLEAN", "NOT_CLEAN", "UNCERTAIN"}
REASONS = {"ITEMS_NOT_PROPERLY_ARRANGED", "VISIBLE_DIRT_WASTE_DEBRIS", "BOTH",
           "NO_SPECIFIC_VISIBLE_ISSUE"}
BANNED = re.compile(r"\btenant\b", re.I)


# ----------------------------------------------------------------- data ------
def dev_rows() -> list[dict]:
    """The authoritative original 136 labelled images (UNCERTAIN excluded)."""
    rows = []
    for r in load_manifest(MANIFEST, supervised_only=True):
        rows.append({"image_file": r["image_file"], "image_path": r["image_path"],
                     "room_id": r["room_id"], "sub_area": r["sub_area"],
                     "property_group": r["property_group"],
                     "human_label": r["final_label"], "human_reason": r["evidence_reason"],
                     "split": "dev_136"})
    assert len(rows) == 136, len(rows)
    assert not ({r["property_group"] for r in rows} & LOCKED_PROPERTIES)
    return rows


def final_test_rows() -> list[dict]:
    with open(COMBINED, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["property_group"] in LOCKED_PROPERTIES]
    out = [{"image_file": r["image_file"], "image_path": r["image_path"],
            "room_id": r["room_id"], "sub_area": r["sub_area"],
            "property_group": r["property_group"], "human_label": r["label"],
            "human_reason": r["human_reason"], "split": "final_test_27"} for r in rows]
    assert len(out) == 27, len(out)
    return out


def encode_image(path: str, max_edge: int = 1024) -> str:
    """EXIF-corrected JPEG, downscaled, base64. Original file is never modified."""
    with Image.open(path) as im:
        im.load()
        img = ImageOps.exif_transpose(im).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# -------------------------------------------------------------- backends ----
def call_dryrun(_b64, _prompt, _model):
    return {"predicted_label": "UNCERTAIN",
            "predicted_reason": "NO_SPECIFIC_VISIBLE_ISSUE",
            "evidence_reason": "DRY RUN - no model was called; this is not a prediction.",
            "maintenance_note": "", "uncertainty_flag": True}, "dryrun", 0.0


def call_ollama(b64, prompt, model):
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt, "images": [b64],
                          "stream": False, "format": "json",
                          "options": {"temperature": 0}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    return parse_json(body.get("response", "")), model, time.time() - t0


def call_groq(b64, prompt, model):
    from openai import OpenAI
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
    return parse_json(resp.choices[0].message.content), model, time.time() - t0


BACKENDS = {"dryrun": call_dryrun, "ollama": call_ollama, "groq": call_groq}


# ------------------------------------------------------------ validation ----
def parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def validate(obj: dict) -> tuple[dict, list[str]]:
    """Coerce to the fixed schema and report every rule the output broke."""
    problems = []
    label = str(obj.get("predicted_label", "")).upper().strip()
    if label not in LABELS:
        problems.append(f"bad predicted_label {label!r}")
        label = "UNCERTAIN"
    reason = str(obj.get("predicted_reason", "")).upper().strip()
    if reason not in REASONS:
        problems.append(f"bad predicted_reason {reason!r}")
        reason = "NO_SPECIFIC_VISIBLE_ISSUE"
    if label in ("CLEAN", "UNCERTAIN") and reason != "NO_SPECIFIC_VISIBLE_ISSUE":
        problems.append(f"reason {reason} not allowed with label {label}")
        reason = "NO_SPECIFIC_VISIBLE_ISSUE"
    evidence = " ".join(str(obj.get("evidence_reason", "")).split())
    if not evidence:
        problems.append("empty evidence_reason")
    if BANNED.search(evidence):
        problems.append("evidence uses the word 'tenant'")
    if len(evidence) > 400:
        problems.append("evidence longer than 400 chars")
        evidence = evidence[:400]
    return {"predicted_label": label, "predicted_reason": reason,
            "evidence_reason": evidence,
            "maintenance_note": " ".join(str(obj.get("maintenance_note", "")).split()),
            "uncertainty_flag": bool(obj.get("uncertainty_flag", False))}, problems


# ------------------------------------------------------------------ main ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--model", default="dryrun")
    ap.add_argument("--split", choices=["dev", "final_test"], default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default=PROMPT_PATH)
    ap.add_argument("--tag", default="")
    ap.add_argument("--i-have-locked-the-config", action="store_true",
                    help="required to touch the locked 27-image final test set")
    args = ap.parse_args()

    if args.split == "final_test" and not args.i_have_locked_the_config:
        sys.exit("REFUSED: the 27 A11/A12 images are the locked final-test set. "
                 "Re-run with --i-have-locked-the-config only after the VLM "
                 "configuration is frozen.")

    prompt = open(args.prompt, encoding="utf-8").read()
    prompt_version = re.search(r"PROMPT_VERSION:\s*(\S+)", prompt).group(1)
    rows = dev_rows() if args.split == "dev" else final_test_rows()
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.tag or f"{args.backend}_{re.sub(r'[^A-Za-z0-9]+', '-', args.model)}"
    stem = f"vlm_predictions_{args.split}_{tag}"
    out_csv = os.path.join(OUT_DIR, stem + ".csv")
    out_jsonl = os.path.join(OUT_DIR, stem + ".jsonl")

    print(f"backend={args.backend} model={args.model} split={args.split} "
          f"images={len(rows)} prompt={prompt_version}")
    if args.backend == "groq":
        print("  NOTE: each image is uploaded to the Groq API.")

    fn = BACKENDS[args.backend]
    results, failures, t_start = [], 0, time.time()
    for i, r in enumerate(rows, 1):
        abs_path = os.path.join(PROJECT_ROOT, r["image_path"].replace("/", os.sep))
        rec = {"image_file": r["image_file"], "image_path": r["image_path"],
               "room_id": r["room_id"], "sub_area": r["sub_area"],
               "property_group": r["property_group"], "split": r["split"],
               "human_label": r["human_label"], "human_reason": r["human_reason"],
               "model_name": args.model, "backend": args.backend,
               "prompt_version": prompt_version}
        try:
            raw, model_used, secs = fn(encode_image(abs_path), prompt, args.model)
            clean, problems = validate(raw)
            rec.update(clean)
            rec.update({"schema_problems": "|".join(problems), "seconds": round(secs, 2),
                        "error": "", "raw_output": json.dumps(raw, ensure_ascii=False)[:1000]})
        except Exception as e:                                    # noqa: BLE001
            failures += 1
            rec.update({"predicted_label": "ERROR", "predicted_reason": "",
                        "evidence_reason": "", "maintenance_note": "",
                        "uncertainty_flag": True, "schema_problems": "call_failed",
                        "seconds": 0.0, "error": f"{type(e).__name__}: {e}"[:300],
                        "raw_output": ""})
        results.append(rec)
        print(f"  [{i}/{len(rows)}] {r['image_file'][:34]:<34} "
              f"{rec['predicted_label']:<10} {rec.get('seconds', 0):>6.1f}s"
              + (f"  !{rec['schema_problems']}" if rec["schema_problems"] else ""), flush=True)

    cols = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(results)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cfg = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "backend": args.backend, "model": args.model, "split": args.split,
           "images": len(rows), "prompt_version": prompt_version,
           "prompt_file": os.path.relpath(args.prompt, PROJECT_ROOT),
           "temperature": 0, "max_image_edge_px": 1024,
           "failures": failures, "wall_seconds": round(time.time() - t_start, 1),
           "is_real_prediction": args.backend != "dryrun"}
    with open(os.path.join(OUT_DIR, stem + "_runconfig.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nwrote {out_csv}\nwrote {out_jsonl}\nfailures: {failures}")
    if args.backend == "dryrun":
        print("DRY RUN - these rows are NOT predictions and must not be scored as a baseline.")


if __name__ == "__main__":
    main()
