"""
Explore the CORD-v2 dataset (naver-clova-ix/cord-v2).

CORD-v2 is a re-packaging of the CORD receipt dataset for Donut-style training:
each example has a receipt `image` and a `ground_truth` field containing a JSON
string with a `gt_parse` structure describing the receipt's fields (menu items,
sub_total, total, etc.) plus a Donut-specific token sequence.

This script:
  1. Loads a few samples from each split.
  2. Prints the raw annotation structure so we can see what fields exist.
  3. Saves a handful of example images + their ground truth (as a side-by-side
     PNG and a plain-text dump) to /results for a quick visual sanity check.

Run:
    python data/explore_data.py
"""

import json
import os

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

DATASET_NAME = "naver-clova-ix/cord-v2"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_EXAMPLES = 3


def load_and_report():
    print(f"Loading dataset: {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)

    print("\n=== Dataset splits ===")
    for split_name, split in ds.items():
        print(f"  {split_name}: {len(split)} examples")

    return ds


def print_sample_structure(ds, split="train", n=N_EXAMPLES):
    print(f"\n=== Inspecting {n} raw samples from '{split}' split ===")
    split_ds = ds[split]
    for i in range(n):
        example = split_ds[i]
        print(f"\n--- Example {i} ---")
        print("Keys:", list(example.keys()))
        print("Image size:", example["image"].size, "mode:", example["image"].mode)

        gt_raw = example["ground_truth"]
        try:
            gt = json.loads(gt_raw)
        except json.JSONDecodeError:
            print("ground_truth is not valid JSON, printing raw string (truncated):")
            print(gt_raw[:500])
            continue

        print("ground_truth top-level keys:", list(gt.keys()))
        if "gt_parse" in gt:
            print("gt_parse (structured fields):")
            print(json.dumps(gt["gt_parse"], indent=2, ensure_ascii=False)[:1500])


def save_examples_for_sanity_check(ds, split="train", n=N_EXAMPLES):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    split_ds = ds[split]

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(n):
        example = split_ds[i]
        image = example["image"].convert("RGB")

        gt_raw = example["ground_truth"]
        try:
            gt = json.loads(gt_raw)
            gt_parse = gt.get("gt_parse", gt)
            gt_text = json.dumps(gt_parse, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            gt_text = gt_raw

        # Save the raw image
        img_path = os.path.join(RESULTS_DIR, f"sample_{i}_image.png")
        image.save(img_path)

        # Save ground truth as plain text alongside it
        gt_path = os.path.join(RESULTS_DIR, f"sample_{i}_ground_truth.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(gt_text)

        # Also build a combined image+text panel for quick visual review
        panel_width = image.width + 500
        panel_height = max(image.height, 600)
        panel = Image.new("RGB", (panel_width, panel_height), "white")
        panel.paste(image, (0, 0))
        draw = ImageDraw.Draw(panel)
        text_x = image.width + 10
        # Wrap long lines crudely for readability
        wrapped_lines = []
        for line in gt_text.splitlines():
            while len(line) > 60:
                wrapped_lines.append(line[:60])
                line = line[60:]
            wrapped_lines.append(line)
        draw.text((text_x, 10), "\n".join(wrapped_lines[:80]), fill="black", font=font)

        combined_path = os.path.join(RESULTS_DIR, f"sample_{i}_combined.png")
        panel.save(combined_path)

        print(f"Saved: {img_path}, {gt_path}, {combined_path}")


def decide_subset_sizes(ds):
    print("\n=== Deciding train/val/test subset sizes ===")
    full_train = len(ds["train"])
    full_val = len(ds["validation"]) if "validation" in ds else 0
    full_test = len(ds["test"]) if "test" in ds else 0
    print(f"Full CORD-v2: train={full_train}, validation={full_val}, test={full_test}")

    # Reasoning: CORD-v2 full train is ~800 images. On a single 6GB-VRAM GPU
    # (RTX 3050 laptop), fine-tuning Donut (a ~200M-param seq2seq VLM) with LoRA
    # is feasible, but full-resolution, full-dataset training would push epoch
    # time and VRAM too far for a portfolio-scale project. We deliberately
    # subsample to keep iteration fast and results reproducible in well under
    # an hour of GPU time:
    #   - train:      200 examples (enough signal for LoRA adapters to visibly
    #                  shift structured-field accuracy, small enough for a few
    #                  epochs in minutes-not-hours)
    #   - validation:  50 examples (loss monitoring / sanity, not heavily used)
    #   - test:       100 examples (held out, used identically for baseline
    #                  and fine-tuned evaluation so the before/after comparison
    #                  is apples-to-apples)
    subset_sizes = {"train": 200, "validation": 50, "test": 100}
    print("Chosen subset sizes:", subset_sizes)
    print(
        "Rationale: keeps GPU training time to well under an hour on a 6GB "
        "laptop GPU while still giving LoRA enough examples to learn from, "
        "and keeps the held-out test set fixed and identical across baseline "
        "and fine-tuned evaluation runs for a fair comparison."
    )
    return subset_sizes


if __name__ == "__main__":
    dataset = load_and_report()
    print_sample_structure(dataset, split="train", n=N_EXAMPLES)
    save_examples_for_sanity_check(dataset, split="train", n=N_EXAMPLES)
    decide_subset_sizes(dataset)
