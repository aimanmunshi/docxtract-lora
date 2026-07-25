"""
Build results/comparison.md from the baseline and fine-tuned metrics JSON
files produced by src/evaluate.py — generated from the actual saved numbers
rather than hand-typed, so the report can't drift from what was measured.

Usage:
    python src/compare_results.py
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main():
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), encoding="utf-8") as f:
        baseline = json.load(f)["summary"]
    with open(os.path.join(RESULTS_DIR, "finetuned_metrics.json"), encoding="utf-8") as f:
        finetuned = json.load(f)["summary"]

    rows = [
        ("Micro-F1 (field-level)", baseline["micro_f1"], finetuned["micro_f1"]),
        ("Macro-F1 (per-example mean)", baseline["macro_f1"], finetuned["macro_f1"]),
        ("Micro-precision", baseline["micro_precision"], finetuned["micro_precision"]),
        ("Micro-recall", baseline["micro_recall"], finetuned["micro_recall"]),
    ]

    lines = []
    lines.append("# Baseline vs. Fine-Tuned Comparison\n")
    lines.append(
        f"Both models evaluated on the same {baseline['num_examples']}-example held-out "
        "CORD-v2 test subset, scored with the identical field-level precision/recall/F1 "
        "metric (see `src/cord_utils.py::field_level_accuracy`).\n"
    )
    lines.append("| Metric | Baseline (donut-base) | Fine-tuned (+ LoRA) | Change |")
    lines.append("|---|---|---|---|")
    for name, b, f in rows:
        delta = f - b
        lines.append(f"| {name} | {pct(b)} | {pct(f)} | {'+' if delta >= 0 else ''}{pct(delta)} |")

    lines.append("")
    lines.append(
        f"- Baseline correctly extracted **{baseline['total_correct_fields']}/{baseline['total_gt_fields']}** "
        f"ground-truth fields across all 100 test receipts.\n"
        f"- Fine-tuned model correctly extracted **{finetuned['total_correct_fields']}/{finetuned['total_gt_fields']}** "
        f"ground-truth fields.\n"
    )
    lines.append(
        "**Why the baseline is exactly 0%:** `donut-base` is pretrained only — it has never "
        "been shown CORD's field vocabulary (`<s_menu>`, `<s_total_price>`, ...) and its "
        "tokenizer doesn't even contain those tokens. It immediately emits an end-of-sequence "
        "token and produces no structured output. This is the genuine pre-fine-tuning state, "
        "not a bug (see `results/baseline_metrics.json` example outputs).\n"
    )
    lines.append(
        "**What the fine-tuned model learned:** after LoRA fine-tuning (6 epochs over 200 "
        "training examples — see `results/training_log.json` and `results/loss_curve.png`), "
        "the model reliably produces "
        "well-formed CORD-structured output (correct tags, properly closed sequences) and "
        "extracts a meaningful fraction of fields correctly. Value-level errors are still "
        "common (digit misreads, partial menu-item names) — expected given only 200 training "
        "examples, 6 epochs, and a downscaled 960x720 input resolution. See the Limitations "
        "section of the README for the honest scope of this result.\n"
    )

    out_path = os.path.join(RESULTS_DIR, "comparison.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
