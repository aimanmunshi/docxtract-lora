# Baseline vs. Fine-Tuned Comparison

Both models evaluated on the same 100-example held-out CORD-v2 test subset, scored with the identical field-level precision/recall/F1 metric (see `src/cord_utils.py::field_level_accuracy`).

| Metric | Baseline (donut-base) | Fine-tuned (+ LoRA) | Change |
|---|---|---|---|
| Micro-F1 (field-level) | 0.0% | 11.0% | +11.0% |
| Macro-F1 (per-example mean) | 0.0% | 13.4% | +13.4% |
| Micro-precision | 0.0% | 18.9% | +18.9% |
| Micro-recall | 0.0% | 7.8% | +7.8% |

- Baseline correctly extracted **0/1301** ground-truth fields across all 100 test receipts.
- Fine-tuned model correctly extracted **101/1301** ground-truth fields.

**Why the baseline is exactly 0%:** `donut-base` is pretrained only — it has never been shown CORD's field vocabulary (`<s_menu>`, `<s_total_price>`, ...) and its tokenizer doesn't even contain those tokens. It immediately emits an end-of-sequence token and produces no structured output. This is the genuine pre-fine-tuning state, not a bug (see `results/baseline_metrics.json` example outputs).

**What the fine-tuned model learned:** after LoRA fine-tuning (6 epochs over 200 training examples — see `results/training_log.json` and `results/loss_curve.png`), the model reliably produces well-formed CORD-structured output (correct tags, properly closed sequences) and extracts a meaningful fraction of fields correctly. Value-level errors are still common (digit misreads, partial menu-item names) — expected given only 200 training examples, 6 epochs, and a downscaled 960x720 input resolution. See the Limitations section of the README for the honest scope of this result.
