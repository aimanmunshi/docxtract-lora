"""
Evaluate a Donut model on the CORD-v2 held-out test subset.

Two modes, selected by --adapter_path:
  - Baseline (no --adapter_path): loads donut-base as-is, with its native
    pretraining-only vocabulary. It has never seen CORD's field tokens
    (<s_menu>, <s_total_price>, ...), so it cannot produce them — this
    measures genuine pre-fine-tuning performance, not a strawman.
  - Fine-tuned (--adapter_path checkpoints/lora-cord-v2): loads the same
    base model, extends the vocab to match what was saved during training,
    and attaches the LoRA adapter (including its trained embedding /
    lm_head copies) before generating.

Both modes run on the identical 100-example test subset (see
src/cord_utils.py SUBSET_SIZES, fixed seed) and score with the same
field-level precision/recall/F1 metric, so results are directly comparable.

Usage:
    python src/evaluate.py                                   # baseline
    python src/evaluate.py --adapter_path checkpoints/lora-cord-v2 \
        --output results/finetuned_metrics.json               # fine-tuned
"""

import argparse
import json
import os
import time

from cord_utils import (
    TASK_START_TOKEN,
    flatten_fields,
    field_level_accuracy,
    generate_and_parse,
    load_base_processor_and_model,
    load_cord_subsets,
    parse_ground_truth,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_finetuned(adapter_path: str):
    """Load donut-base, extend vocab to match the checkpoint, attach LoRA."""
    from peft import PeftModel
    from transformers import DonutProcessor

    processor, model = load_base_processor_and_model()
    # The tokenizer saved alongside the adapter has the CORD special tokens
    # added during training; load it in place of the native one so vocab
    # sizes line up before we resize the model and attach the adapter.
    tuned_processor = DonutProcessor.from_pretrained(adapter_path)
    tuned_processor.image_processor.size = processor.image_processor.size
    tuned_processor.image_processor.do_align_long_axis = False
    processor = tuned_processor

    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    task_start_id = processor.tokenizer.convert_tokens_to_ids(TASK_START_TOKEN)
    model.config.decoder_start_token_id = task_start_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = task_start_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, adapter_path)
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    return processor, model


def run_evaluation(processor, model, test_split, prompt_token: str, max_detail: int = 20):
    per_example = []
    total_correct = total_pred = total_gt = 0
    f1_scores = []

    start = time.time()
    for i, example in enumerate(test_split):
        gt = parse_ground_truth(example["ground_truth"])
        gt_fields = flatten_fields(gt)

        raw_text, parsed = generate_and_parse(model, processor, example["image"], prompt_token)
        pred_fields = flatten_fields(parsed)

        metrics = field_level_accuracy(pred_fields, gt_fields)
        total_correct += metrics["correct_fields"]
        total_pred += metrics["pred_field_count"]
        total_gt += metrics["gt_field_count"]
        f1_scores.append(metrics["f1"])

        if i < max_detail:
            per_example.append(
                {
                    "index": i,
                    "raw_generated_text": raw_text[:2000],
                    "parsed_fields": pred_fields,
                    "gt_fields": gt_fields,
                    "metrics": metrics,
                }
            )

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{i + 1}/{len(test_split)}] elapsed={elapsed:.1f}s running_mean_f1={sum(f1_scores)/len(f1_scores):.3f}")

    micro_precision = total_correct / total_pred if total_pred else 0.0
    micro_recall = total_correct / total_gt if total_gt else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    summary = {
        "num_examples": len(test_split),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "total_correct_fields": total_correct,
        "total_predicted_fields": total_pred,
        "total_gt_fields": total_gt,
        "eval_time_seconds": time.time() - start,
    }
    return summary, per_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default=None, help="Path to a saved LoRA adapter; omit for baseline eval")
    parser.add_argument("--output", default=None, help="Output JSON path; defaults based on mode")
    args = parser.parse_args()

    if args.adapter_path:
        print(f"Loading fine-tuned model from adapter: {args.adapter_path}")
        processor, model = load_finetuned(args.adapter_path)
        prompt_token = TASK_START_TOKEN
        default_output = os.path.join(RESULTS_DIR, "finetuned_metrics.json")
        mode = "fine-tuned"
    else:
        print("Loading baseline donut-base (native vocab, no fine-tuning)")
        processor, model = load_base_processor_and_model()
        prompt_token = "<s>"
        default_output = os.path.join(RESULTS_DIR, "baseline_metrics.json")
        mode = "baseline"

    output_path = args.output or default_output

    print("Loading CORD-v2 test subset...")
    subsets = load_cord_subsets()
    test_split = subsets["test"]
    print(f"Evaluating {mode} model on {len(test_split)} test examples...")

    summary, per_example = run_evaluation(processor, model, test_split, prompt_token)

    result = {
        "mode": mode,
        "model": "naver-clova-ix/donut-base" + (f" + LoRA ({args.adapter_path})" if args.adapter_path else ""),
        "summary": summary,
        "example_details": per_example,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
