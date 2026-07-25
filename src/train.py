"""
LoRA fine-tune donut-base on the CORD-v2 train subset for structured
receipt-field extraction.

What this does and why:
  1. Loads donut-base with its native (pretrained-only) vocabulary.
  2. Extends the vocabulary with CORD's field tokens (<s_menu>, <s_nm>,
     <s_total_price>, ...) discovered from the train subset, resizing the
     decoder's embedding table + tied lm_head to match.
  3. Wraps the decoder's attention projections (q_proj/k_proj/v_proj/
     out_proj) in LoRA adapters, and marks the resized embedding/lm_head
     as fully trainable (`modules_to_save`) — LoRA alone can't teach the
     model to emit brand-new vocabulary, since it never touches the
     embedding table. The pretrained Swin vision encoder is left frozen
     entirely: this task is about learning CORD's output structure, not
     new visual features, and donut-base's encoder already generalizes
     well to document images.
  4. Trains with bf16 mixed precision (native support on this Ampere GPU,
     so no loss-scaling complexity), logging per-step loss and saving a
     loss curve. An initial 3-epoch run left validation loss still
     dropping steadily (6.58 -> 2.72 -> 1.82), so the default here is 6
     epochs — still only ~6 minutes on an RTX 3050 6GB, and val loss
     kept improving through epoch 6 (-> 0.92) without diverging.
  5. Saves only the LoRA adapter (+ resized tokenizer/processor) to
     /checkpoints — never the full ~800MB base model.

Usage:
    python src/train.py
    python src/train.py --epochs 6 --lr 1e-4 --batch_size 1 --grad_accum 4
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

from cord_utils import (
    MAX_TARGET_LENGTH,
    EMBED_MODULE_NAME,
    LM_HEAD_MODULE_NAME,
    LORA_TARGET_MODULES,
    TASK_END_TOKEN,
    extend_vocab_for_cord,
    json2token,
    load_base_processor_and_model,
    load_cord_subsets,
    parse_ground_truth,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "lora-cord-v2")


class CordSeq2SeqDataset(Dataset):
    """Wraps a CORD-v2 HF split, producing (pixel_values, labels) pairs."""

    def __init__(self, hf_split, processor):
        self.hf_split = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        example = self.hf_split[idx]
        pixel_values = self.processor(
            example["image"].convert("RGB"), return_tensors="pt"
        ).pixel_values.squeeze(0)

        gt = parse_ground_truth(example["ground_truth"])
        # No leading TASK_START_TOKEN here: VisionEncoderDecoderModel's
        # label-shifting (shift_tokens_right) already prepends
        # decoder_start_token_id (== TASK_START_TOKEN's id) as the implicit
        # BOS context. Including it again in the label text would train the
        # model on a duplicated "<s_cord-v2><s_cord-v2>..." pattern.
        target_text = json2token(gt) + TASK_END_TOKEN

        labels = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return pixel_values, labels


def build_lora_model(processor, model):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=[EMBED_MODULE_NAME, LM_HEAD_MODULE_NAME],
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            "WARNING: no CUDA GPU detected. LoRA fine-tuning of a ~200M-param "
            "vision-language model on CPU is impractical (many hours per "
            "epoch). Stopping rather than running an unrealistically slow "
            "job — re-run on a machine with a GPU (e.g. Colab) or reduce "
            "scope further (fewer examples, smaller image size)."
        )
        raise SystemExit(1)

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using device={device}, mixed precision dtype={amp_dtype}")

    print("Loading CORD-v2 subsets...")
    subsets = load_cord_subsets()
    train_split, val_split = subsets["train"], subsets["validation"]

    print("Loading donut-base (fp32 master weights for training stability)...")
    processor, model = load_base_processor_and_model(dtype=torch.float32)

    print("Extending vocabulary with CORD field tokens discovered in train split...")
    extend_vocab_for_cord(processor, model, train_split)
    print(f"New vocab size: {len(processor.tokenizer)}")

    model = build_lora_model(processor, model)
    model.to(device)

    train_ds = CordSeq2SeqDataset(train_split, processor)
    val_ds = CordSeq2SeqDataset(val_split, processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    print(f"Effective batch size = {args.batch_size * args.grad_accum}, total optimizer steps ~= {total_steps}")

    training_log = []
    global_step = 0
    start_time = time.time()

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for i, (pixel_values, labels) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / args.grad_accum

            loss.backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = loss.item() * args.grad_accum
                elapsed = time.time() - start_time
                training_log.append(
                    {"epoch": epoch, "step": global_step, "loss": step_loss, "elapsed_seconds": elapsed}
                )
                if global_step % 5 == 0 or global_step == 1:
                    print(f"epoch {epoch} step {global_step}/{total_steps} loss={step_loss:.4f} elapsed={elapsed:.1f}s")

        # End-of-epoch validation loss (quick sanity check, not used for early stopping)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                val_losses.append(outputs.loss.item())
        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        print(f"=== epoch {epoch} done: mean val loss = {mean_val_loss:.4f} ===")
        training_log.append({"epoch": epoch, "val_loss": mean_val_loss, "elapsed_seconds": time.time() - start_time})
        model.train()

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    plot_loss_curve(training_log)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    processor.save_pretrained(CHECKPOINT_DIR)
    print(f"Saved LoRA adapter + processor to {CHECKPOINT_DIR}")


def plot_loss_curve(training_log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [e["step"] for e in training_log if "step" in e]
    losses = [e["loss"] for e in training_log if "step" in e]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label="train loss (per optimizer step)")
    plt.xlabel("optimizer step")
    plt.ylabel("loss")
    plt.title("LoRA fine-tuning loss — donut-base on CORD-v2")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "loss_curve.png")
    plt.savefig(out_path, dpi=120)
    print(f"Saved loss curve to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()
    train(args)
