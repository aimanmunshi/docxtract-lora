"""
Run the fine-tuned Donut+LoRA model on a single receipt image and print the
extracted structured fields.

Usage:
    python src/infer.py path/to/receipt.png
    python src/infer.py path/to/receipt.png --adapter_path checkpoints/lora-cord-v2
"""

import argparse
import json
import os

from PIL import Image

from cord_utils import TASK_START_TOKEN, generate_and_parse
from evaluate import load_finetuned

DEFAULT_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "lora-cord-v2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to a receipt image (jpg/png)")
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER_PATH)
    args = parser.parse_args()

    if not os.path.isdir(args.adapter_path):
        raise SystemExit(
            f"No LoRA adapter found at {args.adapter_path}. Run `python src/train.py` first "
            "to produce checkpoints/lora-cord-v2/, or pass --adapter_path to a different location."
        )

    print(f"Loading fine-tuned model from {args.adapter_path} ...")
    processor, model = load_finetuned(args.adapter_path)

    image = Image.open(args.image_path)
    print(f"Running inference on {args.image_path} ...")
    _, extracted_fields = generate_and_parse(model, processor, image, TASK_START_TOKEN)

    print("\nExtracted fields:")
    print(json.dumps(extracted_fields, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
