"""
Shared utilities for the CORD-v2 / Donut LoRA project.

Used by data/explore_data.py, src/train.py, src/evaluate.py, and src/infer.py
so that dataset subsetting and the structured-field comparison metric stay
identical across the baseline and fine-tuned evaluation runs (an
apples-to-apples before/after comparison depends on this).
"""

import json
import re

import torch
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel

DATASET_NAME = "naver-clova-ix/cord-v2"
BASE_MODEL_NAME = "naver-clova-ix/donut-base"

# donut-base defaults to a 2560x1920 input resolution (tuned for dense
# full-page documents). That's too large to fit LoRA fine-tuning in 6GB of
# VRAM, so we downsize to 960x720 for both training and evaluation — smaller
# than the original Donut paper's CORD setting (1280x960), traded off
# deliberately for speed/memory on a single laptop GPU. Kept identical across
# train/eval/infer so no code path sees a resolution mismatch.
IMAGE_SIZE = {"height": 960, "width": 720}

# 99% of the 200-example train subset's linearized target sequences fit
# within 768 tokens (see data exploration); the remaining ~1% get truncated,
# an accepted tradeoff for keeping decoder sequence length manageable.
MAX_TARGET_LENGTH = 768

# Chosen in data/explore_data.py after inspecting full split sizes.
# Kept small and deterministic (fixed seed) so training stays fast on a
# single 6GB-VRAM laptop GPU, and so the *same* test examples are used for
# both the baseline and fine-tuned evaluation.
SUBSET_SIZES = {"train": 200, "validation": 50, "test": 100}
SEED = 42


def load_cord_subsets():
    """Load CORD-v2 and return deterministic, shuffled subsets per split."""
    ds = load_dataset(DATASET_NAME)
    subsets = {}
    for split, n in SUBSET_SIZES.items():
        if split not in ds:
            continue
        split_ds = ds[split].shuffle(seed=SEED)
        n = min(n, len(split_ds))
        subsets[split] = split_ds.select(range(n))
    return subsets


def parse_ground_truth(gt_raw: str) -> dict:
    """Parse a CORD-v2 `ground_truth` JSON string and return the gt_parse dict."""
    gt = json.loads(gt_raw)
    return gt.get("gt_parse", gt)


def flatten_fields(obj, parent_key: str = "") -> dict:
    """
    Flatten a nested CORD gt_parse structure into {dotted.path: string_value}.

    CORD's gt_parse mixes dicts and lists (e.g. multiple menu items), so list
    entries are indexed (menu.0.nm, menu.1.nm, ...). Only leaf (str/int/float)
    values are kept — this gives us a flat set of fields to compare for
    field-level accuracy.
    """
    fields = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent_key}.{k}" if parent_key else k
            fields.update(flatten_fields(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{parent_key}.{i}"
            fields.update(flatten_fields(v, key))
    else:
        fields[parent_key] = str(obj).strip()
    return fields


def normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise for comparison."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


TASK_START_TOKEN = "<s_cord-v2>"
TASK_END_TOKEN = "</s_cord-v2>"
SEP_TOKEN = "<sep/>"


def json2token(obj, sort_json_key: bool = True) -> str:
    """
    Linearize a CORD gt_parse dict into Donut's XML-like token format, the
    inverse of `DonutProcessor.token2json`. Mirrors the original Donut
    fine-tuning recipe: each key becomes a <s_key>...</s_key> span, and
    sibling list items (e.g. repeated menu entries) are joined with <sep/>
    inside the shared wrapper so that `token2json` can round-trip them back
    into a list of dicts.
    """
    if isinstance(obj, dict):
        keys = sorted(obj.keys()) if sort_json_key else list(obj.keys())
        parts = []
        for k in keys:
            v = obj[k]
            if isinstance(v, list):
                inner = SEP_TOKEN.join(json2token(item, sort_json_key) for item in v)
            else:
                inner = json2token(v, sort_json_key)
            parts.append(f"<s_{k}>{inner}</s_{k}>")
        return "".join(parts)
    elif isinstance(obj, list):
        return SEP_TOKEN.join(json2token(item, sort_json_key) for item in obj)
    else:
        return str(obj)


def collect_field_keys(obj, keys: set) -> None:
    """Recursively collect every dict key appearing in a gt_parse structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            collect_field_keys(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            collect_field_keys(item, keys)


def build_special_tokens(train_split) -> list:
    """
    Scan every training example's gt_parse and build the full set of special
    tokens the tokenizer needs: <s_key>/</s_key> for every field key seen,
    plus the task start/end tokens and the list separator. donut-base is only
    pretrained (not fine-tuned on any task), so none of these exist in its
    vocabulary yet — they must be added and the model's embeddings resized
    before training, exactly as the original Donut recipe does when adapting
    to a new document type.
    """
    all_keys = set()
    for example in train_split:
        gt = parse_ground_truth(example["ground_truth"])
        collect_field_keys(gt, all_keys)

    special_tokens = [TASK_START_TOKEN, TASK_END_TOKEN, SEP_TOKEN]
    for k in sorted(all_keys):
        special_tokens.append(f"<s_{k}>")
        special_tokens.append(f"</s_{k}>")
    return special_tokens


def field_level_accuracy(pred_fields: dict, gt_fields: dict) -> dict:
    """
    Compare two flattened field dicts.

    Returns precision/recall/F1 over field KEYS matched with equal (normalized)
    VALUES — this rewards a model both for finding the right fields and for
    extracting the right values, which is what actually matters for structured
    receipt extraction.
    """
    gt_norm = {k: normalize_text(v) for k, v in gt_fields.items()}
    pred_norm = {k: normalize_text(v) for k, v in pred_fields.items()}

    correct = sum(
        1 for k, v in gt_norm.items() if k in pred_norm and pred_norm[k] == v
    )
    precision = correct / len(pred_norm) if pred_norm else 0.0
    recall = correct / len(gt_norm) if gt_norm else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "correct_fields": correct,
        "gt_field_count": len(gt_norm),
        "pred_field_count": len(pred_norm),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base_processor_and_model(dtype=None):
    """
    Load donut-base with its native (pretrained-only) vocabulary and our
    reduced input resolution. Used as-is for the pre-fine-tuning baseline,
    and as the starting point before vocab extension for LoRA training.
    """
    processor = DonutProcessor.from_pretrained(BASE_MODEL_NAME)
    processor.image_processor.size = IMAGE_SIZE
    processor.image_processor.do_align_long_axis = False

    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_NAME)

    device = get_device()
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    model.to(device, dtype=dtype)
    model.eval()
    return processor, model


def extend_vocab_for_cord(processor, model, train_split) -> None:
    """
    Add CORD's field-name special tokens + task start/end tokens to the
    tokenizer and resize the decoder's embedding table + tied lm_head to
    match, in place. Must be called before LoRA is attached, since the LoRA
    adapter's `modules_to_save` entries need to reference the resized
    embedding/head modules by name.
    """
    special_tokens = build_special_tokens(train_split)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    task_start_id = processor.tokenizer.convert_tokens_to_ids(TASK_START_TOKEN)
    model.config.decoder_start_token_id = task_start_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = task_start_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id


# Full dotted module paths (from the top-level VisionEncoderDecoderModel) for
# the decoder's tied input-embedding / output-head. LoRA on q/k/v/out_proj
# alone can't teach the model to *emit* newly-added vocabulary, since it
# never touches the embedding table — these must be fully trainable
# (peft `modules_to_save`) instead of LoRA-adapted.
EMBED_MODULE_NAME = "decoder.model.decoder.embed_tokens"
LM_HEAD_MODULE_NAME = "decoder.lm_head"

# Scoped to the decoder only: donut-base's encoder (Swin) uses different
# Linear names (query/key/value/dense), so this suffix match never touches
# the pretrained vision encoder — see project notes on why only the decoder's
# attention is adapted for this task.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]


@torch.no_grad()
def generate_and_parse(model, processor, image, prompt_token: str, max_length: int = MAX_TARGET_LENGTH):
    """
    Run image -> structured-JSON generation identically for baseline,
    fine-tuned, and single-image inference use. Returns (raw_decoded_text,
    parsed_dict).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device, dtype=dtype)

    decoder_input_ids = processor.tokenizer(
        prompt_token, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    output_ids = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        num_beams=1,
        do_sample=False,
    )

    raw_text = processor.tokenizer.batch_decode(output_ids)[0]

    # Strip only the wrapper tokens we explicitly fed/expect (pad, the
    # decoder prompt we supplied, and its matching end token if any) —
    # NOT all special tokens, since the CORD field tokens (<s_menu>, etc.)
    # must survive for token2json to parse the structure. Leaving the
    # <s_cord-v2>...</s_cord-v2> wrapper in place would make token2json
    # parse it as an actual outer field, adding a spurious wrapper key.
    clean_text = raw_text.replace(processor.tokenizer.pad_token, "")
    clean_text = clean_text.replace(processor.tokenizer.eos_token, "")
    clean_text = clean_text.replace(prompt_token, "", 1)
    if prompt_token == TASK_START_TOKEN:
        clean_text = clean_text.replace(TASK_END_TOKEN, "")
    clean_text = clean_text.strip()

    parsed = processor.token2json(clean_text)
    return raw_text, parsed
