"""Bonsai 2 parity reference (docs/bonsai2/plan.md, fiche B-3).

Loads prism-ml/Ternary-Bonsai-2-27B-mlx-2bit through the pack's own
runtime/vision_artifact.py (mlx==0.32.0, mlx-vlm==0.6.3, venv-bonsai2 —
NOT venv617, the versions differ from the Flash-Next reference), renders
four prompts with the checkpoint's chat template (enable_thinking=False:
this compares the model, not the reasoning), and for each records:

  - prompt_ids_i   (int32)   the tokenized prompt
  - last_logits_i  (float32, [vocab])  logits at the last prompt position
  - greedy_ids_i   (int32, [32])       32 greedy continuation ids
    (argmax, re-inject — no EOS stopping, matching the Swift side)

Usage:
    venv-bonsai2/bin/python Scripts/references/bonsai2_reference.py \
        --model-dir "$BONSAI" \
        --output parity/bonsai2-reference.safetensors
"""

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx


@contextlib.contextmanager
def compact_tojson():
    """`chat_template.jinja`'s `{{ tool | tojson }}` calls take no args, so
    `transformers`' filter (chat_template_utils.py) falls back to
    `json.dumps`'s own default separators — `", "` / `": "`, i.e. with
    spaces. swift-jinja's `tojson` (Filters.swift) always encodes compact
    (no spaces): Foundation's `JSONEncoder` never inserts them outside
    `.prettyPrinted`. Same rendered bytes on both sides is what B-3 needs
    (docs/bonsai2/plan.md) — not a model concern, a chat-template one
    shared by the whole codebase — so patch `json.dumps` to match Swift's
    compact style for the one call this reference makes.
    """
    original = json.dumps

    def compact_dumps(*args, **kwargs):
        # `chat_template_utils.tojson` always forwards `separators=None`
        # explicitly, so `dict.setdefault` (key-absence only) would not
        # override it — check the value itself.
        if kwargs.get("separators") is None:
            kwargs["separators"] = (",", ":")
        return original(*args, **kwargs)

    json.dumps = compact_dumps
    try:
        yield
    finally:
        json.dumps = original


# Key order matters here: swift-jinja's `tojson` filter always
# JSONEncoder.sortedKeys (Filters.swift), while Python's jinja2 `tojson`
# preserves dict insertion order. To compare the same rendered prompt bytes
# on both sides (docs/bonsai2/plan.md B-3 — token ids first, before any
# numeric comparison), every level here is written in alphabetical key
# order, matching what the Swift side (Qwen38ToolSpec.toolSpecDictionary)
# produces after sorting. Not a model concern; a chat-template rendering
# one shared by the whole codebase, out of scope for this fiche.
READ_FILE_TOOL = {
    "function": {
        "description": "Lit un fichier sous la racine choisie. Renvoie au plus 200 lignes.",
        "name": "read_file",
        "parameters": {
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
            },
            "required": ["path"],
            "type": "object",
        },
    },
    "type": "function",
}

PROMPTS = [
    {"text": "Dis bonjour en un mot.", "tools": None},
    {"text": "Écris une fonction Swift qui renvoie le carré d'un entier.", "tools": None},
    {"text": "Quelle est la capitale de la France ? Réponds en un mot.", "tools": None},
    {"text": "Lis le fichier README.md", "tools": [READ_FILE_TOOL]},
]

GREEDY_TOKENS = 32


def model_sha256_prefix(model_dir: Path, byte_count: int = 1024 * 1024) -> str:
    with open(model_dir / "model.safetensors", "rb") as f:
        data = f.read(byte_count)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.model_dir / "runtime"))
    from vision_artifact import load_vl_model, chat_config
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = load_vl_model(str(args.model_dir))
    lm = model.language_model

    arrays = {}
    for i, entry in enumerate(PROMPTS):
        kwargs = {"num_images": 0, "enable_thinking": False}
        if entry["tools"] is not None:
            kwargs["tools"] = entry["tools"]
        with compact_tojson():
            prompt_str = apply_chat_template(
                processor, chat_config(config), entry["text"], **kwargs
            )
        prompt_ids = processor.tokenizer.encode(prompt_str, add_special_tokens=False)

        cache = lm.make_cache()
        ids = mx.array([prompt_ids])
        out = lm(ids, cache=cache)
        logits = out.logits[0, -1, :]
        mx.eval(logits)
        last_logits = logits.astype(mx.float32)

        greedy_ids = []
        next_ids = mx.argmax(logits, axis=-1)
        for _ in range(GREEDY_TOKENS):
            mx.eval(next_ids)
            token = int(next_ids.item())
            greedy_ids.append(token)
            step_out = lm(next_ids[None, None], cache=cache)
            next_ids = mx.argmax(step_out.logits[0, -1, :], axis=-1)

        arrays[f"prompt_ids_{i}"] = mx.array(prompt_ids, dtype=mx.int32)
        arrays[f"last_logits_{i}"] = last_logits
        arrays[f"greedy_ids_{i}"] = mx.array(greedy_ids, dtype=mx.int32)
        print(f"prompt {i}: {len(prompt_ids)} ids, greedy={greedy_ids[:8]}...")

    metadata = {
        "model_sha256_1mb": model_sha256_prefix(args.model_dir),
        "greedy_tokens": str(GREEDY_TOKENS),
        "enable_thinking": "false",
        "num_prompts": str(len(PROMPTS)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), arrays, metadata)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
