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
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx


# Written in the natural OpenAI order (type, function → name, description,
# parameters → type, properties, required): transformers' `tojson` keeps
# dict insertion order and `json.dumps` defaults, and since swift-jinja 2.5
# plus `Qwen38OrderedJSON` (Qwen38Core) the Swift side renders the client's
# bytes in the client's order the same way — no compensation on either side.
# The Swift parity test (`bonsai2ParityAgainstPythonReference`) parses the
# same JSON text; keep the two in sync.
READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Lit un fichier sous la racine choisie. Renvoie au plus 200 lignes.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
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
