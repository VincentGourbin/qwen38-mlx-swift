#!/usr/bin/env python3
"""Dump the local HF ChatML template for Flash-Next.

Only tokenizer files are loaded. The JSON is consumed by
``qwen38 flash-template-probe`` before any model forward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=True
    )
    messages = [{"role": "user", "content": args.prompt}]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": args.thinking,
        "reasoning_effort": "low",
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort")
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    elif isinstance(rendered, dict) and "input_ids" in rendered:
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    ids = [int(value) for value in rendered]
    decoded = tokenizer.decode(ids, skip_special_tokens=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"ids": ids, "decoded": decoded, "thinking": args.thinking},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"template reference écrite : {args.output} ({len(ids)} tokens)")


if __name__ == "__main__":
    main()
