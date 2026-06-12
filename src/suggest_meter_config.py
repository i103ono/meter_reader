#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from meter_agent import (
    DEFAULT_MODEL,
    AgentError,
    call_openai,
    load_env_file,
    normalize_config,
    validate_input,
    validate_roi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest meter settings for one image with OpenAI Vision."
    )
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the env file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--detail",
        choices=("low", "high", "auto"),
        default="high",
        help="Vision input detail level.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    env_path = Path(args.env_file).expanduser()

    try:
        validate_input(input_path)
        load_env_file(env_path)
        config = normalize_config(call_openai(input_path, args.model, args.detail))
        validate_roi(config.get("roi"), input_path)
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
