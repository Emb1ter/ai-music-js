#!/usr/bin/env python3
"""Create deterministic multi-step ACE planner checkpoints with MLX.

The browser qualification worker uses the same two-row CFG formula, nucleus
membership rule, temperature, and Mulberry32 random stream. The JSON output is
small enough to commit and compare without storing full 64k logit vectors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_parity_helpers():
    path = Path(__file__).with_name("validate-ace-planner-parity.py")
    spec = importlib.util.spec_from_file_location(
        "ai_music_js_validate_ace_planner_parity",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parity helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARITY = load_parity_helpers()
AUDIO_CODE_TOKEN_START = PARITY.AUDIO_CODE_TOKEN_START
AUDIO_CODE_TOKEN_END = PARITY.AUDIO_CODE_TOKEN_END
formatted_prompts = PARITY.formatted_prompts


CFG_SCALE = 2.0
TEMPERATURE = 0.85
TOP_P = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


class Mulberry32:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    @staticmethod
    def imul(left: int, right: int) -> int:
        return (left * right) & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        value = self.state
        value = self.imul(value ^ (value >> 15), value | 1)
        value ^= (
            value
            + self.imul(value ^ (value >> 7), value | 61)
        ) & 0xFFFFFFFF
        value &= 0xFFFFFFFF
        return ((value ^ (value >> 14)) & 0xFFFFFFFF) / 4_294_967_296


def sample_top_p(values: np.ndarray, random: Mulberry32) -> int:
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        raise RuntimeError("Planner reference has no finite CFG logits")
    nucleus_weights = np.exp(values.astype(np.float64) - maximum)
    nucleus_total = float(nucleus_weights.sum(dtype=np.float64))
    order = np.argsort(nucleus_weights)[::-1]
    cumulative = np.cumsum(nucleus_weights[order], dtype=np.float64)
    kept_count = int(np.searchsorted(cumulative / nucleus_total, TOP_P)) + 1
    kept = order[:kept_count]
    sample_weights = np.exp(
        (values[kept].astype(np.float64) - maximum) / TEMPERATURE
    )
    draw = random.random() * float(sample_weights.sum(dtype=np.float64))
    for code, weight in zip(kept, sample_weights, strict=True):
        draw -= float(weight)
        if draw <= 0:
            return int(code)
    return int(kept[-1])


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")

    import mlx.core as mx
    from mlx_lm.utils import load
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint),
        trust_remote_code=True,
    )
    conditional, unconditional = formatted_prompts(tokenizer)
    sequences = [
        tokenizer.encode(value, add_special_tokens=False)
        for value in (conditional, unconditional)
    ]

    print("Loading official MLX planner...", flush=True)
    model, _ = load(str(args.checkpoint))
    random = Mulberry32(args.seed)
    checkpoints: list[dict[str, object]] = []
    for step in range(args.steps):
        rows: list[np.ndarray] = []
        # Run the two prompts independently. This is equivalent to a correctly
        # left-padded browser batch with an attention mask; passing pad tokens
        # to MLX without such a mask changes the unconditional distribution.
        for sequence in sequences:
            logits = model(mx.array(sequence, dtype=mx.int32)[None])
            mx.eval(logits)
            rows.append(np.asarray(logits[0, -1].astype(mx.float32)))
        conditional_audio = rows[0][
            AUDIO_CODE_TOKEN_START:AUDIO_CODE_TOKEN_END
        ]
        unconditional_audio = rows[1][
            AUDIO_CODE_TOKEN_START:AUDIO_CODE_TOKEN_END
        ]
        cfg = unconditional_audio + CFG_SCALE * (
            conditional_audio - unconditional_audio
        )
        code = sample_top_p(cfg, random)
        checkpoints.append(
            {
                "step": step + 1,
                "selected_code": code,
                "conditional_finite": int(
                    np.count_nonzero(np.isfinite(conditional_audio))
                ),
                "unconditional_finite": int(
                    np.count_nonzero(np.isfinite(unconditional_audio))
                ),
                "cfg_finite": int(np.count_nonzero(np.isfinite(cfg))),
                "conditional_top_codes": np.argsort(conditional_audio)[
                    -20:
                ][::-1].tolist(),
                "unconditional_top_codes": np.argsort(unconditional_audio)[
                    -20:
                ][::-1].tolist(),
                "cfg_top_codes": np.argsort(cfg)[-20:][::-1].tolist(),
            }
        )
        token_id = AUDIO_CODE_TOKEN_START + code
        for sequence in sequences:
            sequence.append(token_id)
        print(f"step {step + 1}/{args.steps}: code {code}", flush=True)

    report = {
        "checkpoint": str(args.checkpoint),
        "precision": "official BF16 via MLX",
        "seed": args.seed,
        "cfg_scale": CFG_SCALE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "prompt_lengths": [
            len(tokenizer.encode(value, add_special_tokens=False))
            for value in (conditional, unconditional)
        ],
        "checkpoints": checkpoints,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
