#!/usr/bin/env python3
"""Run the browser planner graph through a complete semantic-code sequence.

This is a native ONNX Runtime smoke test for the same two-row CFG, top-p, and
temperature logic used by the Transformers.js Worker. It catches cached-step
overflows and collapsed code sequences before an 8+ GB export is selected by
the browser demo.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--replace-head-from-checkpoint",
        action="store_true",
        help=(
            "Read the graph's final hidden state and apply the checkpoint "
            "audio-code rows as the output head."
        ),
    )
    return parser.parse_args()


def load_parity_helpers():
    path = Path(__file__).with_name("validate-ace-planner-parity.py")
    spec = importlib.util.spec_from_file_location("ace_planner_parity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import parity helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def padded_batch(
    tokenizer,
    prompts: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    rows = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in prompts
    ]
    lengths = [len(row) for row in rows]
    width = max(lengths)
    input_ids = np.full(
        (2, width),
        int(tokenizer.pad_token_id),
        dtype=np.int64,
    )
    attention_mask = np.zeros((2, width), dtype=np.int64)
    position_ids = np.zeros((2, width), dtype=np.int64)
    for index, row in enumerate(rows):
        start = width - len(row)
        input_ids[index, start:] = row
        attention_mask[index, start:] = 1
        position_ids[index, start:] = np.arange(len(row), dtype=np.int64)
    return input_ids, attention_mask, position_ids, lengths


def sample_top_p(
    values: np.ndarray,
    rng: random.Random,
    temperature: float,
    top_p: float,
) -> int:
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Planner produced {np.count_nonzero(~np.isfinite(values))} "
            "non-finite audio-code logits"
        )
    maximum = float(values.max())
    nucleus_weights = np.exp(values.astype(np.float64) - maximum)
    order = np.argsort(nucleus_weights)[::-1]
    cumulative = np.cumsum(nucleus_weights[order])
    kept_count = int(
        np.searchsorted(cumulative, top_p * cumulative[-1], side="left")
    ) + 1
    kept = order[:kept_count]
    sample_weights = np.exp(
        (values[kept].astype(np.float64) - maximum) / temperature
    )
    draw = rng.random() * float(sample_weights.sum())
    selected = kept[-1]
    for token, weight in zip(kept, sample_weights, strict=True):
        draw -= float(weight)
        if draw <= 0:
            selected = token
            break
    return int(selected)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    helpers = load_parity_helpers()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint),
        trust_remote_code=True,
    )
    prompts = helpers.formatted_prompts(tokenizer)
    input_ids, attention_mask, position_ids, prompt_lengths = padded_batch(
        tokenizer,
        prompts,
    )

    print("Loading ONNX planner...", flush=True)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(args.onnx),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    if args.replace_head_from_checkpoint:
        from safetensors import safe_open

        index = json.loads(
            (args.checkpoint / "model.safetensors.index.json").read_text()
        )
        weight_file = index["weight_map"]["model.embed_tokens.weight"]
        with safe_open(
            args.checkpoint / weight_file,
            framework="pt",
            device="cpu",
        ) as weights:
            embedding = weights.get_slice("model.embed_tokens.weight")
            audio_head = embedding[
                helpers.AUDIO_CODE_TOKEN_START:helpers.AUDIO_CODE_TOKEN_END
            ].float().numpy()
        output_names = [helpers.FINAL_HIDDEN]
    else:
        audio_head = None
        output_names = ["logits"]
    for layer in range(36):
        output_names.extend(
            [f"present.{layer}.key", f"present.{layer}.value"]
        )
    empty_cache = np.empty((2, 8, 0, 128), dtype=np.float16)
    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for layer in range(36):
        feeds[f"past_key_values.{layer}.key"] = empty_cache
        feeds[f"past_key_values.{layer}.value"] = empty_cache

    rng = random.Random(args.seed)
    codes: list[int] = []
    for step in range(args.steps):
        outputs = session.run(output_names, feeds)
        if audio_head is None:
            logits = outputs[0][:, -1].astype(np.float32)
            conditional = logits[
                0,
                helpers.AUDIO_CODE_TOKEN_START:helpers.AUDIO_CODE_TOKEN_END,
            ]
            unconditional = logits[
                1,
                helpers.AUDIO_CODE_TOKEN_START:helpers.AUDIO_CODE_TOKEN_END,
            ]
        else:
            hidden = outputs[0][:, -1].astype(np.float32)
            audio_logits = hidden @ audio_head.T
            conditional = audio_logits[0]
            unconditional = audio_logits[1]
        guided = unconditional + args.cfg_scale * (
            conditional - unconditional
        )
        code = sample_top_p(
            guided,
            rng,
            args.temperature,
            args.top_p,
        )
        codes.append(code)
        token_id = helpers.AUDIO_CODE_TOKEN_START + code

        feeds = {
            "input_ids": np.full((2, 1), token_id, dtype=np.int64),
            "attention_mask": np.concatenate(
                [
                    attention_mask,
                    np.ones((2, step + 1), dtype=np.int64),
                ],
                axis=1,
            ),
            "position_ids": np.asarray(
                [
                    [prompt_lengths[0] + step],
                    [prompt_lengths[1] + step],
                ],
                dtype=np.int64,
            ),
        }
        cache_outputs = outputs[1:]
        for layer in range(36):
            feeds[f"past_key_values.{layer}.key"] = cache_outputs[layer * 2]
            feeds[f"past_key_values.{layer}.value"] = cache_outputs[
                layer * 2 + 1
            ]
        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"semantic step {step + 1}/{args.steps}: code {code}",
                flush=True,
            )

    counts = np.bincount(codes, minlength=64_000)
    longest_run = 1
    run = 1
    for index in range(1, len(codes)):
        if codes[index] == codes[index - 1]:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1
    report = {
        "count": len(codes),
        "unique": len(set(codes)),
        "transitions": sum(
            left != right
            for left, right in zip(codes, codes[1:])
        ),
        "longest_run": longest_run,
        "dominant_count": int(counts.max()),
        "first_20": codes[:20],
        "last_20": codes[-20:],
        "codes": codes,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")

    if report["unique"] < max(2, len(codes) // 4):
        raise SystemExit("Planner semantic sequence collapsed")
    print("Planner cached-step semantic smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
