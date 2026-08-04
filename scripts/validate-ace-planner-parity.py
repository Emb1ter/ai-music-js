#!/usr/bin/env python3
"""Compare an ACE planner ONNX export with the official MLX checkpoint.

This intentionally checks token ranking, not only cosine similarity. A large
shared logit offset can make two broken distributions look deceptively similar
under cosine while their most likely audio codes are completely different.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np


AUDIO_CODE_TOKEN_START = 151_669
AUDIO_CODE_TOKEN_END = AUDIO_CODE_TOKEN_START + 64_000
FINAL_HIDDEN = "/model/layers.36/final_norm_layernorm/output_0"
DEFAULT_CAPTION = (
    "Warm analog synthwave song, steady electronic drums, pulsing bass, "
    "cinematic pads, memorable chorus, clear expressive lead vocal singing "
    "every supplied lyric, polished studio mix"
)
DEFAULT_LYRICS = """[Verse]
Static in the room hums low, like an old radio static.
I sit on the edge, watching the light flicker across the floor.
A ghost from yesterday sits by my side now.
It waits for me to forget what's left behind.

[Chorus]
The neon glow fades, but I'm still here waiting there.
For you, this place feels exactly right again and then gone."""
DEFAULT_METADATA = {
    "bpm": 100,
    "caption": (
        "A polished synthwave song built on warm analog synthesizers, steady "
        "electronic drums, and a pulsing bassline. Cinematic pads create a "
        "nostalgic atmosphere while a memorable chorus supports a clear, "
        "expressive lead vocal."
    ),
    "duration": 30,
    "keyscale": "F major",
    "language": "en",
    "timesignature": 4,
}
SYSTEM_PROMPT = (
    "# Instruction\n"
    "Generate audio semantic tokens based on the given conditions:\n\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--minimum-top-k-overlap",
        type=float,
        default=0.8,
        help="Required overlap fraction for conditional, unconditional, and CFG rankings.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional file receiving the full metrics report.",
    )
    parser.add_argument(
        "--replace-onnx-head-from-checkpoint",
        action="store_true",
        help=(
            "Read the exposed final hidden state from the ONNX graph and apply "
            "the checkpoint's original FP16 tied output embedding in NumPy."
        ),
    )
    parser.add_argument(
        "--q8-sidecar-directory",
        type=Path,
        help=(
            "Read lm_head_q8.bin and lm_head_q8_scales.f16 from this "
            "directory and use them for both token embedding and output-head "
            "validation, matching the browser WebGPU kernels."
        ),
    )
    parser.add_argument(
        "--disable-graph-optimization",
        action="store_true",
        help=(
            "Disable ONNX Runtime graph optimization to reduce temporary "
            "memory while validating very large FP32 browser artifacts."
        ),
    )
    return parser.parse_args()


def formatted_prompts(tokenizer) -> tuple[str, str]:
    user_prompt = f"# Caption\n{DEFAULT_CAPTION}\n\n# Lyric\n{DEFAULT_LYRICS}\n"
    base = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    metadata = DEFAULT_METADATA
    cot = (
        "<think>\n"
        f"bpm: {metadata['bpm']}\n"
        f"caption: {metadata['caption']}\n"
        f"duration: {metadata['duration']}\n"
        f"keyscale: {metadata['keyscale']}\n"
        f"language: {metadata['language']}\n"
        f"timesignature: {metadata['timesignature']}\n"
        "</think>"
    )
    conditional = f"{base}{cot}\n\n"
    unconditional_base = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "NO USER INPUT"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    unconditional = f"{unconditional_base}<think>\n\n</think>\n\n"
    return conditional, unconditional


def run_mlx(checkpoint: Path, prompts: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    import mlx.core as mx
    from mlx_lm.utils import load

    print("Loading official MLX planner...", flush=True)
    model, tokenizer = load(str(checkpoint))
    outputs: list[np.ndarray] = []
    for label, prompt in zip(("conditional", "unconditional"), prompts, strict=True):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        logits = model(mx.array(token_ids, dtype=mx.int32)[None])
        mx.eval(logits)
        value = np.asarray(logits[0, -1].astype(mx.float32))
        outputs.append(value)
        print(f"MLX {label}: {len(token_ids)} prompt tokens", flush=True)
    del model
    mx.clear_cache()
    gc.collect()
    return outputs[0], outputs[1]


def run_onnx(
    graph_path: Path,
    tokenizer,
    prompts: tuple[str, str],
    replace_head_from_checkpoint: Path | None = None,
    q8_sidecar_directory: Path | None = None,
    disable_graph_optimization: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    print("Loading ONNX planner...", flush=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_graph_optimization
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(graph_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    session_input_names = {value.name for value in session.get_inputs()}
    session_output_names = {value.name for value in session.get_outputs()}
    hidden_output_name = (
        FINAL_HIDDEN
        if FINAL_HIDDEN in session_output_names
        else "last_hidden_state"
    )
    outputs: list[np.ndarray] = []
    for label, prompt in zip(("conditional", "unconditional"), prompts, strict=True):
        token_ids = np.asarray(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            dtype=np.int64,
        )
        sequence_length = token_ids.shape[1]
        feeds: dict[str, np.ndarray] = {
            "input_ids": token_ids,
            "attention_mask": np.ones((1, sequence_length), dtype=np.int64),
            "position_ids": np.arange(sequence_length, dtype=np.int64)[None],
        }
        if "inputs_embeds" in session_input_names:
            if q8_sidecar_directory is not None:
                feeds["inputs_embeds"] = q8_token_embeddings(
                    q8_sidecar_directory,
                    token_ids,
                )
            elif replace_head_from_checkpoint is not None:
                feeds["inputs_embeds"] = checkpoint_token_embeddings(
                    replace_head_from_checkpoint,
                    token_ids,
                )
            else:
                raise RuntimeError(
                    "The split planner body requires checkpoint or Q8-sidecar embeddings"
                )
        cache_input = next(
            value
            for value in session.get_inputs()
            if value.name == "past_key_values.0.key"
        )
        cache_dtype = (
            np.float32 if cache_input.type == "tensor(float)" else np.float16
        )
        empty_cache = np.empty((1, 8, 0, 128), dtype=cache_dtype)
        for layer in range(36):
            feeds[f"past_key_values.{layer}.key"] = empty_cache
            feeds[f"past_key_values.{layer}.value"] = empty_cache
        if (
            replace_head_from_checkpoint is None
            and q8_sidecar_directory is None
        ):
            logits = session.run(["logits"], feeds)[0]
            outputs.append(logits[0, -1].astype(np.float32))
        else:
            hidden = session.run([hidden_output_name], feeds)[0]
            if q8_sidecar_directory is not None:
                outputs.append(
                    apply_q8_audio_head(
                        q8_sidecar_directory,
                        hidden[0, -1].astype(np.float32),
                    )
                )
            else:
                outputs.append(
                    apply_checkpoint_audio_head(
                        replace_head_from_checkpoint,
                        hidden[0, -1].astype(np.float32),
                    )
                )
        print(f"ONNX {label}: {sequence_length} prompt tokens", flush=True)
    del session
    gc.collect()
    return outputs[0], outputs[1]


def checkpoint_token_embeddings(
    checkpoint: Path,
    token_ids: np.ndarray,
) -> np.ndarray:
    from safetensors import safe_open

    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text()
    )
    weight_file = index["weight_map"]["model.embed_tokens.weight"]
    with safe_open(
        checkpoint / weight_file,
        framework="pt",
        device="cpu",
    ) as weights:
        embedding = weights.get_tensor("model.embed_tokens.weight")
        return embedding[token_ids].half().numpy()


def q8_tied_matrix(sidecar_directory: Path) -> tuple[np.memmap, np.memmap]:
    weights = np.memmap(
        sidecar_directory / "lm_head_q8.bin",
        dtype=np.int8,
        mode="r",
        shape=(217_204, 2_560),
    )
    scales = np.memmap(
        sidecar_directory / "lm_head_q8_scales.f16",
        dtype=np.float16,
        mode="r",
        shape=(217_204, 2_560 // 32),
    )
    return weights, scales


def q8_token_embeddings(
    sidecar_directory: Path,
    token_ids: np.ndarray,
) -> np.ndarray:
    weights, scales = q8_tied_matrix(sidecar_directory)
    rows = weights[token_ids]
    row_scales = np.repeat(scales[token_ids], 32, axis=-1)
    return (rows.astype(np.float32) * row_scales.astype(np.float32)).astype(
        np.float16
    )


def apply_q8_audio_head(
    sidecar_directory: Path,
    hidden: np.ndarray,
) -> np.ndarray:
    weights, scales = q8_tied_matrix(sidecar_directory)
    logits = np.full((217_204,), -np.inf, dtype=np.float32)
    for start in range(AUDIO_CODE_TOKEN_START, AUDIO_CODE_TOKEN_END, 4096):
        end = min(start + 4096, AUDIO_CODE_TOKEN_END)
        rows = weights[start:end].astype(np.float32)
        row_scales = np.repeat(
            scales[start:end].astype(np.float32),
            32,
            axis=-1,
        )
        logits[start:end] = (rows * row_scales) @ hidden
    return logits


def apply_checkpoint_audio_head(
    checkpoint: Path,
    hidden: np.ndarray,
) -> np.ndarray:
    from safetensors import safe_open

    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text()
    )
    weight_file = index["weight_map"]["model.embed_tokens.weight"]
    logits = np.full((217_204,), -np.inf, dtype=np.float32)
    with safe_open(
        checkpoint / weight_file,
        framework="pt",
        device="cpu",
    ) as weights:
        embedding = weights.get_slice("model.embed_tokens.weight")
        for start in range(AUDIO_CODE_TOKEN_START, AUDIO_CODE_TOKEN_END, 4096):
            end = min(start + 4096, AUDIO_CODE_TOKEN_END)
            rows = embedding[start:end].float().numpy()
            logits[start:end] = rows @ hidden
    return logits


def ranking_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    top_k: int,
) -> dict[str, object]:
    reference_audio = reference[AUDIO_CODE_TOKEN_START:AUDIO_CODE_TOKEN_END]
    candidate_audio = candidate[AUDIO_CODE_TOKEN_START:AUDIO_CODE_TOKEN_END]
    reference_finite = np.isfinite(reference_audio)
    candidate_finite = np.isfinite(candidate_audio)
    reference_rankable = np.where(
        reference_finite, reference_audio, -np.inf
    )
    candidate_rankable = np.where(
        candidate_finite, candidate_audio, -np.inf
    )
    reference_order = np.argsort(reference_rankable)[::-1]
    candidate_order = np.argsort(candidate_rankable)[::-1]
    reference_top = reference_order[:top_k]
    candidate_top = candidate_order[:top_k]
    intersection = len(set(reference_top.tolist()) & set(candidate_top.tolist()))
    reference_rank_of_candidate_top = int(
        np.flatnonzero(reference_order == candidate_top[0])[0]
    )
    candidate_rank_of_reference_top = int(
        np.flatnonzero(candidate_order == reference_top[0])[0]
    )
    shared_finite = reference_finite & candidate_finite
    if np.count_nonzero(shared_finite) > 1:
        centered_reference = (
            reference_audio[shared_finite]
            - reference_audio[shared_finite].mean(dtype=np.float64)
        ).astype(np.float64)
        centered_candidate = (
            candidate_audio[shared_finite]
            - candidate_audio[shared_finite].mean(dtype=np.float64)
        ).astype(np.float64)
        correlation = float(
            np.dot(centered_reference, centered_candidate)
            / (
                np.linalg.norm(centered_reference)
                * np.linalg.norm(centered_candidate)
            )
        )
    else:
        correlation = None
    return {
        "correlation": correlation,
        "reference_finite": int(np.count_nonzero(reference_finite)),
        "candidate_finite": int(np.count_nonzero(candidate_finite)),
        "top_k": top_k,
        "top_k_intersection": intersection,
        "top_k_overlap": intersection / top_k,
        "reference_top_code": int(reference_top[0]),
        "candidate_top_code": int(candidate_top[0]),
        "candidate_rank_of_reference_top": candidate_rank_of_reference_top,
        "reference_rank_of_candidate_top": reference_rank_of_candidate_top,
        "reference_top_codes": reference_top.tolist(),
        "candidate_top_codes": candidate_top.tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.replace_onnx_head_from_checkpoint and args.q8_sidecar_directory:
        raise SystemExit(
            "Choose either --replace-onnx-head-from-checkpoint or "
            "--q8-sidecar-directory, not both."
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint),
        trust_remote_code=True,
    )
    prompts = formatted_prompts(tokenizer)
    mlx_conditional, mlx_unconditional = run_mlx(args.checkpoint, prompts)
    onnx_conditional, onnx_unconditional = run_onnx(
        args.onnx,
        tokenizer,
        prompts,
        (
            args.checkpoint
            if args.replace_onnx_head_from_checkpoint
            else None
        ),
        args.q8_sidecar_directory,
        args.disable_graph_optimization,
    )

    mlx_cfg = mlx_unconditional + 2.0 * (mlx_conditional - mlx_unconditional)
    with np.errstate(invalid="ignore"):
        onnx_cfg = onnx_unconditional + 2.0 * (
            onnx_conditional - onnx_unconditional
        )
    report = {
        "conditional": ranking_metrics(
            mlx_conditional, onnx_conditional, args.top_k
        ),
        "unconditional": ranking_metrics(
            mlx_unconditional, onnx_unconditional, args.top_k
        ),
        "cfg": ranking_metrics(mlx_cfg, onnx_cfg, args.top_k),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")

    failed = [
        name
        for name, metrics in report.items()
        if float(metrics["top_k_overlap"]) < args.minimum_top_k_overlap
    ]
    if failed:
        raise SystemExit(
            "Planner parity failed for "
            + ", ".join(failed)
            + f"; required top-{args.top_k} overlap "
            + f"{args.minimum_top_k_overlap:.0%}."
        )
    print("Planner ranking parity passed.", flush=True)


if __name__ == "__main__":
    main()
