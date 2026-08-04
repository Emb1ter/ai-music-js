#!/usr/bin/env python3
"""Package the ACE planner body separately from its WebGPU Q8 output head.

ONNX Runtime Web has a practical WASM-session ceiling below the size of the
single-file 4B planner. This utility removes the tied language-model head from
the ONNX graph, compacts the remaining external-data shards, and publishes the
Q8 head as raw sidecar buffers for a small WebGPU compute kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper


FINAL_HIDDEN = "/model/layers.36/final_norm_layernorm/output_0"
VOCAB_SIZE = 217_204
HIDDEN_SIZE = 2_560
BLOCK_SIZE = 32
Q8_WEIGHT_BYTES = VOCAB_SIZE * HIDDEN_SIZE
Q8_SCALE_BYTES = VOCAB_SIZE * (HIDDEN_SIZE // BLOCK_SIZE) * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-source", type=Path, required=True)
    parser.add_argument("--q8-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def external_location(tensor: onnx.TensorProto) -> str | None:
    for entry in tensor.external_data:
        if entry.key == "location":
            return entry.value
    return None


def set_external_location(tensor: onnx.TensorProto, location: str) -> None:
    for entry in tensor.external_data:
        if entry.key == "location":
            entry.value = location
            return
    raise RuntimeError(f"Tensor {tensor.name} has no external-data location")


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    onnx_dir = output / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    source_graph = args.body_source / "onnx" / "model_body.onnx"
    model = onnx.load(source_graph, load_external_data=False)
    if not model.graph.output or model.graph.output[0].name != FINAL_HIDDEN:
        raise RuntimeError("Body graph does not expose the expected final hidden state")
    embedding_node = next(
        (
            node
            for node in model.graph.node
            if node.name == "/model/embed_tokens/Gather_Q4"
        ),
        None,
    )
    scale_node = next(
        (
            node
            for node in model.graph.node
            if node.name == "/model/embed_tokens/ScaleResidual"
        ),
        None,
    )
    if embedding_node is None or scale_node is None:
        raise RuntimeError("Body graph does not contain the expected Q4 embedding")
    model.graph.node.remove(embedding_node)
    scale_node.input[0] = "inputs_embeds"
    model.graph.input.append(
        helper.make_tensor_value_info(
            "inputs_embeds",
            TensorProto.FLOAT16,
            ["batch_size", "sequence_length", HIDDEN_SIZE],
        )
    )
    model.graph.node.append(
        helper.make_node(
            "Identity",
            [FINAL_HIDDEN],
            ["last_hidden_state"],
            name="/model/ExposeLastHiddenState",
        )
    )
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info(
            "last_hidden_state",
            TensorProto.FLOAT16,
            ["batch_size", "sequence_length", HIDDEN_SIZE],
        )
    )
    model = onnx.utils.Extractor(model).extract_model(
        [value.name for value in model.graph.input],
        [value.name for value in model.graph.output],
    )

    used_locations = sorted(
        {
            location
            for tensor in model.graph.initializer
            if (location := external_location(tensor)) is not None
        }
    )
    source_metadata_path = args.body_source / "conversion-metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text())
    external_files = [
        value["name"]
        for value in source_metadata.get("external_data_files", [])
    ]
    expected_locations = [
        name
        for name in external_files
        if name not in {
            "model_quantized.onnx_data",
            "model_quantized.onnx_data_1",
        }
    ]
    if not expected_locations or used_locations != expected_locations:
        raise RuntimeError(
            f"Unexpected body external-data layout: {used_locations}"
        )
    compact_locations = [
        "model_quantized.onnx_data"
        if index == 0
        else f"model_quantized.onnx_data_{index}"
        for index in range(len(expected_locations))
    ]
    location_map = dict(zip(used_locations, compact_locations, strict=True))
    for tensor in model.graph.initializer:
        location = external_location(tensor)
        if location is not None:
            set_external_location(tensor, location_map[location])
    onnx.save(model, onnx_dir / "model_quantized.onnx")

    source_onnx = args.body_source / "onnx"
    for source_name, destination_name in location_map.items():
        link_or_copy(
            source_onnx / source_name,
            onnx_dir / destination_name,
        )

    for name in (
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        source = args.body_source / name
        if source.exists():
            shutil.copyfile(source, output / name)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["transformers.js_config"] = {
        "dtype": "q8",
        "kv_cache_dtype": {"q8": "float16", "fp16": "float16"},
        "use_external_data_format": {
            "model_quantized.onnx": len(compact_locations)
        },
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    q8_onnx = args.q8_source / "onnx"
    weight_source = q8_onnx / "model_quantized.onnx_data"
    if weight_source.stat().st_size != Q8_WEIGHT_BYTES:
        raise RuntimeError(
            f"Unexpected Q8 head weight size: {weight_source.stat().st_size}"
        )
    link_or_copy(weight_source, output / "lm_head_q8.bin")
    scale_source = q8_onnx / "model_quantized.onnx_data_1"
    with scale_source.open("rb") as source, (output / "lm_head_q8_scales.f16").open(
        "wb"
    ) as destination:
        remaining = Q8_SCALE_BYTES
        while remaining:
            chunk = source.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("Q8 scale source ended early")
            destination.write(chunk)
            remaining -= len(chunk)

    metadata = {
        "architecture": "ACE-Step 1.5 5 Hz LM 4B",
        "onnx_body": {
            "transformer": source_metadata["precision"][
                "transformer_projections"
            ],
            "embedding": "external WebGPU blockwise Q8 gather",
            "residual_stream_scale": 1 / 256,
        },
        "webgpu_head": {
            "weight": "lm_head_q8.bin",
            "weight_dtype": "int8",
            "scales": "lm_head_q8_scales.f16",
            "scale_dtype": "float16",
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "block_size": BLOCK_SIZE,
        },
    }
    (output / "conversion-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"Packaged {output} ({total / 1e9:.2f} GB total)")


if __name__ == "__main__":
    main()
