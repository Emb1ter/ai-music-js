#!/usr/bin/env python3
"""Build an unquantized FP32 ACE-Step 5 Hz planner for browser experiments.

The ACE planner checkpoint is BF16, while browser WebGPU exposes FP32 and FP16
but not BF16. This builder promotes the official Qwen3-4B Transformers.js FP16
graph template to FP32 and streams every ACE checkpoint tensor into FP32 ONNX
external-data files.

The result is intentionally an isolated diagnostic artifact. It is not selected
by the npm package or demo because its weights are roughly 16.8 GB and its tied
embedding alone is larger than Chrome's ordinary ArrayBuffer limit.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper


DEFAULT_CHUNK_MIB = 2200
EMBEDDING_BATCH_ROWS = 1024
FINAL_HIDDEN = "/model/layers.36/final_norm_layernorm/output_0"
EMBEDDING_OUTPUT = "/model/embed_tokens/Gather/output_0"


def load_fp16_builder():
    module_path = Path(__file__).with_name("build-ace-planner-fp16.py")
    spec = importlib.util.spec_from_file_location(
        "ai_music_js_planner_fp16_builder",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FP16_BUILDER = load_fp16_builder()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--tokenizer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_MIB)
    parser.add_argument(
        "--body-only",
        action="store_true",
        help=(
            "Remove the tied embedding/output head, expose inputs_embeds and "
            "last_hidden_state, and build only the transformer body."
        ),
    )
    return parser.parse_args()


def promote_value_info(value_info: onnx.ValueInfoProto) -> None:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type == TensorProto.FLOAT16:
        tensor_type.elem_type = TensorProto.FLOAT


def promote_graph_to_fp32(model: onnx.ModelProto) -> None:
    for value_info in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    ):
        promote_value_info(value_info)


def expose_transformer_body(
    model: onnx.ModelProto,
    hidden_size: int,
) -> onnx.ModelProto:
    """Remove the >2 GiB tied embedding and language-model head."""
    original_inputs = [value.name for value in model.graph.input]
    if "input_ids" not in original_inputs:
        raise RuntimeError("Template graph does not expose input_ids")

    model.graph.input.append(
        helper.make_tensor_value_info(
            "inputs_embeds",
            TensorProto.FLOAT16,
            ["batch_size", "sequence_length", hidden_size],
        )
    )
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == EMBEDDING_OUTPUT:
                node.input[index] = "inputs_embeds"

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
            ["batch_size", "sequence_length", hidden_size],
        )
    )

    # Keep input_ids as a shape-only input. The original position-id reformat
    # graph needs its rank-2 [batch, sequence] shape; substituting the rank-3
    # inputs_embeds shape causes ORT shape inference to merge 3 with 2.
    body_inputs = [value.name for value in model.graph.input]
    body_outputs = [value.name for value in model.graph.output]
    body = onnx.utils.Extractor(model).extract_model(body_inputs, body_outputs)
    if any(
        value.name == "model.embed_tokens.weight"
        for value in body.graph.initializer
    ):
        raise RuntimeError("Body extraction retained the tied embedding")
    return body


def replace_embedding(
    initializer: onnx.TensorProto,
    checkpoint,
    writer,
    vocab_size: int,
    hidden_size: int,
) -> None:
    byte_count = vocab_size * hidden_size * np.dtype(np.float32).itemsize
    initializer.data_type = TensorProto.FLOAT
    del initializer.dims[:]
    initializer.dims.extend([vocab_size, hidden_size])
    sink = writer.reserve(initializer, byte_count, force_new_file=True)
    with checkpoint.slice_source("model.embed_tokens.weight") as source:
        embedding = source.get_slice("model.embed_tokens.weight")
        if list(embedding.get_shape()) != [vocab_size, hidden_size]:
            raise RuntimeError(
                f"Unexpected embedding shape {embedding.get_shape()}"
            )
        for start in range(0, vocab_size, EMBEDDING_BATCH_ROWS):
            end = min(start + EMBEDDING_BATCH_ROWS, vocab_size)
            rows = embedding[start:end].to(torch.float32).contiguous().numpy()
            sink.write(rows)
            if start % (EMBEDDING_BATCH_ROWS * 32) == 0:
                print(f"embedding rows {start:,}/{vocab_size:,}", flush=True)
    sink.finish()


def replace_rotary_cache(
    initializers: dict[str, onnx.TensorProto],
    writer,
    max_positions: int,
    head_dim: int,
    rope_theta: float,
) -> None:
    inv_frequency = 1.0 / (
        rope_theta
        ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    frequencies = np.outer(
        np.arange(max_positions, dtype=np.float32),
        inv_frequency,
    )
    writer.write_array(
        initializers["cos_cache"],
        np.cos(frequencies).astype(np.float32),
    )
    writer.write_array(
        initializers["sin_cache"],
        np.sin(frequencies).astype(np.float32),
    )


def copy_support_files(source: Path, output: Path, chunk_count: int) -> None:
    for file_name in (
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        path = source / file_name
        if path.is_file():
            shutil.copy2(path, output / file_name)

    template_path = output / "chat_template.jinja"
    tokenizer_config_path = output / "tokenizer_config.json"
    if not template_path.is_file() or not tokenizer_config_path.is_file():
        raise RuntimeError(
            "Planner tokenizer source must include chat_template.jinja and "
            "tokenizer_config.json"
        )
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["chat_template"] = template_path.read_text()
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n"
    )

    config = json.loads((source / "config.json").read_text())
    config["dtype"] = "float32"
    config["torch_dtype"] = "float32"
    config["transformers.js_config"] = {
        "dtype": "fp32",
        "kv_cache_dtype": {"fp32": "float32"},
        "use_external_data_format": {"model.onnx": chunk_count},
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def validate_serialized_graph(
    graph_path: Path,
    vocab_size: int,
    *,
    body_only: bool,
) -> None:
    serialized = onnx.load(graph_path, load_external_data=False)
    if body_only:
        if serialized.graph.output[0].name != "last_hidden_state":
            raise RuntimeError("Body graph does not expose last_hidden_state")
        input_names = {value.name for value in serialized.graph.input}
        if not {"input_ids", "inputs_embeds"}.issubset(input_names):
            raise RuntimeError(
                "Body graph must expose input_ids for shape calculations and "
                "inputs_embeds for the external embedding lookup"
            )
        if any(
            value.name == "model.embed_tokens.weight"
            for value in serialized.graph.initializer
        ):
            raise RuntimeError("Body graph still contains the tied embedding")
    else:
        embedding = next(
            value
            for value in serialized.graph.initializer
            if value.name == "model.embed_tokens.weight"
        )
        if embedding.data_type != TensorProto.FLOAT:
            raise RuntimeError("Serialized embedding is not FP32")
        if list(embedding.dims) != [vocab_size, 2560]:
            raise RuntimeError(
                f"Unexpected serialized embedding shape {embedding.dims}"
            )
        output_shape = serialized.graph.output[0].type.tensor_type.shape
        if output_shape.dim[-1].dim_value != vocab_size:
            raise RuntimeError(
                "Serialized logits output has the wrong vocabulary size"
            )
    for value_info in (
        list(serialized.graph.input)
        + list(serialized.graph.output)
        + list(serialized.graph.value_info)
    ):
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT16:
            raise RuntimeError(
                f"{value_info.name} still declares an FP16 tensor type"
            )
    for tensor in serialized.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT16:
            raise RuntimeError(f"{tensor.name} is still an FP16 initializer")
        has_external = bool(tensor.external_data)
        has_embedded = bool(
            tensor.raw_data
            or tensor.float_data
            or tensor.int32_data
            or tensor.int64_data
            or tensor.double_data
            or tensor.uint64_data
        )
        if has_external == has_embedded:
            raise RuntimeError(
                f"{tensor.name} must have exactly one serialized value source"
            )
    try:
        onnx.checker.check_model(str(graph_path), full_check=False)
    except onnx.checker.ValidationError as error:
        if "No Op registered for SimplifiedLayerNormalization" not in str(error):
            raise
        print(
            "ONNX checker reached the known upstream "
            "SimplifiedLayerNormalization schema limitation.",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    vocab_size = int(config["vocab_size"])
    hidden_size = int(config["hidden_size"])
    layer_count = int(config["num_hidden_layers"])
    if layer_count != 36 or hidden_size != 2560:
        raise RuntimeError(
            "The selected graph template is only valid for the 36-layer, "
            "2560-hidden ACE planner"
        )

    output = args.output.resolve()
    onnx_dir = output / "onnx"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(args.template, load_external_data=False)
    if args.body_only:
        model = expose_transformer_body(model, hidden_size)
    promote_graph_to_fp32(model)
    checkpoint = FP16_BUILDER.Checkpoint(args.checkpoint)
    writer = FP16_BUILDER.ExternalDataWriter(
        onnx_dir,
        "model.onnx_data",
        args.chunk_mib * 1024 * 1024,
    )
    initializers = {value.name: value for value in model.graph.initializer}

    if args.body_only:
        # The browser loads embedding rows and evaluates the sharded output
        # head outside ONNX Runtime, so this learned tensor is deliberately
        # absent from the transformer-body graph.
        checkpoint.used.add("model.embed_tokens.weight")
    else:
        replace_embedding(
            initializers["model.embed_tokens.weight"],
            checkpoint,
            writer,
            vocab_size,
            hidden_size,
        )
    replace_rotary_cache(
        initializers,
        writer,
        int(config["max_position_embeddings"]),
        int(config["head_dim"]),
        float(config["rope_theta"]),
    )

    for initializer in model.graph.initializer:
        source_name = FP16_BUILDER.source_name_for_norm(initializer.name)
        if source_name:
            value = (
                checkpoint.tensor(source_name)
                .to(torch.float32)
                .contiguous()
                .numpy()
            )
            FP16_BUILDER.set_raw_tensor(initializer, value)
            continue
        source_name = FP16_BUILDER.source_name_for_projection(initializer.name)
        if source_name:
            print(f"writing {source_name}", flush=True)
            value = (
                checkpoint.tensor(source_name)
                .to(torch.float32)
                .transpose(0, 1)
                .contiguous()
                .numpy()
            )
            writer.write_array(initializer, value)
            del value
            gc.collect()

    checkpoint.assert_fully_consumed()
    writer.close()

    if not args.body_only:
        logits_shape = model.graph.output[0].type.tensor_type.shape
        logits_shape.dim[-1].ClearField("dim_param")
        logits_shape.dim[-1].dim_value = vocab_size
    model.doc_string = (
        "ACE-Step 1.5 5 Hz LM 4B planner. Unquantized FP32 "
        + ("transformer-body " if args.body_only else "")
        + "diagnostic export for browser WebGPU."
    )
    graph_path = onnx_dir / "model.onnx"
    onnx.save_model(model, graph_path)
    copy_support_files(args.tokenizer_source, output, len(writer.files))
    validate_serialized_graph(
        graph_path,
        vocab_size,
        body_only=args.body_only,
    )

    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "source_precision": "bfloat16",
        "template_graph": str(args.template),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "layers": layer_count,
        "precision": "float32",
        "quantized": False,
        "body_only": args.body_only,
        "external_data_files": [
            {"name": path.name, "bytes": size}
            for path, size in zip(writer.files, writer.sizes, strict=True)
        ],
    }
    (output / "conversion-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    total_bytes = graph_path.stat().st_size + sum(writer.sizes)
    print(
        f"Built {graph_path} with {len(writer.files)} data chunks "
        f"({total_bytes / 1e9:.2f} GB total).",
        flush=True,
    )


if __name__ == "__main__":
    main()
