#!/usr/bin/env python3
"""Build a full-precision ACE-Step 5 Hz planner for Transformers.js/WebGPU.

The public Qwen3-4B Transformers.js FP16 graph has the same decoder
architecture as the ACE-Step planner, but a smaller vocabulary. This script
uses that graph only as an execution template and streams every learned tensor
from the official ACE-Step checkpoint into new external-data files.

Unlike the experimental INT4 conversion, this export is intended to be a
numerical-parity reference. Do not publish or select it in the browser until
the first-step logit ranking has been compared with the Python/MLX planner.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto
from safetensors import safe_open


DEFAULT_CHUNK_MIB = 1800
EMBEDDING_BATCH_ROWS = 2048

PROJECTION_NAME = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\."
    r"(?P<section>attn|mlp)\.(?P<projection>q_proj|k_proj|v_proj|o_proj|"
    r"gate_proj|up_proj|down_proj)\.MatMul\.weight$"
)
NORM_NAME = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\."
    r"(?P<name>input_layernorm|post_attention_layernorm)\.weight$"
)
ATTENTION_NORM_NAME = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.attn\."
    r"(?P<name>q_norm|k_norm)\.layernorm\.weight$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--tokenizer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_MIB)
    return parser.parse_args()


def clear_tensor_storage(tensor: onnx.TensorProto) -> None:
    tensor.ClearField("raw_data")
    tensor.ClearField("float_data")
    tensor.ClearField("int32_data")
    tensor.ClearField("int64_data")
    tensor.ClearField("double_data")
    tensor.ClearField("uint64_data")
    tensor.ClearField("external_data")
    tensor.data_location = TensorProto.DEFAULT


def set_raw_tensor(tensor: onnx.TensorProto, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    clear_tensor_storage(tensor)
    tensor.data_type = onnx.helper.np_dtype_to_tensor_dtype(contiguous.dtype)
    del tensor.dims[:]
    tensor.dims.extend(contiguous.shape)
    tensor.raw_data = contiguous.tobytes()


@dataclass
class ExternalSink:
    file_handle: object
    expected_bytes: int
    written_bytes: int = 0

    def write(self, value: np.ndarray | bytes | memoryview) -> None:
        payload = value if isinstance(value, (bytes, memoryview)) else memoryview(value)
        byte_count = len(payload) if isinstance(payload, bytes) else payload.nbytes
        if self.written_bytes + byte_count > self.expected_bytes:
            raise RuntimeError("External initializer wrote beyond its reservation")
        self.file_handle.write(payload)
        self.written_bytes += byte_count

    def finish(self) -> None:
        if self.written_bytes != self.expected_bytes:
            raise RuntimeError(
                f"External initializer wrote {self.written_bytes} bytes; "
                f"expected {self.expected_bytes}"
            )


class ExternalDataWriter:
    def __init__(self, output_dir: Path, base_name: str, max_chunk_bytes: int):
        self.output_dir = output_dir
        self.base_name = base_name
        self.max_chunk_bytes = max_chunk_bytes
        self.files: list[Path] = []
        self.handles: list[object] = []
        self.sizes: list[int] = []
        self.current_index = -1

    def _new_file(self) -> None:
        index = len(self.files)
        suffix = "" if index == 0 else f"_{index}"
        path = self.output_dir / f"{self.base_name}{suffix}"
        self.files.append(path)
        self.handles.append(path.open("wb"))
        self.sizes.append(0)
        self.current_index = index

    def reserve(
        self,
        tensor: onnx.TensorProto,
        byte_count: int,
        *,
        force_new_file: bool = False,
    ) -> ExternalSink:
        if byte_count > self.max_chunk_bytes:
            raise ValueError(
                f"{tensor.name} needs {byte_count} bytes, above the configured "
                f"{self.max_chunk_bytes}-byte chunk limit"
            )
        if self.current_index < 0:
            self._new_file()
        elif force_new_file or (
            self.sizes[self.current_index] + byte_count > self.max_chunk_bytes
        ):
            self._new_file()
        index = self.current_index
        offset = self.sizes[index]
        self.sizes[index] += byte_count
        clear_tensor_storage(tensor)
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in (
            ("location", self.files[index].name),
            ("offset", str(offset)),
            ("length", str(byte_count)),
        ):
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = value
        handle = self.handles[index]
        handle.seek(offset)
        return ExternalSink(handle, byte_count)

    def write_array(
        self,
        tensor: onnx.TensorProto,
        array: np.ndarray,
        *,
        force_new_file: bool = False,
    ) -> None:
        contiguous = np.ascontiguousarray(array)
        tensor.data_type = onnx.helper.np_dtype_to_tensor_dtype(contiguous.dtype)
        del tensor.dims[:]
        tensor.dims.extend(contiguous.shape)
        sink = self.reserve(
            tensor, contiguous.nbytes, force_new_file=force_new_file
        )
        sink.write(contiguous)
        sink.finish()

    def close(self) -> None:
        for handle in self.handles:
            handle.flush()
            handle.close()


class Checkpoint:
    def __init__(self, root: Path):
        index_path = root / "model.safetensors.index.json"
        self.root = root
        self.weight_map: dict[str, str] = json.loads(index_path.read_text())[
            "weight_map"
        ]
        self.used: set[str] = set()

    def tensor(self, key: str) -> torch.Tensor:
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"Checkpoint is missing {key}")
        self.used.add(key)
        with safe_open(self.root / shard, framework="pt", device="cpu") as source:
            return source.get_tensor(key)

    def slice_source(self, key: str):
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"Checkpoint is missing {key}")
        self.used.add(key)
        return safe_open(self.root / shard, framework="pt", device="cpu")

    def assert_fully_consumed(self) -> None:
        unused = sorted(set(self.weight_map) - self.used)
        if unused:
            raise RuntimeError(
                "Not every learned checkpoint tensor was consumed: "
                + ", ".join(unused[:20])
            )


def source_name_for_norm(initializer_name: str) -> str | None:
    if initializer_name == "model.layers.36.final_norm_layernorm.weight":
        return "model.norm.weight"
    match = NORM_NAME.match(initializer_name)
    if match:
        return (
            f"model.layers.{match['layer']}."
            f"{match['name']}.weight"
        )
    match = ATTENTION_NORM_NAME.match(initializer_name)
    if match:
        return (
            f"model.layers.{match['layer']}.self_attn."
            f"{match['name']}.weight"
        )
    return None


def source_name_for_projection(initializer_name: str) -> str | None:
    match = PROJECTION_NAME.match(initializer_name)
    if not match:
        return None
    section = "self_attn" if match["section"] == "attn" else "mlp"
    return (
        f"model.layers.{match['layer']}.{section}."
        f"{match['projection']}.weight"
    )


def replace_embedding(
    initializer: onnx.TensorProto,
    checkpoint: Checkpoint,
    writer: ExternalDataWriter,
    vocab_size: int,
    hidden_size: int,
) -> None:
    byte_count = vocab_size * hidden_size * np.dtype(np.float16).itemsize
    initializer.data_type = TensorProto.FLOAT16
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
            rows = (
                embedding[start:end]
                .to(torch.float16)
                .contiguous()
                .numpy()
            )
            sink.write(rows)
            if start % (EMBEDDING_BATCH_ROWS * 16) == 0:
                print(f"embedding rows {start:,}/{vocab_size:,}", flush=True)
    sink.finish()


def replace_rotary_cache(
    initializers: dict[str, onnx.TensorProto],
    writer: ExternalDataWriter,
    max_positions: int,
    head_dim: int,
    rope_theta: float,
) -> None:
    inv_frequency = 1.0 / (
        rope_theta
        ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    frequencies = np.outer(
        np.arange(max_positions, dtype=np.float32), inv_frequency
    )
    writer.write_array(initializers["cos_cache"], np.cos(frequencies).astype(np.float16))
    writer.write_array(initializers["sin_cache"], np.sin(frequencies).astype(np.float16))


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
    config["torch_dtype"] = "float16"
    config["transformers.js_config"] = {
        "dtype": "fp16",
        "kv_cache_dtype": {"fp16": "float16"},
        "use_external_data_format": {"model_fp16.onnx": chunk_count},
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def validate_serialized_graph(graph_path: Path, vocab_size: int) -> None:
    serialized = onnx.load(graph_path, load_external_data=False)
    embedding = next(
        value
        for value in serialized.graph.initializer
        if value.name == "model.embed_tokens.weight"
    )
    if list(embedding.dims) != [vocab_size, 2560]:
        raise RuntimeError(f"Unexpected serialized embedding shape {embedding.dims}")
    output_shape = serialized.graph.output[0].type.tensor_type.shape
    if output_shape.dim[-1].dim_value != vocab_size:
        raise RuntimeError("Serialized logits output has the wrong vocabulary size")
    for tensor in serialized.graph.initializer:
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
    checkpoint = Checkpoint(args.checkpoint)
    writer = ExternalDataWriter(
        onnx_dir,
        "model_fp16.onnx_data",
        args.chunk_mib * 1024 * 1024,
    )
    initializers = {value.name: value for value in model.graph.initializer}

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
        source_name = source_name_for_norm(initializer.name)
        if source_name:
            value = (
                checkpoint.tensor(source_name)
                .to(torch.float16)
                .contiguous()
                .numpy()
            )
            set_raw_tensor(initializer, value)
            continue
        source_name = source_name_for_projection(initializer.name)
        if source_name:
            print(f"writing {source_name}", flush=True)
            value = (
                checkpoint.tensor(source_name)
                .to(torch.float16)
                .transpose(0, 1)
                .contiguous()
                .numpy()
            )
            writer.write_array(initializer, value)
            del value
            gc.collect()

    checkpoint.assert_fully_consumed()
    writer.close()

    logits_shape = model.graph.output[0].type.tensor_type.shape
    logits_shape.dim[-1].ClearField("dim_param")
    logits_shape.dim[-1].dim_value = vocab_size
    model.doc_string = (
        "ACE-Step 1.5 5 Hz LM 4B planner. Full FP16 parity-reference export "
        "for Transformers.js/WebGPU."
    )
    graph_path = onnx_dir / "model_fp16.onnx"
    onnx.save_model(model, graph_path)
    copy_support_files(args.tokenizer_source, output, len(writer.files))
    validate_serialized_graph(graph_path, vocab_size)

    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "template_graph": str(args.template),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "layers": layer_count,
        "precision": "float16",
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
