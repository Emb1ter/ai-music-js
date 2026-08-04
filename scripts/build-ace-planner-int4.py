#!/usr/bin/env python3
"""Build the ACE-Step 5 Hz planner without materializing a full FP16 ONNX model.

The ACE planner is a Qwen3 4B fine-tune whose architecture matches the public
Qwen3-4B Transformers.js ONNX graph, except for ACE's larger vocabulary. This
script uses that optimized graph as a structural template, replaces every
learned tensor from the official ACE safetensors, and quantizes one layer at a
time. The embedding and tied LM head are both INT4 so neither creates a single
>1 GiB WebGPU buffer.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits
from safetensors import safe_open


BLOCK_SIZE = 32
BITS = 4
DEFAULT_CHUNK_BYTES = 450 * 1024 * 1024
EMBEDDING_BATCH_ROWS = 1024

Q4_NAME = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\."
    r"(?P<section>attn|mlp)\.(?P<projection>q_proj|k_proj|v_proj|o_proj|"
    r"gate_proj|up_proj|down_proj)\.MatMul\.weight_Q4$"
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
    parser.add_argument("--chunk-mib", type=int, default=450)
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="Use learned per-block UINT4 zero points for transformer weights.",
    )
    parser.add_argument(
        "--fp16-embedding-head",
        action="store_true",
        help="Keep the tied token embedding and output head in FP16.",
    )
    return parser.parse_args()


def load_fp16_builder():
    module_path = Path(__file__).with_name("build-ace-planner-fp16.py")
    spec = importlib.util.spec_from_file_location(
        "ace_planner_fp16_builder_for_q4",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clear_tensor_storage(tensor: onnx.TensorProto) -> None:
    tensor.ClearField("raw_data")
    tensor.ClearField("float_data")
    tensor.ClearField("int32_data")
    tensor.ClearField("int64_data")
    tensor.ClearField("double_data")
    tensor.ClearField("uint64_data")
    tensor.ClearField("external_data")
    tensor.data_location = TensorProto.DEFAULT


def set_raw_tensor(
    tensor: onnx.TensorProto,
    array: np.ndarray,
    data_type: int | None = None,
) -> None:
    clear_tensor_storage(tensor)
    tensor.data_type = data_type or onnx.helper.np_dtype_to_tensor_dtype(array.dtype)
    del tensor.dims[:]
    tensor.dims.extend(array.shape)
    tensor.raw_data = np.ascontiguousarray(array).tobytes()


def blank_tensor(name: str, data_type: int, dims: list[int]) -> onnx.TensorProto:
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.data_type = data_type
    tensor.dims.extend(dims)
    return tensor


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
        self.root = root
        index_path = root / "model.safetensors.index.json"
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
    match = Q4_NAME.match(initializer_name)
    if not match:
        return None
    section = "self_attn" if match["section"] == "attn" else "mlp"
    return (
        f"model.layers.{match['layer']}.{section}."
        f"{match['projection']}.weight"
    )


def quantize_matmul_weight(
    weight: torch.Tensor,
    *,
    asymmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    source = np.ascontiguousarray(weight.to(torch.float16).numpy().T)
    rows, cols = source.shape
    blocks = math.ceil(rows / BLOCK_SIZE)
    if rows % BLOCK_SIZE:
        source = np.pad(source, ((0, blocks * BLOCK_SIZE - rows), (0, 0)))
    packed = np.zeros((cols, blocks, BLOCK_SIZE // 2), dtype=np.uint8)
    scales = np.zeros((cols, blocks), dtype=np.float16)
    zero_points = np.zeros((cols, math.ceil(blocks / 2)), dtype=np.uint8)
    quantize_matmul_4bits(
        packed,
        source,
        scales,
        zero_points,
        BLOCK_SIZE,
        cols,
        rows,
        not asymmetric,
    )
    return packed, scales, zero_points if asymmetric else None


def quantize_embedding_rows(
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count, width = rows.shape
    if width % BLOCK_SIZE:
        raise ValueError("The planner embedding width must be divisible by 32")
    block_count = width // BLOCK_SIZE
    blocks = rows.reshape(row_count, block_count, BLOCK_SIZE)
    maximum = blocks.max(axis=2)
    minimum = blocks.min(axis=2)
    abs_maximum = np.where(
        np.abs(maximum) > np.abs(minimum), maximum, minimum
    )
    scales = (abs_maximum / np.float16(-8.0)).astype(np.float16)
    normalized = np.divide(
        blocks,
        scales[..., None],
        out=np.zeros_like(blocks),
        where=scales[..., None] != 0,
    )
    np.rint(normalized, out=normalized)
    np.clip(normalized, -8, 7, out=normalized)
    quantized = normalized.astype(np.int8)
    flat = quantized.reshape(-1)
    gather_packed = (
        (flat[::2].astype(np.uint8) & 0x0F)
        | ((flat[1::2].astype(np.uint8) & 0x0F) << 4)
    )

    matmul_source = np.ascontiguousarray(rows.T)
    matmul_packed = np.zeros(
        (row_count, block_count, BLOCK_SIZE // 2), dtype=np.uint8
    )
    matmul_scales = np.zeros((row_count, block_count), dtype=np.float16)
    zero_points = np.zeros(
        (row_count, math.ceil(block_count / 2)), dtype=np.uint8
    )
    quantize_matmul_4bits(
        matmul_packed,
        matmul_source,
        matmul_scales,
        zero_points,
        BLOCK_SIZE,
        row_count,
        width,
        True,
    )
    if not np.array_equal(scales, matmul_scales):
        raise RuntimeError("Gather and MatMul quantizers produced different scales")
    return gather_packed, matmul_packed, scales


def replace_embedding_and_head(
    model: onnx.ModelProto,
    checkpoint: Checkpoint,
    writer: ExternalDataWriter,
    vocab_size: int,
    hidden_size: int,
) -> None:
    initializers = {value.name: value for value in model.graph.initializer}
    original = initializers["model.embed_tokens.weight"]
    model.graph.initializer.remove(original)

    block_count = hidden_size // BLOCK_SIZE
    gather_weight = blank_tensor(
        "model.embed_tokens.weight_Q4",
        TensorProto.INT4,
        [vocab_size, hidden_size],
    )
    head_weight = blank_tensor(
        "lm_head.weight_Q4",
        TensorProto.UINT8,
        [vocab_size, block_count, BLOCK_SIZE // 2],
    )
    shared_scales = blank_tensor(
        "model.embed_tokens.weight_scales",
        TensorProto.FLOAT16,
        [vocab_size, block_count],
    )
    packed_bytes = vocab_size * hidden_size // 2
    scales_bytes = vocab_size * block_count * np.dtype(np.float16).itemsize
    gather_sink = writer.reserve(
        gather_weight, packed_bytes, force_new_file=True
    )
    head_sink = writer.reserve(head_weight, packed_bytes, force_new_file=True)
    scales_sink = writer.reserve(
        shared_scales, scales_bytes, force_new_file=True
    )
    # protobuf repeated-message fields copy appended values. Reserve external
    # storage first so the copies placed in the graph retain that metadata.
    model.graph.initializer.extend([gather_weight, head_weight, shared_scales])

    source_context = checkpoint.slice_source("model.embed_tokens.weight")
    with source_context as source:
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
            gather, head, scales = quantize_embedding_rows(rows)
            gather_sink.write(gather)
            head_sink.write(head)
            scales_sink.write(scales)
            if start % (EMBEDDING_BATCH_ROWS * 16) == 0:
                print(f"embedding rows {start:,}/{vocab_size:,}", flush=True)
    gather_sink.finish()
    head_sink.finish()
    scales_sink.finish()

    nodes: list[onnx.NodeProto] = []
    saw_gather = False
    saw_transpose = False
    saw_head = False
    for node in model.graph.node:
        if node.name == "/model/embed_tokens/Gather":
            nodes.append(
                helper.make_node(
                    "GatherBlockQuantized",
                    [gather_weight.name, node.input[1], shared_scales.name],
                    list(node.output),
                    name="/model/embed_tokens/Gather_Q4",
                    domain="com.microsoft",
                    gather_axis=0,
                    quantize_axis=1,
                    block_size=BLOCK_SIZE,
                )
            )
            saw_gather = True
        elif node.name == "Transpose_778":
            saw_transpose = True
        elif node.name == "/lm_head/MatMul":
            nodes.append(
                helper.make_node(
                    "MatMulNBits",
                    [node.input[0], head_weight.name, shared_scales.name],
                    list(node.output),
                    name="/lm_head/MatMul_Q4",
                    domain="com.microsoft",
                    K=hidden_size,
                    N=vocab_size,
                    bits=BITS,
                    block_size=BLOCK_SIZE,
                )
            )
            saw_head = True
        else:
            nodes.append(node)
    if not (saw_gather and saw_transpose and saw_head):
        raise RuntimeError("Template embedding/head nodes did not match")
    del model.graph.node[:]
    model.graph.node.extend(nodes)


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


def copy_tokenizer_files(source: Path, output: Path, chunk_count: int) -> None:
    tokenizer_files = (
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    for file_name in tokenizer_files:
        path = source / file_name
        if path.is_file():
            shutil.copy2(path, output / file_name)

    # Transformers.js loads chat templates from tokenizer_config.json. Unlike
    # current Python Transformers, it does not automatically fall back to the
    # sibling chat_template.jinja file for AutoTokenizer. Keep the standalone
    # file for Python clients and embed the exact same template for the browser.
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
        "dtype": "q4f16",
        "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
        "use_external_data_format": {"model_q4f16.onnx": chunk_count},
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def validate_serialized_graph(graph_path: Path) -> None:
    serialized = onnx.load(graph_path, load_external_data=False)
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
        if has_external and tensor.data_location != TensorProto.EXTERNAL:
            raise RuntimeError(
                f"{tensor.name} has external data without EXTERNAL data_location"
            )
    try:
        onnx.checker.check_model(str(graph_path), full_check=False)
    except onnx.checker.ValidationError as error:
        # The upstream Transformers.js Qwen3 graph intentionally uses ORT's
        # fused SimplifiedLayerNormalization in the default domain. ORT Web
        # accepts it, while the generic ONNX checker has no schema for it.
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
    if output.exists():
        existing = list(output.iterdir())
        if existing:
            raise RuntimeError(
                f"Refusing to overwrite non-empty output directory: {output}"
            )
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(args.template, load_external_data=False)
    checkpoint = Checkpoint(args.checkpoint)
    writer = ExternalDataWriter(
        onnx_dir,
        "model_q4f16.onnx_data",
        args.chunk_mib * 1024 * 1024,
    )

    if args.fp16_embedding_head:
        fp16_builder = load_fp16_builder()
        fp16_builder.replace_embedding(
            {
                value.name: value
                for value in model.graph.initializer
            }["model.embed_tokens.weight"],
            checkpoint,
            writer,
            vocab_size,
            hidden_size,
        )
    else:
        replace_embedding_and_head(
            model, checkpoint, writer, vocab_size, hidden_size
        )
    initializers = {value.name: value for value in model.graph.initializer}
    replace_rotary_cache(
        initializers,
        writer,
        int(config["max_position_embeddings"]),
        int(config["head_dim"]),
        float(config["rope_theta"]),
    )

    zero_point_initializers: list[onnx.TensorProto] = []
    nodes_by_weight = {
        node.input[1]: node
        for node in model.graph.node
        if node.op_type == "MatMulNBits" and len(node.input) >= 2
    }
    for initializer in list(model.graph.initializer):
        source_name = source_name_for_norm(initializer.name)
        if source_name:
            value = (
                checkpoint.tensor(source_name)
                .to(torch.float16)
                .contiguous()
                .numpy()
            )
            set_raw_tensor(initializer, value, TensorProto.FLOAT16)
            continue
        source_name = source_name_for_projection(initializer.name)
        if source_name:
            print(f"quantizing {source_name}", flush=True)
            packed, scales, zero_points = quantize_matmul_weight(
                checkpoint.tensor(source_name),
                asymmetric=args.asymmetric,
            )
            writer.write_array(initializer, packed)
            scale_name = initializer.name.replace(
                ".weight_Q4", ".weight_scales"
            )
            scale_initializer = initializers[scale_name]
            writer.write_array(scale_initializer, scales.reshape(-1))
            if zero_points is not None:
                node = nodes_by_weight.get(initializer.name)
                if node is None:
                    raise RuntimeError(
                        f"No MatMulNBits node consumes {initializer.name}"
                    )
                zero_name = initializer.name.replace(
                    ".weight_Q4",
                    ".weight_zero_points",
                )
                zero_initializer = blank_tensor(
                    zero_name,
                    TensorProto.UINT8,
                    list(zero_points.shape),
                )
                writer.write_array(zero_initializer, zero_points)
                zero_point_initializers.append(zero_initializer)
                node.input.append(zero_name)
            del packed, scales, zero_points
            gc.collect()

    model.graph.initializer.extend(zero_point_initializers)
    checkpoint.assert_fully_consumed()
    writer.close()

    logits_shape = model.graph.output[0].type.tensor_type.shape
    logits_shape.dim[-1].ClearField("dim_param")
    logits_shape.dim[-1].dim_value = vocab_size
    model.doc_string = (
        "ACE-Step 1.5 5 Hz LM 4B planner. Layer-streamed "
        f"{'asymmetric' if args.asymmetric else 'symmetric'} INT4 conversion "
        "for Transformers.js/WebGPU; tied embedding/head precision is "
        f"{'FP16' if args.fp16_embedding_head else 'INT4'}."
    )
    graph_path = onnx_dir / "model_q4f16.onnx"
    onnx.save_model(model, graph_path)
    copy_tokenizer_files(args.tokenizer_source, output, len(writer.files))

    validate_serialized_graph(graph_path)
    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "template_graph": str(args.template),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "layers": layer_count,
        "quantization": {
            "bits": BITS,
            "block_size": BLOCK_SIZE,
            "symmetric": not args.asymmetric,
            "activation_dtype": "float16",
            "embedding_operator": (
                "Gather"
                if args.fp16_embedding_head
                else "GatherBlockQuantized"
            ),
            "projection_operator": "MatMulNBits",
            "zero_points": (
                "per-block packed UINT4"
                if args.asymmetric
                else "implicit midpoint"
            ),
            "embedding_and_tied_head": (
                "float16"
                if args.fp16_embedding_head
                else "blockwise int4"
            ),
        },
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
