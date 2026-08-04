#!/usr/bin/env python3
"""Build a browser-feasible INT8/FP16 ACE-Step 5 Hz planner.

The generic Q4 conversion is small enough for browsers but does not preserve
the ACE planner's semantic-token ranking. The full FP16 graph does preserve
ranking, but its 8.43 GB external-data set causes Chromium renderer reloads
while Transformers.js assembles the model.

This export keeps the tied token embedding/head in FP16 and stores every
transformer projection as symmetric blockwise INT8. WebGPU expands one
projection at a time using Cast + Reshape + Mul immediately before MatMul,
avoiding both unsupported 8-bit MatMulNBits and a second full-precision weight
copy in the downloadable artifact.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper


def load_fp16_builder():
    module_path = Path(__file__).with_name("build-ace-planner-fp16.py")
    spec = importlib.util.spec_from_file_location(
        "ace_planner_fp16_builder",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FP16_BUILDER = load_fp16_builder()


def load_int4_builder():
    module_path = Path(__file__).with_name("build-ace-planner-int4.py")
    spec = importlib.util.spec_from_file_location(
        "ace_planner_int4_builder",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INT4_BUILDER = load_int4_builder()
DEFAULT_CHUNK_MIB = 1200
BLOCK_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Residual-scaled FP16 parity graph; external weights are not read.",
    )
    parser.add_argument("--tokenizer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_MIB)
    parser.add_argument(
        "--q4-embedding-head",
        action="store_true",
        help=(
            "Use the browser-native blockwise Q4 Gather/MatMulNBits for the "
            "large tied embedding/head while retaining blockwise INT8 body."
        ),
    )
    parser.add_argument(
        "--q8-embedding-head",
        action="store_true",
        help=(
            "Store the tied embedding/head as blockwise INT8 and dequantize "
            "it with WebGPU-compatible standard operators."
        ),
    )
    parser.add_argument(
        "--q4-projections",
        default="",
        help=(
            "Comma-separated projection names to store with MatMulNBits Q4 "
            "(for example gate_proj,up_proj) to keep the graph below the "
            "ONNX Runtime Web 4 GB session ceiling."
        ),
    )
    parser.add_argument(
        "--q6-body",
        action="store_true",
        help=(
            "Pack all transformer projection weights at 6 bits and unpack "
            "them with standard WebGPU ONNX arithmetic before MatMul."
        ),
    )
    parser.add_argument(
        "--q5-body",
        action="store_true",
        help=(
            "Pack all transformer projection weights at 5 bits and unpack "
            "them with standard WebGPU ONNX arithmetic before MatMul."
        ),
    )
    return parser.parse_args()


def quantize_projection(
    weight: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    # PyTorch Linear is [output, input]; ONNX MatMul is [input, output].
    source = (
        weight.to(torch.float32)
        .transpose(0, 1)
        .contiguous()
        .numpy()
    )
    input_width, output_width = source.shape
    if input_width % BLOCK_SIZE:
        raise RuntimeError(
            f"Projection input width {input_width} is not divisible by "
            f"{BLOCK_SIZE}"
        )
    blocks = source.reshape(
        input_width // BLOCK_SIZE,
        BLOCK_SIZE,
        output_width,
    )
    maximum = np.max(np.abs(blocks), axis=1)
    scale = np.maximum(maximum / np.float32(127.0), np.finfo(np.float16).tiny)
    # Quantize against the exact FP16 scale the browser will multiply by.
    scale_fp16 = scale.astype(np.float16)
    quantized = np.rint(
        blocks / scale_fp16.astype(np.float32)[:, None, :]
    )
    np.clip(quantized, -127, 127, out=quantized)
    return (
        quantized.astype(np.int8).reshape(input_width, output_width),
        scale_fp16[:, None, :],
    )


def quantize_projection_q6(
    weight: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    source = (
        weight.to(torch.float32)
        .transpose(0, 1)
        .contiguous()
        .numpy()
    )
    input_width, output_width = source.shape
    if input_width % BLOCK_SIZE:
        raise RuntimeError(
            f"Projection input width {input_width} is not divisible by "
            f"{BLOCK_SIZE}"
        )
    blocks = source.reshape(
        input_width // BLOCK_SIZE,
        BLOCK_SIZE,
        output_width,
    )
    maximum = np.max(np.abs(blocks), axis=1)
    scale = np.maximum(maximum / np.float32(31.0), np.finfo(np.float16).tiny)
    scale_fp16 = scale.astype(np.float16)
    quantized = np.rint(
        blocks / scale_fp16.astype(np.float32)[:, None, :]
    )
    np.clip(quantized, -32, 31, out=quantized)
    unsigned = (
        quantized.astype(np.int16) + np.int16(32)
    ).astype(np.uint16).reshape(input_width // 4, 4, output_width)
    q0, q1, q2, q3 = (
        unsigned[:, 0],
        unsigned[:, 1],
        unsigned[:, 2],
        unsigned[:, 3],
    )
    packed = np.stack(
        [
            q0 | ((q1 & 0x03) << 6),
            (q1 >> 2) | ((q2 & 0x0F) << 4),
            (q2 >> 4) | (q3 << 2),
        ],
        axis=1,
    ).astype(np.uint8)
    return packed, scale_fp16[:, None, :]


def quantize_projection_q5(
    weight: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    source = (
        weight.to(torch.float32)
        .transpose(0, 1)
        .contiguous()
        .numpy()
    )
    input_width, output_width = source.shape
    if input_width % BLOCK_SIZE:
        raise RuntimeError(
            f"Projection input width {input_width} is not divisible by "
            f"{BLOCK_SIZE}"
        )
    blocks = source.reshape(
        input_width // BLOCK_SIZE,
        BLOCK_SIZE,
        output_width,
    )
    maximum = np.max(np.abs(blocks), axis=1)
    scale = np.maximum(maximum / np.float32(15.0), np.finfo(np.float16).tiny)
    scale_fp16 = scale.astype(np.float16)
    quantized = np.rint(
        blocks / scale_fp16.astype(np.float32)[:, None, :]
    )
    np.clip(quantized, -16, 15, out=quantized)
    unsigned = (
        quantized.astype(np.int16) + np.int16(16)
    ).astype(np.uint16).reshape(input_width // 8, 8, output_width)
    q0, q1, q2, q3, q4, q5, q6, q7 = (
        unsigned[:, index] for index in range(8)
    )
    packed = np.stack(
        [
            q0 | ((q1 & 0x07) << 5),
            (q1 >> 3) | (q2 << 2) | ((q3 & 0x01) << 7),
            (q3 >> 1) | ((q4 & 0x0F) << 4),
            (q4 >> 4) | (q5 << 1) | ((q6 & 0x03) << 6),
            (q6 >> 2) | (q7 << 3),
        ],
        axis=1,
    ).astype(np.uint8)
    return packed, scale_fp16[:, None, :]


def replace_embedding_with_q8(
    model: onnx.ModelProto,
    checkpoint,
    writer,
    vocab_size: int,
    hidden_size: int,
) -> None:
    initializers = {value.name: value for value in model.graph.initializer}
    original = initializers["model.embed_tokens.weight"]
    model.graph.initializer.remove(original)

    block_count = hidden_size // BLOCK_SIZE
    quantized_weight = INT4_BUILDER.blank_tensor(
        "model.embed_tokens.weight.q8",
        TensorProto.INT8,
        [vocab_size, hidden_size],
    )
    scales = INT4_BUILDER.blank_tensor(
        "model.embed_tokens.weight.q8_scale",
        TensorProto.FLOAT16,
        [vocab_size, block_count, 1],
    )
    weight_sink = writer.reserve(
        quantized_weight,
        vocab_size * hidden_size,
        force_new_file=True,
    )
    scale_sink = writer.reserve(
        scales,
        vocab_size * block_count * np.dtype(np.float16).itemsize,
        force_new_file=True,
    )
    model.graph.initializer.extend([quantized_weight, scales])

    with checkpoint.slice_source("model.embed_tokens.weight") as source:
        embedding = source.get_slice("model.embed_tokens.weight")
        for start in range(0, vocab_size, 1024):
            end = min(start + 1024, vocab_size)
            rows = embedding[start:end].float().numpy()
            blocks = rows.reshape(end - start, block_count, BLOCK_SIZE)
            maximum = np.max(np.abs(blocks), axis=2)
            block_scales = np.maximum(
                maximum / np.float32(127.0),
                np.finfo(np.float16).tiny,
            ).astype(np.float16)
            quantized = np.rint(
                blocks / block_scales.astype(np.float32)[..., None]
            )
            np.clip(quantized, -127, 127, out=quantized)
            weight_sink.write(
                quantized.astype(np.int8).reshape(end - start, hidden_size)
            )
            scale_sink.write(block_scales[..., None])
            if start % (1024 * 16) == 0:
                print(f"embedding rows {start:,}/{vocab_size:,}", flush=True)
    weight_sink.finish()
    scale_sink.finish()

    blocked_shape = "model.embed_tokens.weight.q8_block_shape"
    restored_shape = "model.embed_tokens.weight.q8_weight_shape"
    model.graph.initializer.extend(
        [
            helper.make_tensor(
                blocked_shape,
                TensorProto.INT64,
                [3],
                [vocab_size, block_count, BLOCK_SIZE],
            ),
            helper.make_tensor(
                restored_shape,
                TensorProto.INT64,
                [2],
                [vocab_size, hidden_size],
            ),
        ]
    )
    nodes: list[onnx.NodeProto] = []
    inserted = False
    for node in model.graph.node:
        if node.name == "/model/embed_tokens/Gather":
            nodes.extend(
                [
                    helper.make_node(
                        "Cast",
                        [quantized_weight.name],
                        ["model.embed_tokens.weight.q8_fp16"],
                        name="/model/embed_tokens/CastQ8Weight",
                        to=TensorProto.FLOAT16,
                    ),
                    helper.make_node(
                        "Reshape",
                        [
                            "model.embed_tokens.weight.q8_fp16",
                            blocked_shape,
                        ],
                        ["model.embed_tokens.weight.q8_blocks"],
                        name="/model/embed_tokens/BlockQ8Weight",
                    ),
                    helper.make_node(
                        "Mul",
                        [
                            "model.embed_tokens.weight.q8_blocks",
                            scales.name,
                        ],
                        ["model.embed_tokens.weight.fp16_blocks"],
                        name="/model/embed_tokens/DequantizeQ8Weight",
                    ),
                    helper.make_node(
                        "Reshape",
                        [
                            "model.embed_tokens.weight.fp16_blocks",
                            restored_shape,
                        ],
                        ["model.embed_tokens.weight"],
                        name="/model/embed_tokens/RestoreQ8WeightShape",
                    ),
                ]
            )
            inserted = True
        nodes.append(node)
    if not inserted:
        raise RuntimeError("Did not find the token-embedding Gather node")
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def add_dequantization_nodes(
    model: onnx.ModelProto,
    replacements: dict[str, tuple[str, str, str, str]],
) -> None:
    nodes: list[onnx.NodeProto] = []
    inserted: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) >= 2:
            original_weight = node.input[1]
            replacement = replacements.get(original_weight)
            if replacement is not None:
                (
                    quantized_weight,
                    scale,
                    block_shape,
                    weight_shape,
                ) = replacement
                cast_weight = f"{original_weight}.q8_fp16"
                blocked_weight = f"{original_weight}.q8_blocks"
                dequantized_blocks = f"{original_weight}.fp16_blocks"
                nodes.extend(
                    [
                        helper.make_node(
                            "Cast",
                            [quantized_weight],
                            [cast_weight],
                            name=f"{node.name}/CastQ8Weight",
                            to=TensorProto.FLOAT16,
                        ),
                        helper.make_node(
                            "Reshape",
                            [cast_weight, block_shape],
                            [blocked_weight],
                            name=f"{node.name}/BlockQ8Weight",
                        ),
                        helper.make_node(
                            "Mul",
                            [blocked_weight, scale],
                            [dequantized_blocks],
                            name=f"{node.name}/DequantizeQ8Weight",
                        ),
                        helper.make_node(
                            "Reshape",
                            [dequantized_blocks, weight_shape],
                            [original_weight],
                            name=f"{node.name}/RestoreQ8WeightShape",
                        ),
                    ]
                )
                inserted.add(original_weight)
        nodes.append(node)
    missing = sorted(set(replacements) - inserted)
    if missing:
        raise RuntimeError(
            "No MatMul consumed these projection weights: "
            + ", ".join(missing[:10])
        )
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def replace_q4_projection_nodes(
    model: onnx.ModelProto,
    replacements: dict[str, tuple[str, str, int, int]],
) -> None:
    nodes: list[onnx.NodeProto] = []
    replaced: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) >= 2:
            original_weight = node.input[1]
            replacement = replacements.get(original_weight)
            if replacement is not None:
                quantized_weight, scale, input_width, output_width = replacement
                nodes.append(
                    helper.make_node(
                        "MatMulNBits",
                        [node.input[0], quantized_weight, scale],
                        list(node.output),
                        name=f"{node.name}/Q4",
                        domain="com.microsoft",
                        K=input_width,
                        N=output_width,
                        bits=4,
                        block_size=BLOCK_SIZE,
                    )
                )
                replaced.add(original_weight)
                continue
        nodes.append(node)
    missing = sorted(set(replacements) - replaced)
    if missing:
        raise RuntimeError(
            "No MatMul consumed these Q4 projection weights: "
            + ", ".join(missing[:10])
        )
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def replace_q6_projection_nodes(
    model: onnx.ModelProto,
    replacements: dict[str, tuple[str, str, str, str]],
) -> None:
    if not replacements:
        return
    constants = {
        "four": ("/model/q6/four", np.float16(4)),
        "sixteen": ("/model/q6/sixteen", np.float16(16)),
        "thirty_two": ("/model/q6/thirty_two", np.float16(32)),
        "sixty_four": ("/model/q6/sixty_four", np.float16(64)),
    }
    model.graph.initializer.extend(
        helper.make_tensor(
            name,
            TensorProto.FLOAT16,
            [],
            value.tobytes(),
            raw=True,
        )
        for name, value in constants.values()
    )
    c4 = constants["four"][0]
    c16 = constants["sixteen"][0]
    c32 = constants["thirty_two"][0]
    c64 = constants["sixty_four"][0]

    nodes: list[onnx.NodeProto] = []
    replaced: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            nodes.append(node)
            continue
        original_weight = node.input[1]
        replacement = replacements.get(original_weight)
        if replacement is None:
            nodes.append(node)
            continue

        packed, scale, blocked_shape, weight_shape = replacement
        prefix = f"{node.name}/Q6"
        b0 = f"{prefix}/b0"
        b1 = f"{prefix}/b1"
        b2 = f"{prefix}/b2"
        div_b0_64 = f"{prefix}/div_b0_64"
        div_b1_16 = f"{prefix}/div_b1_16"
        div_b2_4 = f"{prefix}/div_b2_4"
        mod_b0_64 = f"{prefix}/mod_b0_64"
        mod_b1_16 = f"{prefix}/mod_b1_16"
        mod_b2_4 = f"{prefix}/mod_b2_4"
        q1_high = f"{prefix}/q1_high"
        q2_high = f"{prefix}/q2_high"
        q1 = f"{prefix}/q1"
        q2 = f"{prefix}/q2"
        unpacked = f"{prefix}/unpacked"
        blocked = f"{prefix}/blocked"
        centered = f"{prefix}/centered"
        dequantized = f"{prefix}/dequantized"
        nodes.extend(
            [
                helper.make_node(
                    "Cast",
                    [packed],
                    [f"{prefix}/packed_fp16"],
                    name=f"{prefix}/CastPacked",
                    to=TensorProto.FLOAT16,
                ),
                helper.make_node(
                    "Split",
                    [f"{prefix}/packed_fp16"],
                    [b0, b1, b2],
                    name=f"{prefix}/SplitBytes",
                    axis=1,
                ),
                helper.make_node(
                    "Div", [b0, c64], [f"{prefix}/b0_div"], name=f"{prefix}/DivideB0"
                ),
                helper.make_node(
                    "Floor",
                    [f"{prefix}/b0_div"],
                    [div_b0_64],
                    name=f"{prefix}/FloorB0",
                ),
                helper.make_node(
                    "Mul",
                    [div_b0_64, c64],
                    [f"{prefix}/b0_high"],
                    name=f"{prefix}/ScaleB0High",
                ),
                helper.make_node(
                    "Sub",
                    [b0, f"{prefix}/b0_high"],
                    [mod_b0_64],
                    name=f"{prefix}/ModuloB0",
                ),
                helper.make_node(
                    "Div", [b1, c16], [f"{prefix}/b1_div"], name=f"{prefix}/DivideB1"
                ),
                helper.make_node(
                    "Floor",
                    [f"{prefix}/b1_div"],
                    [div_b1_16],
                    name=f"{prefix}/FloorB1",
                ),
                helper.make_node(
                    "Mul",
                    [div_b1_16, c16],
                    [f"{prefix}/b1_high"],
                    name=f"{prefix}/ScaleB1High",
                ),
                helper.make_node(
                    "Sub",
                    [b1, f"{prefix}/b1_high"],
                    [mod_b1_16],
                    name=f"{prefix}/ModuloB1",
                ),
                helper.make_node(
                    "Mul",
                    [mod_b1_16, c4],
                    [q1_high],
                    name=f"{prefix}/ScaleQ1High",
                ),
                helper.make_node(
                    "Add",
                    [div_b0_64, q1_high],
                    [q1],
                    name=f"{prefix}/CombineQ1",
                ),
                helper.make_node(
                    "Div", [b2, c4], [f"{prefix}/b2_div"], name=f"{prefix}/DivideB2"
                ),
                helper.make_node(
                    "Floor",
                    [f"{prefix}/b2_div"],
                    [div_b2_4],
                    name=f"{prefix}/FloorB2",
                ),
                helper.make_node(
                    "Mul",
                    [div_b2_4, c4],
                    [f"{prefix}/b2_high"],
                    name=f"{prefix}/ScaleB2High",
                ),
                helper.make_node(
                    "Sub",
                    [b2, f"{prefix}/b2_high"],
                    [mod_b2_4],
                    name=f"{prefix}/ModuloB2",
                ),
                helper.make_node(
                    "Mul",
                    [mod_b2_4, c16],
                    [q2_high],
                    name=f"{prefix}/ScaleQ2High",
                ),
                helper.make_node(
                    "Add",
                    [div_b1_16, q2_high],
                    [q2],
                    name=f"{prefix}/CombineQ2",
                ),
                helper.make_node(
                    "Concat",
                    [mod_b0_64, q1, q2, div_b2_4],
                    [unpacked],
                    name=f"{prefix}/ConcatQ6",
                    axis=1,
                ),
                helper.make_node(
                    "Reshape",
                    [unpacked, blocked_shape],
                    [blocked],
                    name=f"{prefix}/BlockQ6Weight",
                ),
                helper.make_node(
                    "Sub",
                    [blocked, c32],
                    [centered],
                    name=f"{prefix}/CenterQ6Weight",
                ),
                helper.make_node(
                    "Mul",
                    [centered, scale],
                    [dequantized],
                    name=f"{prefix}/DequantizeQ6Weight",
                ),
                helper.make_node(
                    "Reshape",
                    [dequantized, weight_shape],
                    [original_weight],
                    name=f"{prefix}/RestoreQ6WeightShape",
                ),
                node,
            ]
        )
        replaced.add(original_weight)
    missing = sorted(set(replacements) - replaced)
    if missing:
        raise RuntimeError(
            "No MatMul consumed these Q6 projection weights: "
            + ", ".join(missing[:10])
        )
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def replace_q5_projection_nodes(
    model: onnx.ModelProto,
    replacements: dict[str, tuple[str, str, str, str]],
) -> None:
    if not replacements:
        return
    constant_values = (2, 4, 8, 16, 32, 64, 128)
    constants = {
        value: f"/model/q5/{value}"
        for value in constant_values
    }
    model.graph.initializer.extend(
        helper.make_tensor(
            name,
            TensorProto.FLOAT16,
            [],
            np.float16(value).tobytes(),
            raw=True,
        )
        for value, name in constants.items()
    )

    nodes: list[onnx.NodeProto] = []
    replaced: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            nodes.append(node)
            continue
        original_weight = node.input[1]
        replacement = replacements.get(original_weight)
        if replacement is None:
            nodes.append(node)
            continue

        packed, scale, blocked_shape, weight_shape = replacement
        prefix = f"{node.name}/Q5"
        packed_fp16 = f"{prefix}/packed_fp16"
        byte_values = [f"{prefix}/b{index}" for index in range(5)]
        nodes.append(
            helper.make_node(
                "Cast",
                [packed],
                [packed_fp16],
                name=f"{prefix}/CastPacked",
                to=TensorProto.FLOAT16,
            )
        )
        nodes.append(
            helper.make_node(
                "Split",
                [packed_fp16],
                byte_values,
                name=f"{prefix}/SplitBytes",
                axis=1,
            )
        )

        def floor_div(value: str, divisor: int, label: str) -> str:
            divided = f"{prefix}/{label}_div"
            output = f"{prefix}/{label}_floor"
            nodes.extend(
                [
                    helper.make_node(
                        "Div",
                        [value, constants[divisor]],
                        [divided],
                        name=f"{prefix}/Divide{label}",
                    ),
                    helper.make_node(
                        "Floor",
                        [divided],
                        [output],
                        name=f"{prefix}/Floor{label}",
                    ),
                ]
            )
            return output

        def modulo(value: str, divisor: int, label: str) -> str:
            quotient = floor_div(value, divisor, f"{label}Quotient")
            multiple = f"{prefix}/{label}_multiple"
            output = f"{prefix}/{label}_mod"
            nodes.extend(
                [
                    helper.make_node(
                        "Mul",
                        [quotient, constants[divisor]],
                        [multiple],
                        name=f"{prefix}/Scale{label}Quotient",
                    ),
                    helper.make_node(
                        "Sub",
                        [value, multiple],
                        [output],
                        name=f"{prefix}/Modulo{label}",
                    ),
                ]
            )
            return output

        def scaled(value: str, multiplier: int, label: str) -> str:
            output = f"{prefix}/{label}_scaled"
            nodes.append(
                helper.make_node(
                    "Mul",
                    [value, constants[multiplier]],
                    [output],
                    name=f"{prefix}/Scale{label}",
                )
            )
            return output

        def added(left: str, right: str, label: str) -> str:
            output = f"{prefix}/{label}"
            nodes.append(
                helper.make_node(
                    "Add",
                    [left, right],
                    [output],
                    name=f"{prefix}/Combine{label}",
                )
            )
            return output

        b0, b1, b2, b3, b4 = byte_values
        q0 = modulo(b0, 32, "Q0")
        q1 = added(
            floor_div(b0, 32, "Q1Low"),
            scaled(modulo(b1, 4, "Q1HighBits"), 8, "Q1High"),
            "q1",
        )
        q2 = modulo(floor_div(b1, 4, "Q2Shift"), 32, "Q2")
        q3 = added(
            floor_div(b1, 128, "Q3Low"),
            scaled(modulo(b2, 16, "Q3HighBits"), 2, "Q3High"),
            "q3",
        )
        q4 = added(
            floor_div(b2, 16, "Q4Low"),
            scaled(modulo(b3, 2, "Q4HighBits"), 16, "Q4High"),
            "q4",
        )
        q5 = modulo(floor_div(b3, 2, "Q5Shift"), 32, "Q5")
        q6 = added(
            floor_div(b3, 64, "Q6Low"),
            scaled(modulo(b4, 8, "Q6HighBits"), 4, "Q6High"),
            "q6",
        )
        q7 = floor_div(b4, 8, "Q7")
        unpacked = f"{prefix}/unpacked"
        blocked = f"{prefix}/blocked"
        centered = f"{prefix}/centered"
        dequantized = f"{prefix}/dequantized"
        nodes.extend(
            [
                helper.make_node(
                    "Concat",
                    [q0, q1, q2, q3, q4, q5, q6, q7],
                    [unpacked],
                    name=f"{prefix}/ConcatQ5",
                    axis=1,
                ),
                helper.make_node(
                    "Reshape",
                    [unpacked, blocked_shape],
                    [blocked],
                    name=f"{prefix}/BlockQ5Weight",
                ),
                helper.make_node(
                    "Sub",
                    [blocked, constants[16]],
                    [centered],
                    name=f"{prefix}/CenterQ5Weight",
                ),
                helper.make_node(
                    "Mul",
                    [centered, scale],
                    [dequantized],
                    name=f"{prefix}/DequantizeQ5Weight",
                ),
                helper.make_node(
                    "Reshape",
                    [dequantized, weight_shape],
                    [original_weight],
                    name=f"{prefix}/RestoreQ5WeightShape",
                ),
                node,
            ]
        )
        replaced.add(original_weight)
    missing = sorted(set(replacements) - replaced)
    if missing:
        raise RuntimeError(
            "No MatMul consumed these Q5 projection weights: "
            + ", ".join(missing[:10])
        )
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def validate_graph(
    graph_path: Path,
    q8_projection_count: int,
    q4_projection_count: int,
    q6_projection_count: int,
    q5_projection_count: int,
    vocab_size: int,
    q4_embedding_head: bool,
    q8_embedding_head: bool,
) -> None:
    model = onnx.load(graph_path, load_external_data=False)
    q8_initializers = [
        value
        for value in model.graph.initializer
        if value.name.endswith(".MatMul.weight.q8")
    ]
    if len(q8_initializers) != q8_projection_count:
        raise RuntimeError(
            f"Serialized {len(q8_initializers)} INT8 projections; "
            f"expected {q8_projection_count}"
        )
    if any(value.data_type != TensorProto.INT8 for value in q8_initializers):
        raise RuntimeError("Every Q8 projection initializer must be INT8")
    q4_initializers = [
        value
        for value in model.graph.initializer
        if value.name.endswith(".MatMul.weight.q4")
    ]
    if len(q4_initializers) != q4_projection_count:
        raise RuntimeError(
            f"Serialized {len(q4_initializers)} INT4 projections; "
            f"expected {q4_projection_count}"
        )
    q6_initializers = [
        value
        for value in model.graph.initializer
        if value.name.endswith(".MatMul.weight.q6_packed")
    ]
    if len(q6_initializers) != q6_projection_count:
        raise RuntimeError(
            f"Serialized {len(q6_initializers)} Q6 projections; "
            f"expected {q6_projection_count}"
        )
    q5_initializers = [
        value
        for value in model.graph.initializer
        if value.name.endswith(".MatMul.weight.q5_packed")
    ]
    if len(q5_initializers) != q5_projection_count:
        raise RuntimeError(
            f"Serialized {len(q5_initializers)} Q5 projections; "
            f"expected {q5_projection_count}"
        )
    if q4_embedding_head:
        q4_embedding = next(
            value
            for value in model.graph.initializer
            if value.name == "model.embed_tokens.weight_Q4"
        )
        if list(q4_embedding.dims) != [vocab_size, 2560]:
            raise RuntimeError("The Q4 token embedding has the wrong shape")
    elif q8_embedding_head:
        q8_embedding = next(
            value
            for value in model.graph.initializer
            if value.name == "model.embed_tokens.weight.q8"
        )
        if (
            q8_embedding.data_type != TensorProto.INT8
            or list(q8_embedding.dims) != [vocab_size, 2560]
        ):
            raise RuntimeError("The Q8 token embedding has the wrong shape")
    else:
        embedding = next(
            value
            for value in model.graph.initializer
            if value.name == "model.embed_tokens.weight"
        )
        if (
            embedding.data_type != TensorProto.FLOAT16
            or list(embedding.dims) != [vocab_size, 2560]
        ):
            raise RuntimeError("The tied token embedding/head must remain FP16")
    output_shape = model.graph.output[0].type.tensor_type.shape
    if output_shape.dim[-1].dim_value != vocab_size:
        raise RuntimeError("Serialized logits output has the wrong vocabulary size")


def main() -> None:
    args = parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    vocab_size = int(config["vocab_size"])
    hidden_size = int(config["hidden_size"])
    layer_count = int(config["num_hidden_layers"])
    if layer_count != 36 or hidden_size != 2560:
        raise RuntimeError(
            "The selected template is only valid for the 36-layer, "
            "2560-hidden ACE planner"
        )
    if args.q4_embedding_head and args.q8_embedding_head:
        raise ValueError(
            "--q4-embedding-head and --q8-embedding-head are mutually exclusive"
        )
    allowed_q4_projections = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    q4_projection_names = {
        value.strip()
        for value in args.q4_projections.split(",")
        if value.strip()
    }
    unknown_q4_projections = q4_projection_names - allowed_q4_projections
    if unknown_q4_projections:
        raise ValueError(
            "Unknown --q4-projections values: "
            + ", ".join(sorted(unknown_q4_projections))
        )
    if args.q6_body and args.q5_body:
        raise ValueError("--q6-body and --q5-body are mutually exclusive")
    if (args.q6_body or args.q5_body) and q4_projection_names:
        raise ValueError(
            "--q6-body/--q5-body cannot be combined with --q4-projections"
        )

    output = args.output.resolve()
    onnx_dir = output / "onnx"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(args.template, load_external_data=False)
    checkpoint = FP16_BUILDER.Checkpoint(args.checkpoint)
    writer = FP16_BUILDER.ExternalDataWriter(
        onnx_dir,
        "model_quantized.onnx_data",
        args.chunk_mib * 1024 * 1024,
    )
    initializers = {value.name: value for value in model.graph.initializer}

    if args.q4_embedding_head:
        INT4_BUILDER.replace_embedding_and_head(
            model,
            checkpoint,
            writer,
            vocab_size,
            hidden_size,
        )
    elif args.q8_embedding_head:
        replace_embedding_with_q8(
            model,
            checkpoint,
            writer,
            vocab_size,
            hidden_size,
        )
    else:
        FP16_BUILDER.replace_embedding(
            initializers["model.embed_tokens.weight"],
            checkpoint,
            writer,
            vocab_size,
            hidden_size,
        )
    FP16_BUILDER.replace_rotary_cache(
        initializers,
        writer,
        int(config["max_position_embeddings"]),
        int(config["head_dim"]),
        float(config["rope_theta"]),
    )

    replacements: dict[str, tuple[str, str, str, str]] = {}
    q4_replacements: dict[str, tuple[str, str, int, int]] = {}
    q6_replacements: dict[str, tuple[str, str, str, str]] = {}
    q5_replacements: dict[str, tuple[str, str, str, str]] = {}
    dequant_initializers: list[onnx.TensorProto] = []
    for initializer in list(model.graph.initializer):
        source_name = FP16_BUILDER.source_name_for_norm(initializer.name)
        if source_name:
            value = (
                checkpoint.tensor(source_name)
                .to(torch.float16)
                .contiguous()
                .numpy()
            )
            FP16_BUILDER.set_raw_tensor(initializer, value)
            continue

        source_name = FP16_BUILDER.source_name_for_projection(initializer.name)
        if source_name is None:
            continue
        original_name = initializer.name
        print(f"quantizing {source_name}", flush=True)
        projection_match = FP16_BUILDER.PROJECTION_NAME.match(original_name)
        if projection_match is None:
            raise RuntimeError(f"Could not parse projection {original_name}")
        projection_name = projection_match["projection"]
        checkpoint_weight = checkpoint.tensor(source_name)
        if args.q6_body or args.q5_body:
            input_width, output_width = initializer.dims
            packing_bits = 6 if args.q6_body else 5
            quantizer = (
                quantize_projection_q6
                if args.q6_body
                else quantize_projection_q5
            )
            packed, scale = quantizer(checkpoint_weight)
            quantized_name = (
                f"{original_name}.q{packing_bits}_packed"
            )
            scale_name = f"{original_name}.q{packing_bits}_scale"
            block_shape_name = (
                f"{original_name}.q{packing_bits}_block_shape"
            )
            weight_shape_name = (
                f"{original_name}.q{packing_bits}_weight_shape"
            )
            initializer.name = quantized_name
            writer.write_array(initializer, packed)
            dequant_initializers.extend(
                [
                    helper.make_tensor(
                        scale_name,
                        TensorProto.FLOAT16,
                        list(scale.shape),
                        scale.tobytes(),
                        raw=True,
                    ),
                    helper.make_tensor(
                        block_shape_name,
                        TensorProto.INT64,
                        [3],
                        [
                            int(input_width) // BLOCK_SIZE,
                            BLOCK_SIZE,
                            int(output_width),
                        ],
                    ),
                    helper.make_tensor(
                        weight_shape_name,
                        TensorProto.INT64,
                        [2],
                        [int(input_width), int(output_width)],
                    ),
                ]
            )
            target_replacements = (
                q6_replacements if args.q6_body else q5_replacements
            )
            target_replacements[original_name] = (
                quantized_name,
                scale_name,
                block_shape_name,
                weight_shape_name,
            )
            del packed, scale, checkpoint_weight
            gc.collect()
            continue
        if projection_name in q4_projection_names:
            input_width, output_width = initializer.dims
            packed, scale, _ = INT4_BUILDER.quantize_matmul_weight(
                checkpoint_weight
            )
            quantized_name = f"{original_name}.q4"
            scale_name = f"{original_name}.q4_scale"
            initializer.name = quantized_name
            writer.write_array(initializer, packed)
            dequant_initializers.append(
                helper.make_tensor(
                    scale_name,
                    TensorProto.FLOAT16,
                    list(scale.shape),
                    scale.tobytes(),
                    raw=True,
                )
            )
            q4_replacements[original_name] = (
                quantized_name,
                scale_name,
                int(input_width),
                int(output_width),
            )
            del packed, scale, checkpoint_weight
            gc.collect()
            continue

        quantized, scale = quantize_projection(checkpoint_weight)
        quantized_name = f"{original_name}.q8"
        scale_name = f"{original_name}.q8_scale"
        block_shape_name = f"{original_name}.q8_block_shape"
        weight_shape_name = f"{original_name}.q8_weight_shape"
        initializer.name = quantized_name
        writer.write_array(initializer, quantized)
        dequant_initializers.extend(
            [
            helper.make_tensor(
                scale_name,
                TensorProto.FLOAT16,
                list(scale.shape),
                scale.tobytes(),
                raw=True,
            ),
            helper.make_tensor(
                block_shape_name,
                TensorProto.INT64,
                [3],
                [
                    quantized.shape[0] // BLOCK_SIZE,
                    BLOCK_SIZE,
                    quantized.shape[1],
                ],
            ),
            helper.make_tensor(
                weight_shape_name,
                TensorProto.INT64,
                [2],
                list(quantized.shape),
            ),
            ]
        )
        replacements[original_name] = (
            quantized_name,
            scale_name,
            block_shape_name,
            weight_shape_name,
        )
        del quantized, scale, checkpoint_weight
        gc.collect()

    model.graph.initializer.extend(dequant_initializers)
    add_dequantization_nodes(model, replacements)
    replace_q4_projection_nodes(model, q4_replacements)
    replace_q6_projection_nodes(model, q6_replacements)
    replace_q5_projection_nodes(model, q5_replacements)
    checkpoint.assert_fully_consumed()
    writer.close()

    logits_shape = model.graph.output[0].type.tensor_type.shape
    logits_shape.dim[-1].ClearField("dim_param")
    logits_shape.dim[-1].dim_value = vocab_size
    embedding_precision = (
        "Q4 tied embedding/head"
        if args.q4_embedding_head
        else (
            "blockwise Q8 tied embedding/head"
            if args.q8_embedding_head
            else "FP16 tied embedding/head"
        )
    )
    body_precision = (
        "packed blockwise Q6"
        if args.q6_body
        else (
            "packed blockwise Q5"
            if args.q5_body
            else "mixed blockwise INT8/INT4"
        )
    )
    model.doc_string = (
        "ACE-Step 1.5 5 Hz LM 4B planner. "
        f"{body_precision} transformer projections with {embedding_precision} "
        "and residual-range scaling for Transformers.js/WebGPU."
    )
    graph_path = onnx_dir / "model_quantized.onnx"
    onnx.save_model(model, graph_path)
    FP16_BUILDER.copy_support_files(
        args.tokenizer_source,
        output,
        len(writer.files),
    )
    browser_config = json.loads((output / "config.json").read_text())
    browser_config["transformers.js_config"] = {
        "dtype": "q8",
        "kv_cache_dtype": {"q8": "float16", "fp16": "float16"},
        "use_external_data_format": {
            "model_quantized.onnx": len(writer.files)
        },
    }
    (output / "config.json").write_text(
        json.dumps(browser_config, indent=2, sort_keys=True) + "\n"
    )
    validate_graph(
        graph_path,
        len(replacements),
        len(q4_replacements),
        len(q6_replacements),
        len(q5_replacements),
        vocab_size,
        args.q4_embedding_head,
        args.q8_embedding_head,
    )

    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "template_graph": str(args.template),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "layers": layer_count,
        "precision": {
            "transformer_projections": (
                "packed symmetric blockwise int6"
                if args.q6_body
                else (
                    "packed symmetric blockwise int5"
                    if args.q5_body
                    else "symmetric blockwise int8"
                )
            ),
            "q4_projection_names": sorted(q4_projection_names),
            "body_packing_bits": (
                6 if args.q6_body else (5 if args.q5_body else None)
            ),
            "projection_block_size": BLOCK_SIZE,
            "projection_dequantization": (
                "ONNX bit unpack to UINT8 + Cast(FP16) + Mul(scale) + Reshape"
                if args.q6_body or args.q5_body
                else "Cast(INT8 to FP16) + Reshape + Mul(scale) + Reshape"
            ),
            "embedding_and_tied_head": (
                "blockwise int4"
                if args.q4_embedding_head
                else (
                    "blockwise int8"
                    if args.q8_embedding_head
                    else "float16"
                )
            ),
            "residual_stream_scale": 1.0 / 256.0,
        },
        "q8_projection_count": len(replacements),
        "q4_projection_count": len(q4_replacements),
        "q6_projection_count": len(q6_replacements),
        "q5_projection_count": len(q5_replacements),
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
