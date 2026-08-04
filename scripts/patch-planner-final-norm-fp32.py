#!/usr/bin/env python3
"""Compute the ACE planner's final MLP projection and RMS norm in FP32.

Chromium/Metal WebGPU produces the planner's first non-finite semantic value
inside the last MLP down-projection at longer contexts and inside the final
fused SkipSimplifiedLayerNormalization at shorter contexts. This patch keeps
the final down-projection output, residual add, and RMS normalization in FP32,
then casts the bounded normalized result back to FP16 for the language-model
head. Both the INT4 MatMulNBits export and the FP16 MatMul parity export are
supported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

FINAL_NORM_NAME = "/model/layers.36/final_norm_layernorm/SkipLayerNorm"
FINAL_DOWN_PROJECTION_NAMES = (
    "/model/layers.35/mlp/down_proj/MatMul_Q4",
    "/model/layers.35/mlp/down_proj/MatMul",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.source, load_external_data=False)
    nodes = list(model.graph.node)

    down_projection_index = next(
        index
        for index, node in enumerate(nodes)
        if node.name in FINAL_DOWN_PROJECTION_NAMES
    )
    down_projection = nodes[down_projection_index]
    if down_projection.op_type not in {"MatMulNBits", "MatMul"}:
        raise RuntimeError(
            f"{down_projection.name} is {down_projection.op_type}, "
            "expected MatMulNBits or MatMul"
        )
    expected_inputs = 3 if down_projection.op_type == "MatMulNBits" else 2
    if (
        len(down_projection.input) != expected_inputs
        or len(down_projection.output) != 1
    ):
        raise RuntimeError(
            "Unexpected final down-projection signature: "
            f"{list(down_projection.input)} -> "
            f"{list(down_projection.output)}"
        )
    down_attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in down_projection.attribute
    }
    down_prefix = "/model/layers.35/mlp/down_proj/fp32"
    activation = down_projection.input[0]
    down_output = down_projection.output[0]
    down_value_info = next(
        value_info
        for value_info in model.graph.value_info
        if value_info.name == down_output
    )
    down_value_info.type.tensor_type.elem_type = TensorProto.FLOAT
    replacement_down_projection = [
        helper.make_node(
            "Cast",
            [activation],
            [f"{down_prefix}/activation"],
            name=f"{down_prefix}/CastActivation",
            to=TensorProto.FLOAT,
        ),
    ]
    if down_projection.op_type == "MatMulNBits":
        _, weight_q4, scales = down_projection.input
        replacement_down_projection.extend(
            [
                helper.make_node(
                    "Cast",
                    [scales],
                    [f"{down_prefix}/scales"],
                    name=f"{down_prefix}/CastScales",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "MatMulNBits",
                    [
                        f"{down_prefix}/activation",
                        weight_q4,
                        f"{down_prefix}/scales",
                    ],
                    [down_output],
                    name=f"{down_prefix}/MatMul_Q4",
                    domain="com.microsoft",
                    **down_attributes,
                ),
            ]
        )
    else:
        weight = down_projection.input[1]
        replacement_down_projection.extend(
            [
                helper.make_node(
                    "Cast",
                    [weight],
                    [f"{down_prefix}/weight"],
                    name=f"{down_prefix}/CastWeight",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "MatMul",
                    [f"{down_prefix}/activation", f"{down_prefix}/weight"],
                    [down_output],
                    name=f"{down_prefix}/MatMul",
                ),
            ]
        )
    nodes[
        down_projection_index : down_projection_index + 1
    ] = replacement_down_projection

    node_index = next(
        index for index, node in enumerate(nodes) if node.name == FINAL_NORM_NAME
    )
    original = nodes[node_index]
    if original.op_type != "SkipSimplifiedLayerNormalization":
        raise RuntimeError(
            f"{FINAL_NORM_NAME} is {original.op_type}, expected "
            "SkipSimplifiedLayerNormalization"
        )
    if len(original.input) != 3 or len(original.output) != 1:
        raise RuntimeError(
            f"Unexpected final norm signature: {list(original.input)} -> "
            f"{list(original.output)}"
        )

    epsilon = next(
        (
            helper.get_attribute_value(attribute)
            for attribute in original.attribute
            if attribute.name == "epsilon"
        ),
        1e-6,
    )
    prefix = "/model/layers.36/final_norm_layernorm/fp32"
    epsilon_name = f"{prefix}/epsilon"
    model.graph.initializer.append(
        helper.make_tensor(
            epsilon_name,
            TensorProto.FLOAT,
            [],
            [float(epsilon)],
        )
    )

    residual, update, weight = original.input
    output = original.output[0]
    replacement = [
        helper.make_node(
            "Cast",
            [residual],
            [f"{prefix}/residual"],
            name=f"{prefix}/CastResidual",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Cast",
            [update],
            [f"{prefix}/update"],
            name=f"{prefix}/CastUpdate",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Add",
            [f"{prefix}/residual", f"{prefix}/update"],
            [f"{prefix}/sum"],
            name=f"{prefix}/AddResidual",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}/sum", f"{prefix}/sum"],
            [f"{prefix}/square"],
            name=f"{prefix}/Square",
        ),
        helper.make_node(
            "ReduceMean",
            [f"{prefix}/square"],
            [f"{prefix}/mean_square"],
            name=f"{prefix}/MeanSquare",
            axes=[-1],
            keepdims=1,
        ),
        helper.make_node(
            "Add",
            [f"{prefix}/mean_square", epsilon_name],
            [f"{prefix}/variance_epsilon"],
            name=f"{prefix}/AddEpsilon",
        ),
        helper.make_node(
            "Sqrt",
            [f"{prefix}/variance_epsilon"],
            [f"{prefix}/rms"],
            name=f"{prefix}/SqrtRms",
        ),
        helper.make_node(
            "Div",
            [f"{prefix}/sum", f"{prefix}/rms"],
            [f"{prefix}/normalized"],
            name=f"{prefix}/Normalize",
        ),
        helper.make_node(
            "Cast",
            [weight],
            [f"{prefix}/weight"],
            name=f"{prefix}/CastWeight",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}/normalized", f"{prefix}/weight"],
            [f"{prefix}/weighted"],
            name=f"{prefix}/ApplyWeight",
        ),
        helper.make_node(
            "Cast",
            [f"{prefix}/weighted"],
            [output],
            name=f"{prefix}/CastOutput",
            to=TensorProto.FLOAT16,
        ),
    ]
    nodes[node_index : node_index + 1] = replacement
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.doc_string = (
        f"{model.doc_string}\n\n"
        "ai-music-js WebGPU patch: final MLP down-projection, residual add, "
        "and RMS normalization run in FP32 before returning a bounded FP16 "
        "activation."
    ).strip()

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.destination)
    print(
        f"Wrote {args.destination} ({args.destination.stat().st_size:,} "
        "bytes) with FP32 final projection and residual normalization."
    )


if __name__ == "__main__":
    main()
