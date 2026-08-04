#!/usr/bin/env python3
"""Keep an FP16-activation Qwen/ACE residual stream numerically bounded.

The ACE 4B planner checkpoint is trained/stored as BF16. A direct FP16 ONNX
conversion can overflow in an MLP output projection because FP16 has a much
smaller exponent range. This graph rewrite carries the residual stream at a
fixed smaller scale while leaving every normalized activation mathematically
unchanged:

* scale token embeddings before the first residual block;
* scale each attention and MLP output-projection input by the same factor;
* scale residual RMSNorm epsilon by factor squared.

For a linear projection ``W(x * s) == W(x) * s``. Every residual addition
therefore remains consistently scaled, and RMSNorm removes that common scale.
The rewrite supports both unquantized ``MatMul``/``Gather`` graphs and Q4
``MatMulNBits``/``GatherBlockQuantized`` graphs. No learned weights are changed
and no copy of external-data shards is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


EMBEDDING_OUTPUT = "/model/embed_tokens/Gather/output_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--scale", type=float, default=1.0 / 256.0)
    return parser.parse_args()


def is_residual_norm(node: onnx.NodeProto) -> bool:
    if node.op_type not in {
        "SimplifiedLayerNormalization",
        "SkipSimplifiedLayerNormalization",
    }:
        return False
    return (
        "/input_layernorm/" in node.name
        or "/post_attention_layernorm/" in node.name
        or "/final_norm_layernorm/" in node.name
    )


def is_residual_projection(node: onnx.NodeProto) -> bool:
    return (
        node.op_type in {"MatMul", "MatMulNBits"}
        and (
            "/attn/o_proj/MatMul" in node.name
            or "/mlp/down_proj/MatMul" in node.name
        )
    )


def main() -> None:
    args = parse_args()
    if not (0 < args.scale < 1):
        raise ValueError("--scale must be between zero and one")

    model = onnx.load(args.source, load_external_data=False)
    scale_name = "/model/residual_scale"
    model.graph.initializer.append(
        helper.make_tensor(
            scale_name,
            TensorProto.FLOAT16,
            [],
            np.asarray(args.scale, dtype=np.float16).tobytes(),
            raw=True,
        )
    )

    nodes: list[onnx.NodeProto] = []
    embedding_scaled = False
    projection_count = 0
    norm_count = 0
    for node in model.graph.node:
        if EMBEDDING_OUTPUT in node.output:
            if node.op_type not in {"Gather", "GatherBlockQuantized"}:
                raise RuntimeError(
                    "Embedding producer is "
                    f"{node.op_type}, expected Gather or GatherBlockQuantized"
                )
            original_output = node.output[0]
            unscaled_output = f"{original_output}/unscaled"
            node.output[0] = unscaled_output
            nodes.extend(
                [
                    node,
                    helper.make_node(
                        "Mul",
                        [unscaled_output, scale_name],
                        [original_output],
                        name="/model/embed_tokens/ScaleResidual",
                    ),
                ]
            )
            embedding_scaled = True
            continue

        if is_residual_projection(node):
            activation = node.input[0]
            scaled_activation = f"{activation}/residual_scaled"
            node.input[0] = scaled_activation
            nodes.extend(
                [
                    helper.make_node(
                        "Mul",
                        [activation, scale_name],
                        [scaled_activation],
                        name=f"{node.name}/ScaleResidualInput",
                    ),
                    node,
                ]
            )
            projection_count += 1
            continue

        if is_residual_norm(node):
            saw_epsilon = False
            for attribute in node.attribute:
                if attribute.name == "epsilon":
                    attribute.f = float(attribute.f) * args.scale * args.scale
                    saw_epsilon = True
            if not saw_epsilon:
                node.attribute.append(
                    helper.make_attribute(
                        "epsilon",
                        1e-6 * args.scale * args.scale,
                    )
                )
            norm_count += 1
        nodes.append(node)

    if not embedding_scaled:
        raise RuntimeError("Did not find the token-embedding Gather output")
    if projection_count != 72:
        raise RuntimeError(
            f"Scaled {projection_count} residual projections; expected 72"
        )
    if norm_count != 73:
        raise RuntimeError(f"Scaled epsilon on {norm_count} norms; expected 73")

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.doc_string = (
        f"{model.doc_string}\n\n"
        f"ai-music-js FP16 residual-range patch: residual scale {args.scale}; "
        "attention/MLP output projections and residual RMSNorm epsilon scaled "
        "consistently."
    ).strip()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.destination)
    print(
        f"Wrote {args.destination} with residual scale {args.scale}, "
        f"{projection_count} projections, and {norm_count} norms."
    )


if __name__ == "__main__":
    main()
