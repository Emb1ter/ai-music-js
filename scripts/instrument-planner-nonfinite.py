#!/usr/bin/env python3
"""Add compact non-finite probes to the ACE planner ONNX graph.

Each probe turns NaN or infinity values into a scalar count:

    bad = not(equal(value * 0, value * 0))

Finite values multiplied by zero remain zero. NaN remains NaN, and infinity
multiplied by zero becomes NaN. The scalar outputs make it possible to locate
the first failing WebGPU node without transferring every activation back to
JavaScript.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import onnx
from onnx import TensorProto, helper


PROBED_OPERATORS = {
    "GatherBlockQuantized",
    "GroupQueryAttention",
    "MatMulNBits",
    "Mul",
    "RotaryEmbedding",
    "Sigmoid",
    "SimplifiedLayerNormalization",
    "SkipSimplifiedLayerNormalization",
}


def diagnostic_name(
    node_index: int,
    operator: str,
    node_name: str,
    output_name: str,
    output_index: int,
) -> str:
    readable = f"{operator}.{node_name or 'unnamed'}.{output_name}.out{output_index}"
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", readable).strip("_")
    return f"planner_diag.{node_index:04d}.{readable}"


def instrument(model: onnx.ModelProto) -> int:
    zero_name = "planner_diag.zero_fp16"
    model.graph.initializer.append(
        helper.make_tensor(zero_name, TensorProto.FLOAT16, [], [0.0])
    )

    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    probe_count = 0

    for node_index, node in enumerate(original_nodes):
        model.graph.node.append(node)
        if node.op_type not in PROBED_OPERATORS:
            continue
        if node.name.startswith(
            "/model/layers.36/final_norm_layernorm/fp32/"
        ) or node.name.startswith(
            "/model/layers.35/mlp/down_proj/fp32/"
        ):
            # These deliberately run in FLOAT. The compact probes use an FP16
            # zero and resume at the FP16 lm_head immediately downstream.
            continue

        for output_index, output_name in enumerate(node.output):
            if not output_name:
                continue
            diagnostic = diagnostic_name(
                node_index,
                node.op_type,
                node.name,
                output_name,
                output_index,
            )
            zeroed = f"{diagnostic}.zeroed"
            self_equal = f"{diagnostic}.self_equal"
            bad_mask = f"{diagnostic}.bad_mask"
            bad_int = f"{diagnostic}.bad_int"

            model.graph.node.extend(
                [
                    helper.make_node(
                        "Mul",
                        [output_name, zero_name],
                        [zeroed],
                        name=f"{diagnostic}.MulZero",
                    ),
                    helper.make_node(
                        "Equal",
                        [zeroed, zeroed],
                        [self_equal],
                        name=f"{diagnostic}.EqualSelf",
                    ),
                    helper.make_node(
                        "Not",
                        [self_equal],
                        [bad_mask],
                        name=f"{diagnostic}.NotFinite",
                    ),
                    helper.make_node(
                        "Cast",
                        [bad_mask],
                        [bad_int],
                        name=f"{diagnostic}.CastCount",
                        to=TensorProto.INT32,
                    ),
                    helper.make_node(
                        "ReduceSum",
                        [bad_int],
                        [diagnostic],
                        name=f"{diagnostic}.ReduceCount",
                        keepdims=0,
                    ),
                ]
            )
            model.graph.output.append(
                helper.make_tensor_value_info(
                    diagnostic,
                    TensorProto.INT32,
                    [],
                )
            )
            probe_count += 1

    model.doc_string = (
        f"{model.doc_string}\n\n"
        f"ai-music-js WebGPU non-finite diagnostic graph: {probe_count} "
        "scalar probes."
    ).strip()
    return probe_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.source, load_external_data=False)
    probe_count = instrument(model)
    # The graph intentionally keeps the source model's external-data
    # references. The ONNX checker requires all 2.65 GB of those files to sit
    # beside the graph, so structural validation happens in the browser session
    # that already owns those cached files.
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.destination)
    print(
        f"Wrote {args.destination} with {probe_count} non-finite probes "
        f"({args.destination.stat().st_size:,} bytes)."
    )


if __name__ == "__main__":
    main()
