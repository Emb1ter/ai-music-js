#!/usr/bin/env python3
"""Expose the ACE planner's final normalized hidden state for parity checks.

The diagnostic output lets us distinguish transformer-body quantization error
from output-head quantization error without rebuilding or loading a second
multi-gigabyte ONNX model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


FINAL_HIDDEN = "/model/layers.36/final_norm_layernorm/output_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--replace-logits",
        action="store_true",
        help=(
            "Replace the logits output with the final hidden state and prune "
            "the tied language-model head. KV-cache outputs are retained."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.source, load_external_data=False)
    if not any(FINAL_HIDDEN in node.output for node in model.graph.node):
        raise RuntimeError(f"Graph does not produce {FINAL_HIDDEN}")
    if any(output.name == FINAL_HIDDEN for output in model.graph.output):
        raise RuntimeError(f"Graph already exposes {FINAL_HIDDEN}")

    final_hidden_output = helper.make_tensor_value_info(
        FINAL_HIDDEN,
        TensorProto.FLOAT16,
        ["batch_size", "sequence_length", 2560],
    )
    if args.replace_logits:
        retained_outputs = [
            output for output in model.graph.output if output.name != "logits"
        ]
        del model.graph.output[:]
        model.graph.output.extend([final_hidden_output, *retained_outputs])
        model = onnx.utils.Extractor(model).extract_model(
            [value.name for value in model.graph.input],
            [value.name for value in model.graph.output],
        )
    else:
        model.graph.output.append(final_hidden_output)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.destination)
    print(f"Wrote {args.destination}")


if __name__ == "__main__":
    main()
