#!/usr/bin/env python3
"""Export ACE-Step XL's 5 Hz audio-code detokenizer for WebGPU.

The graph maps integer FSQ code indices [batch, codes, 1] to the
64-channel, 25 Hz semantic hint tensor consumed by the condition encoder.
Only the quantizer output projection and detokenizer weights are loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from safetensors import safe_open


def load_prefixed_state(checkpoint: Path, prefix: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = source.get_tensor(key)
    if not state:
        raise RuntimeError(f"No {prefix!r} tensors found in {checkpoint}")
    return state


def configure_encoder(config):
    config = copy.deepcopy(config)
    config.hidden_size = config.encoder_hidden_size
    config.intermediate_size = config.encoder_intermediate_size
    config.num_attention_heads = config.encoder_num_attention_heads
    config.num_key_value_heads = config.encoder_num_key_value_heads
    config._attn_implementation = "eager"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ace-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.ace_source.resolve()))
    from acestep.models.common.configuration_acestep_v15 import AceStepConfig
    from acestep.models.xl_turbo.modeling_acestep_v15_xl_turbo import (
        AudioTokenDetokenizer,
        ResidualFSQ,
    )

    config = configure_encoder(
        AceStepConfig.from_pretrained(str(args.checkpoint))
    )
    quantizer = ResidualFSQ(
        dim=config.fsq_dim,
        levels=config.fsq_input_levels,
        num_quantizers=config.fsq_input_num_quantizers,
    )
    quantizer_state = load_prefixed_state(
        args.checkpoint, "tokenizer.quantizer."
    )
    missing, unexpected = quantizer.load_state_dict(
        quantizer_state, strict=False
    )
    required_missing = [
        key for key in missing if not key.startswith("project_in.")
    ]
    if required_missing or unexpected:
        raise RuntimeError(
            f"Quantizer state mismatch: missing={required_missing}, "
            f"unexpected={unexpected}"
        )

    detokenizer = AudioTokenDetokenizer(config)
    missing, unexpected = detokenizer.load_state_dict(
        load_prefixed_state(args.checkpoint, "detokenizer."), strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Detokenizer state mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    class AudioCodeDetokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # ResidualFSQ's convenience method uses einx/vmap, which the legacy
            # ONNX tracer cannot represent. With one quantizer it is exactly a
            # codebook gather followed by project_out.
            self.register_buffer("codebook", quantizer.codebooks[0])
            self.project_out = quantizer.project_out
            self.detokenizer = detokenizer

        def forward(self, code_indices: torch.Tensor) -> torch.Tensor:
            codes = torch.nn.functional.embedding(
                code_indices.squeeze(-1), self.codebook
            )
            quantized = self.project_out(codes)
            return self.detokenizer(quantized)

    model = AudioCodeDetokenizer().eval().half()
    code_indices = torch.tensor(
        [[[0], [1], [31_337], [63_999]]], dtype=torch.int64
    )
    with torch.inference_mode():
        expected = model(code_indices).float().cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (code_indices,),
        str(args.output),
        input_names=["code_indices"],
        output_names=["semantic_hints_25hz"],
        dynamic_axes={
            "code_indices": {0: "batch", 1: "audio_codes_5hz"},
            "semantic_hints_25hz": {
                0: "batch",
                1: "semantic_frames_25hz",
            },
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    graph = onnx.load(str(args.output), load_external_data=True)
    data_path = args.output.with_name(f"{args.output.name}.data")
    onnx.save_model(
        graph,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=0,
        convert_attribute=False,
    )

    fixture = args.fixture or args.output.with_suffix(".fixture.npz")
    np.savez_compressed(
        fixture,
        code_indices=code_indices.numpy(),
        semantic_hints_25hz=expected,
    )
    metadata = {
        "input": {"name": "code_indices", "dtype": "int64", "shape": ["B", "T", 1]},
        "output": {
            "name": "semantic_hints_25hz",
            "dtype": "float16",
            "shape": ["B", "T*5", 64],
        },
        "levels": config.fsq_input_levels,
        "checkpoint": args.checkpoint.name,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    print(data_path)
    print(fixture)


if __name__ == "__main__":
    main()
