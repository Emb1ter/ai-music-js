#!/usr/bin/env python3
"""Build a local Q4F16 WebGPU export of the pinned HauhauCS Qwen3.5 2B model.

The source checkpoint is BF16 Safetensors and cannot be loaded by
Transformers.js. This script retains the already-qualified Qwen3.5 2B ONNX
graph, downloads the 55 intentionally modified source tensors, requantizes
the changed matrices to MatMulNBits Q4F16, and patches the 13 changed Mamba
A_log constants. Vision assets are intentionally omitted.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import quantize_matmul_4bits


BASE_MODEL = "onnx-community/Qwen3.5-2B-ONNX-OPT"
BASE_REVISION = "2ea7886f48b926aca97de8b0e041ffca7e3ebaa9"
SOURCE_MODEL = (
    "DreamFast/"
    "Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark"
)
SOURCE_REVISION = "9af5b83f777a0c8d3e36d65a1446aac3a463d0d6"
SOURCE_WEIGHTS = "model.safetensors"

BASE_ASSETS = {
    "config.json": 2_993,
    "generation_config.json": 248,
    "chat_template.jinja": 7_755,
    "tokenizer.json": 19_226_111,
    "tokenizer_config.json": 9_161,
    "onnx/decoder_model_merged_q4f16.onnx": 707_377,
    "onnx/decoder_model_merged_q4f16.onnx_data": 1_088_892_928,
    "onnx/embed_tokens_q4f16.onnx": 1_064,
    "onnx/embed_tokens_q4f16.onnx_data": 294_010_880,
}

# Confirmed against the pinned official Qwen checkpoint. The list deliberately
# excludes the 18 linear-attention norm rounding artefacts documented by the
# source model card.
CHANGED_TENSORS = (
    "model.language_model.layers.0.linear_attn.A_log",
    "model.language_model.layers.1.linear_attn.A_log",
    "model.language_model.layers.2.linear_attn.A_log",
    "model.language_model.layers.4.linear_attn.A_log",
    "model.language_model.layers.5.linear_attn.A_log",
    "model.language_model.layers.6.linear_attn.A_log",
    "model.language_model.layers.8.linear_attn.A_log",
    "model.language_model.layers.8.linear_attn.out_proj.weight",
    "model.language_model.layers.9.linear_attn.A_log",
    "model.language_model.layers.9.linear_attn.out_proj.weight",
    "model.language_model.layers.10.linear_attn.A_log",
    "model.language_model.layers.10.linear_attn.out_proj.weight",
    "model.language_model.layers.10.mlp.up_proj.weight",
    "model.language_model.layers.11.mlp.up_proj.weight",
    "model.language_model.layers.11.self_attn.o_proj.weight",
    "model.language_model.layers.12.linear_attn.A_log",
    "model.language_model.layers.12.linear_attn.out_proj.weight",
    "model.language_model.layers.12.mlp.down_proj.weight",
    "model.language_model.layers.12.mlp.up_proj.weight",
    "model.language_model.layers.13.linear_attn.A_log",
    "model.language_model.layers.13.linear_attn.out_proj.weight",
    "model.language_model.layers.13.mlp.down_proj.weight",
    "model.language_model.layers.13.mlp.gate_proj.weight",
    "model.language_model.layers.13.mlp.up_proj.weight",
    "model.language_model.layers.14.linear_attn.out_proj.weight",
    "model.language_model.layers.14.mlp.down_proj.weight",
    "model.language_model.layers.14.mlp.gate_proj.weight",
    "model.language_model.layers.14.mlp.up_proj.weight",
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.15.mlp.gate_proj.weight",
    "model.language_model.layers.15.mlp.up_proj.weight",
    "model.language_model.layers.15.self_attn.o_proj.weight",
    "model.language_model.layers.16.linear_attn.A_log",
    "model.language_model.layers.16.linear_attn.out_proj.weight",
    "model.language_model.layers.16.mlp.down_proj.weight",
    "model.language_model.layers.16.mlp.gate_proj.weight",
    "model.language_model.layers.16.mlp.up_proj.weight",
    "model.language_model.layers.17.linear_attn.out_proj.weight",
    "model.language_model.layers.17.mlp.down_proj.weight",
    "model.language_model.layers.17.mlp.gate_proj.weight",
    "model.language_model.layers.17.mlp.up_proj.weight",
    "model.language_model.layers.18.linear_attn.out_proj.weight",
    "model.language_model.layers.18.mlp.down_proj.weight",
    "model.language_model.layers.18.mlp.up_proj.weight",
    "model.language_model.layers.19.mlp.down_proj.weight",
    "model.language_model.layers.19.mlp.up_proj.weight",
    "model.language_model.layers.20.linear_attn.out_proj.weight",
    "model.language_model.layers.20.mlp.down_proj.weight",
    "model.language_model.layers.20.mlp.up_proj.weight",
    "model.language_model.layers.21.linear_attn.out_proj.weight",
    "model.language_model.layers.21.mlp.down_proj.weight",
    "model.language_model.layers.22.linear_attn.A_log",
    "model.language_model.layers.22.linear_attn.out_proj.weight",
    "model.language_model.layers.22.mlp.down_proj.weight",
    "model.language_model.layers.23.mlp.down_proj.weight",
)


def hub_url(model: str, revision: str, file_name: str) -> str:
    return f"https://huggingface.co/{model}/resolve/{revision}/{file_name}"


def request_bytes(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end - 1}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        value = response.read()
    if start is not None and end is not None and len(value) != end - start:
        raise RuntimeError(
            f"Range response length mismatch for {url}: "
            f"expected {end - start}, received {len(value)}"
        )
    return value


def download(url: str, output: Path, expected_bytes: int) -> None:
    if output.is_file() and output.stat().st_size == expected_bytes:
        print(f"  cached {output.name}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"  downloading {output.name} ({expected_bytes / 1e6:.2f} MB)")
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as target:
        copied = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            copied += len(chunk)
            print(
                f"\r    {copied / expected_bytes:6.1%} · {copied / 1e6:,.0f} MB",
                end="",
                flush=True,
            )
    print()
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Download size mismatch for {output}: expected {expected_bytes}, "
            f"received {partial.stat().st_size}"
        )
    partial.replace(output)


def prepare_base_assets(output: Path, base_dir: Path | None) -> None:
    print("Preparing the qualified Qwen3.5 2B Q4F16 text-only graph.")
    for relative, expected_bytes in BASE_ASSETS.items():
        target = output / relative
        source = base_dir / relative if base_dir else None
        if source and source.is_file() and source.stat().st_size == expected_bytes:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Always restore cached base assets before patching. This makes an
            # interrupted conversion safely restartable instead of applying
            # quantization on top of a partially patched output.
            print(f"  copying {relative} from the base cache")
            shutil.copyfile(source, target)
            continue
        download(hub_url(BASE_MODEL, BASE_REVISION, relative), target, expected_bytes)


def safetensors_header(url: str) -> tuple[dict[str, dict], int]:
    header_bytes = struct.unpack("<Q", request_bytes(url, 0, 8))[0]
    header = json.loads(request_bytes(url, 8, 8 + header_bytes))
    return header, 8 + header_bytes


def bf16_tensor(raw: bytes, shape: list[int]) -> np.ndarray:
    uint32 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return uint32.view(np.float32).reshape(shape)


def external_range(tensor: onnx.TensorProto) -> tuple[int, int]:
    fields = {item.key: item.value for item in tensor.external_data}
    return int(fields.get("offset", "0")), int(fields.get("length", "0"))


def onnx_prefix(source_name: str) -> str:
    parts = source_name.split(".")
    layer = parts[3]
    component = ".".join(parts[4:-1])
    component = component.replace("linear_attn", "gdn")
    component = component.replace("self_attn", "attn")
    component = component.replace(".", "_")
    return f"model_layers_{layer}_{component}_MatMul_weight_"


def quantize_q4f16(weight: np.ndarray) -> tuple[bytes, bytes, bytes]:
    # PyTorch stores [N, K]. MatMulNBits consumes B as [K, N]. Q4F16 keeps
    # block scales in FP16 and uses asymmetric 4-bit zero points.
    matrix = np.ascontiguousarray(weight.T.astype(np.float16))
    rows, columns = matrix.shape
    block_size = 32
    blocks = math.ceil(rows / block_size)
    packed = np.zeros((columns, blocks, 16), dtype=np.uint8)
    scales = np.zeros((columns, blocks), dtype=np.float16)
    zero_points = np.zeros((columns, math.ceil(blocks / 2)), dtype=np.uint8)
    quantize_matmul_4bits(
        packed,
        matrix,
        scales,
        zero_points,
        block_size,
        columns,
        rows,
        False,
    )
    return packed.tobytes(), scales.tobytes(), zero_points.tobytes()


def patch_export(output: Path) -> None:
    graph_path = output / "onnx/decoder_model_merged_q4f16.onnx"
    data_path = output / "onnx/decoder_model_merged_q4f16.onnx_data"
    model = onnx.load(graph_path, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    source_url = hub_url(SOURCE_MODEL, SOURCE_REVISION, SOURCE_WEIGHTS)
    source_header, source_data_start = safetensors_header(source_url)
    if not all(name in source_header for name in CHANGED_TENSORS):
        missing = [name for name in CHANGED_TENSORS if name not in source_header]
        raise RuntimeError(f"Pinned source checkpoint is missing tensors: {missing}")

    matrix_count = 0
    constant_count = 0
    with data_path.open("r+b") as external_data:
        for index, source_name in enumerate(CHANGED_TENSORS, start=1):
            metadata = source_header[source_name]
            tensor_start, tensor_end = metadata["data_offsets"]
            print(f"[{index:02d}/{len(CHANGED_TENSORS)}] patching {source_name}")
            raw = request_bytes(
                source_url,
                source_data_start + tensor_start,
                source_data_start + tensor_end,
            )
            if source_name.endswith("A_log"):
                a_log = np.frombuffer(raw, dtype="<f4")
                layer = source_name.split(".")[3]
                target_name = f"model.layers.{layer}.gdn.A_neg_exp"
                target = initializers.get(target_name)
                if target is None:
                    raise RuntimeError(f"Missing ONNX initializer: {target_name}")
                replacement = numpy_helper.from_array(
                    np.ascontiguousarray(-np.exp(a_log).astype(np.float32)),
                    target_name,
                )
                position = list(model.graph.initializer).index(target)
                model.graph.initializer.remove(target)
                model.graph.initializer.insert(position, replacement)
                initializers[target_name] = replacement
                constant_count += 1
                continue

            if metadata["dtype"] != "BF16":
                raise RuntimeError(
                    f"Unexpected dtype for {source_name}: {metadata['dtype']}"
                )
            weight = bf16_tensor(raw, metadata["shape"])
            quantized = quantize_q4f16(weight)
            prefix = onnx_prefix(source_name)
            for suffix, replacement in zip(
                ("quant", "scales", "zp"), quantized, strict=True
            ):
                target_name = prefix + suffix
                target = initializers.get(target_name)
                if target is None:
                    raise RuntimeError(f"Missing ONNX initializer: {target_name}")
                offset, length = external_range(target)
                if len(replacement) != length:
                    raise RuntimeError(
                        f"Quantized length mismatch for {target_name}: "
                        f"expected {length}, received {len(replacement)}"
                    )
                external_data.seek(offset)
                external_data.write(replacement)
            matrix_count += 1

    onnx.save_model(model, graph_path)
    try:
        onnx.checker.check_model(graph_path)
    except onnx.checker.ValidationError as error:
        # The qualified graph contains ONNX Runtime contrib operators such as
        # SimplifiedLayerNormalization and CausalConvWithState. The generic
        # ONNX checker has no schema for them; ONNX Runtime session creation is
        # the authoritative graph validation performed after this build.
        if "No Op registered" not in str(error):
            raise
        print(f"Generic ONNX checker skipped contrib-op schemas: {error}")
    print(
        f"Patched {matrix_count} Q4F16 matrices and "
        f"{constant_count} FP32 Mamba constants."
    )


def write_manifest(output: Path) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "export-manifest.json":
            files[str(path.relative_to(output))] = path.stat().st_size
    manifest = {
        "format": "transformers.js-q4f16-text-only",
        "sourceModel": SOURCE_MODEL,
        "sourceRevision": SOURCE_REVISION,
        "baseGraphModel": BASE_MODEL,
        "baseGraphRevision": BASE_REVISION,
        "changedTensors": len(CHANGED_TENSORS),
        "files": files,
        "downloadBytes": sum(files.values()),
    }
    (output / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Export ready: {output}")
    print(f"Text-only model size: {manifest['downloadBytes'] / 1e9:.3f} GB")


def validate_sessions(output: Path) -> None:
    """Make ORT resolve external data and all contrib-operator schemas."""
    onnx_dir = output / "onnx"
    for file_name in (
        "embed_tokens_q4f16.onnx",
        "decoder_model_merged_q4f16.onnx",
    ):
        print(f"Validating ONNX Runtime session: {file_name}")
        session = ort.InferenceSession(
            str(onnx_dir / file_name), providers=["CPUExecutionProvider"]
        )
        if not session.get_inputs() or not session.get_outputs():
            raise RuntimeError(f"Invalid empty ONNX Runtime session: {file_name}")
        del session


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    default_output = (
        repository
        / "experiments/lyrics-lab/public/models"
        / SOURCE_MODEL
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Optional directory containing already-downloaded base ONNX assets.",
    )
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    manifest_path = output / "export-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("sourceRevision") == SOURCE_REVISION
            and manifest.get("baseGraphRevision") == BASE_REVISION
        ):
            print(f"Pinned export already exists: {output}")
            return 0
    output.mkdir(parents=True, exist_ok=True)
    prepare_base_assets(output, arguments.base_dir)
    patch_export(output)
    validate_sessions(output)
    write_manifest(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
