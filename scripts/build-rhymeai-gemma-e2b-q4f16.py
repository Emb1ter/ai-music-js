#!/usr/bin/env python3
"""Build a local text-only WebGPU export of a RhymeAI Gemma 4 LoRA.

The source repository publishes complete GGUF models and F16 GGUF LoRA
adapters, while Transformers.js consumes ONNX. This builder starts from the
pinned, browser-qualified Gemma 4 Q4F16 ONNX graph, merges the matching
rank-16 adapter, and requantizes the affected matrices to ONNX Runtime
MatMulNBits Q4F16. Both the E2B export and the larger E4B v3 export are
supported.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from gguf import GGUFReader
from onnxruntime.quantization.matmul_nbits_quantizer import quantize_matmul_4bits


SOURCE_MODEL = "Shayde182/rhymeai-gemma-4-gguf"
SOURCE_REVISION = "27492c6696f2f868bbbe3aeafb633009ef932598"

VARIANTS = {
    "e2b": {
        "export_model_id": SOURCE_MODEL,
        "source_adapter": "gemma-4-E2B-it-rhymeai-lora-adapter-F16.gguf",
        "source_adapter_bytes": 50_709_920,
        "base_model": "onnx-community/gemma-4-E2B-it-ONNX",
        "base_revision": "9f4bef82ea6e296bc69f8a2f5939f73af81b07a6",
        "base_assets": {
            "config.json": 5_549,
            "generation_config.json": 238,
            "chat_template.jinja": 16_317,
            "tokenizer.json": 19_439_251,
            "tokenizer_config.json": 18_807,
            "onnx/decoder_model_merged_q4f16.onnx": 673_231,
            "onnx/decoder_model_merged_q4f16.onnx_data": 1_519_700_992,
            "onnx/embed_tokens_q4f16.onnx": 5_621,
            "onnx/embed_tokens_q4f16.onnx_data": 1_590_689_792,
        },
        "layers": 35,
        "adapter_tensors": 490,
        "patched_matrices": 170,
        "format": "transformers.js-gemma4-e2b-q4f16-text-only",
    },
    "e4b-v3": {
        "export_model_id": "ai-music-js/rhymeai-gemma4-e4b-v3",
        "source_adapter": "gemma-4-E4B-it-rhymeai-v3-lora-adapter-F16.gguf",
        "source_adapter_bytes": 73_441_120,
        "base_model": "onnx-community/gemma-4-E4B-it-ONNX",
        "base_revision": "843f250f23bc91754def1e0f0db390dacd1e6b05",
        "base_assets": {
            "config.json": 5_741,
            "generation_config.json": 238,
            "chat_template.jinja": 16_317,
            "tokenizer.json": 19_439_251,
            "tokenizer_config.json": 18_807,
            "onnx/decoder_model_merged_q4f16.onnx": 850_610,
            "onnx/decoder_model_merged_q4f16.onnx_data": 2_074_847_232,
            "onnx/decoder_model_merged_q4f16.onnx_data_1": 812_318_720,
            "onnx/embed_tokens_q4f16.onnx": 5_619,
            "onnx/embed_tokens_q4f16.onnx_data": 2_017_460_224,
        },
        "layers": 42,
        "adapter_tensors": 588,
        "patched_matrices": 216,
        "format": "transformers.js-gemma4-e4b-v3-q4f16-text-only",
    },
}

EXPORT_MODEL_ID = ""
SOURCE_ADAPTER = ""
SOURCE_ADAPTER_BYTES = 0
BASE_MODEL = ""
BASE_REVISION = ""
BASE_ASSETS: dict[str, int] = {}
NUM_LAYERS = 0
EXPECTED_ADAPTER_TENSORS = 0
EXPECTED_PATCHED_MATRICES = 0
EXPORT_FORMAT = ""

BLOCK_SIZE = 32


def configure_variant(name: str) -> None:
    global EXPORT_MODEL_ID
    global SOURCE_ADAPTER
    global SOURCE_ADAPTER_BYTES
    global BASE_MODEL
    global BASE_REVISION
    global BASE_ASSETS
    global NUM_LAYERS
    global EXPECTED_ADAPTER_TENSORS
    global EXPECTED_PATCHED_MATRICES
    global EXPORT_FORMAT

    variant = VARIANTS[name]
    EXPORT_MODEL_ID = str(variant["export_model_id"])
    SOURCE_ADAPTER = str(variant["source_adapter"])
    SOURCE_ADAPTER_BYTES = int(variant["source_adapter_bytes"])
    BASE_MODEL = str(variant["base_model"])
    BASE_REVISION = str(variant["base_revision"])
    BASE_ASSETS = dict(variant["base_assets"])
    NUM_LAYERS = int(variant["layers"])
    EXPECTED_ADAPTER_TENSORS = int(variant["adapter_tensors"])
    EXPECTED_PATCHED_MATRICES = int(variant["patched_matrices"])
    EXPORT_FORMAT = str(variant["format"])


def hub_url(model: str, revision: str, file_name: str) -> str:
    return f"https://huggingface.co/{model}/resolve/{revision}/{file_name}"


def download(url: str, output: Path, expected_bytes: int, force: bool = False) -> None:
    if force:
        output.unlink(missing_ok=True)
        output.with_suffix(output.suffix + ".part").unlink(missing_ok=True)
    if output.is_file() and output.stat().st_size == expected_bytes:
        print(f"  cached {output.name}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    copied = partial.stat().st_size if partial.is_file() else 0
    if copied > expected_bytes:
        partial.unlink()
        copied = 0
    headers = {"Range": f"bytes={copied}-"} if copied else {}
    print(
        f"  downloading {output.name} ({expected_bytes / 1e9:.3f} GB)"
        + (f" from {copied / 1e6:,.0f} MB" if copied else "")
    )
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        if copied and response.status != 206:
            partial.unlink(missing_ok=True)
            copied = 0
        mode = "ab" if copied else "wb"
        with partial.open(mode) as target:
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


def prepare_base_assets(output: Path, restore_decoder: bool) -> None:
    print(f"Preparing the qualified {BASE_MODEL} Q4F16 text-only graph.")
    for relative, expected_bytes in BASE_ASSETS.items():
        download(
            hub_url(BASE_MODEL, BASE_REVISION, relative),
            output / relative,
            expected_bytes,
            force=restore_decoder
            and relative.startswith("onnx/decoder_model_merged_q4f16.onnx_data"),
        )


def external_range(tensor: onnx.TensorProto) -> tuple[int, int]:
    fields = {item.key: item.value for item in tensor.external_data}
    return int(fields.get("offset", "0")), int(fields.get("length", "0"))


def external_location(tensor: onnx.TensorProto) -> str:
    fields = {item.key: item.value for item in tensor.external_data}
    location = fields.get("location")
    if not location:
        raise RuntimeError(f"Missing external-data location for {tensor.name}")
    return location


def read_external_array(
    external_files, tensor: onnx.TensorProto, dtype: np.dtype
) -> np.ndarray:
    external_data = external_files[external_location(tensor)]
    offset, length = external_range(tensor)
    external_data.seek(offset)
    raw = external_data.read(length)
    if len(raw) != length:
        raise RuntimeError(f"Short external-data read for {tensor.name}")
    return np.frombuffer(raw, dtype=dtype).reshape(tensor.dims).copy()


def write_external_array(external_files, tensor: onnx.TensorProto, value: bytes) -> None:
    external_data = external_files[external_location(tensor)]
    offset, length = external_range(tensor)
    if len(value) != length:
        raise RuntimeError(
            f"Quantized length mismatch for {tensor.name}: "
            f"expected {length}, received {len(value)}"
        )
    external_data.seek(offset)
    external_data.write(value)


def unpack_zero_points(packed: np.ndarray, blocks: int) -> np.ndarray:
    result = np.empty((packed.shape[0], blocks), dtype=np.int16)
    result[:, 0::2] = (packed[:, : math.ceil(blocks / 2)] & 0x0F)[:, : result[:, 0::2].shape[1]]
    if blocks > 1:
        result[:, 1::2] = (packed[:, : blocks // 2] >> 4)[:, : result[:, 1::2].shape[1]]
    return result


def dequantize_q4f16(
    packed: np.ndarray, scales: np.ndarray, zero_points: np.ndarray
) -> np.ndarray:
    columns, blocks, _ = packed.shape
    values = np.empty((columns, blocks, BLOCK_SIZE), dtype=np.int16)
    values[:, :, 0::2] = packed & 0x0F
    values[:, :, 1::2] = packed >> 4
    zeros = unpack_zero_points(zero_points, blocks)
    return (
        (values - zeros[:, :, None]).astype(np.float32)
        * scales.astype(np.float32)[:, :, None]
    ).reshape(columns, blocks * BLOCK_SIZE)


def quantize_q4f16(weight: np.ndarray) -> tuple[bytes, bytes, bytes]:
    # PyTorch/GGUF weights use [N, K]. MatMulNBits consumes B as [K, N].
    matrix = np.ascontiguousarray(weight.T.astype(np.float16))
    rows, columns = matrix.shape
    blocks = math.ceil(rows / BLOCK_SIZE)
    packed = np.zeros((columns, blocks, 16), dtype=np.uint8)
    scales = np.zeros((columns, blocks), dtype=np.float16)
    zero_points = np.zeros((columns, math.ceil(blocks / 2)), dtype=np.uint8)
    quantize_matmul_4bits(
        packed,
        matrix,
        scales,
        zero_points,
        BLOCK_SIZE,
        columns,
        rows,
        False,
    )
    return packed.tobytes(), scales.tobytes(), zero_points.tobytes()


def adapter_alpha(reader: GGUFReader) -> float:
    field = reader.fields["adapter.lora.alpha"]
    value = field.parts[field.data[0]]
    return float(value[0])


def adapter_pair(
    tensors: dict[str, np.ndarray], consumed: set[str], prefix: str
) -> tuple[np.ndarray, np.ndarray]:
    a_name = prefix + ".weight.lora_a"
    b_name = prefix + ".weight.lora_b"
    try:
        a = tensors[a_name].astype(np.float32)
        b = tensors[b_name].astype(np.float32)
    except KeyError as error:
        raise RuntimeError(f"Missing adapter tensor: {error.args[0]}") from error
    consumed.update((a_name, b_name))
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise RuntimeError(f"Invalid LoRA pair shapes for {prefix}: {a.shape}, {b.shape}")
    return a, b


def add_lora_in_place(
    weight: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float
) -> None:
    if weight.shape != (b.shape[0], a.shape[1]):
        raise RuntimeError(
            f"LoRA/base shape mismatch: base {weight.shape}, A {a.shape}, B {b.shape}"
        )
    # Chunk the low-rank product to keep peak memory bounded for wide MLPs.
    for start in range(0, b.shape[0], 1024):
        end = min(start + 1024, b.shape[0])
        weight[start:end] += (b[start:end] @ a) * scale


def matrix_prefix(layer: int, component: str) -> str:
    return f"model_layers_{layer}_{component}_MatMul_weight_"


def read_matrix(external_files, initializers, prefix: str) -> np.ndarray:
    packed = read_external_array(external_files, initializers[prefix + "quant"], np.uint8)
    scales = read_external_array(external_files, initializers[prefix + "scales"], "<f2")
    zero_points = read_external_array(external_files, initializers[prefix + "zp"], np.uint8)
    return dequantize_q4f16(packed, scales, zero_points)


def write_matrix(external_files, initializers, prefix: str, weight: np.ndarray) -> None:
    for suffix, value in zip(
        ("quant", "scales", "zp"), quantize_q4f16(weight), strict=True
    ):
        write_external_array(external_files, initializers[prefix + suffix], value)


def patch_single_matrix(
    external_files,
    initializers,
    prefix: str,
    a: np.ndarray,
    b: np.ndarray,
    scale: float,
) -> None:
    weight = read_matrix(external_files, initializers, prefix)
    add_lora_in_place(weight, a, b, scale)
    write_matrix(external_files, initializers, prefix, weight)


def patch_export(output: Path, adapter_path: Path) -> None:
    graph_path = output / "onnx/decoder_model_merged_q4f16.onnx"
    model = onnx.load(graph_path, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    reader = GGUFReader(str(adapter_path))
    if len(reader.tensors) != EXPECTED_ADAPTER_TENSORS:
        raise RuntimeError(
            f"Expected {EXPECTED_ADAPTER_TENSORS} adapter tensors, "
            f"received {len(reader.tensors)}"
        )
    tensors = {tensor.name: tensor.data for tensor in reader.tensors}
    consumed: set[str] = set()
    alpha = adapter_alpha(reader)
    patched = 0

    locations = sorted(
        {
            external_location(tensor)
            for tensor in model.graph.initializer
            if tensor.external_data
        }
    )
    with ExitStack() as stack:
        external_files = {
            location: stack.enter_context((graph_path.parent / location).open("r+b"))
            for location in locations
        }
        for layer in range(NUM_LAYERS):
            for adapter_name, onnx_name in (
                ("attn_q", "attn_q_proj"),
                ("attn_k", "attn_k_proj"),
                ("attn_v", "attn_v_proj"),
                ("attn_output", "attn_o_proj"),
            ):
                a, b = adapter_pair(
                    tensors, consumed, f"blk.{layer}.{adapter_name}"
                )
                prefix = matrix_prefix(layer, onnx_name)
                has_target = prefix + "quant" in initializers
                if not has_target:
                    # Gemma 4 shares K/V for a suffix of its layers. The
                    # adapter carries structurally-present zero-B pairs for
                    # those absent projections.
                    if np.any(b):
                        raise RuntimeError(
                            f"Non-zero adapter has no ONNX target: blk.{layer}.{adapter_name}"
                        )
                    continue
                scale = alpha / a.shape[0]
                print(f"[{patched + 1:03d}/{EXPECTED_PATCHED_MATRICES}] {prefix}")
                patch_single_matrix(
                    external_files, initializers, prefix, a, b, scale
                )
                patched += 1

            gate_a, gate_b = adapter_pair(
                tensors, consumed, f"blk.{layer}.ffn_gate"
            )
            up_a, up_b = adapter_pair(tensors, consumed, f"blk.{layer}.ffn_up")
            gate_up_prefix = matrix_prefix(layer, "mlp_gate_up_proj")
            gate_up = read_matrix(external_files, initializers, gate_up_prefix)
            midpoint = gate_up.shape[0] // 2
            add_lora_in_place(
                gate_up[:midpoint], gate_a, gate_b, alpha / gate_a.shape[0]
            )
            add_lora_in_place(
                gate_up[midpoint:], up_a, up_b, alpha / up_a.shape[0]
            )
            print(
                f"[{patched + 1:03d}/{EXPECTED_PATCHED_MATRICES}] {gate_up_prefix}"
            )
            write_matrix(external_files, initializers, gate_up_prefix, gate_up)
            patched += 1

            down_a, down_b = adapter_pair(
                tensors, consumed, f"blk.{layer}.ffn_down"
            )
            down_prefix = matrix_prefix(layer, "mlp_down_proj")
            print(f"[{patched + 1:03d}/{EXPECTED_PATCHED_MATRICES}] {down_prefix}")
            patch_single_matrix(
                external_files,
                initializers,
                down_prefix,
                down_a,
                down_b,
                alpha / down_a.shape[0],
            )
            patched += 1

    if patched != EXPECTED_PATCHED_MATRICES:
        raise RuntimeError(
            f"Expected to patch {EXPECTED_PATCHED_MATRICES} matrices, patched {patched}"
        )
    unconsumed = sorted(set(tensors) - consumed)
    if unconsumed:
        raise RuntimeError(f"Unconsumed adapter tensors: {unconsumed[:8]}")
    print(
        f"Merged all {len(consumed)} LoRA tensors into {patched} Q4F16 matrices."
    )


def validate_sessions(output: Path) -> None:
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


def write_manifest(output: Path) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.name != "export-manifest.json"
        ):
            files[str(path.relative_to(output))] = path.stat().st_size
    manifest = {
        "format": EXPORT_FORMAT,
        "exportModelId": EXPORT_MODEL_ID,
        "sourceModel": SOURCE_MODEL,
        "sourceRevision": SOURCE_REVISION,
        "sourceAdapter": SOURCE_ADAPTER,
        "baseGraphModel": BASE_MODEL,
        "baseGraphRevision": BASE_REVISION,
        "adapterTensors": EXPECTED_ADAPTER_TENSORS,
        "patchedMatrices": EXPECTED_PATCHED_MATRICES,
        "files": files,
        "downloadBytes": sum(files.values()),
    }
    (output / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Export ready: {output}")
    print(f"Text-only model size: {manifest['downloadBytes'] / 1e9:.3f} GB")


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="e2b",
        help="RhymeAI checkpoint to merge (default: e2b).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--adapter-file",
        type=Path,
        help="Optional already-downloaded matching F16 GGUF LoRA adapter.",
    )
    arguments = parser.parse_args()
    configure_variant(arguments.variant)
    default_output = (
        repository / "experiments/lyrics-lab/public/models" / EXPORT_MODEL_ID
    )
    output = (arguments.output_dir or default_output).resolve()
    manifest_path = output / "export-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("sourceRevision") == SOURCE_REVISION
            and manifest.get("baseGraphRevision") == BASE_REVISION
            and manifest.get("patchedMatrices") == EXPECTED_PATCHED_MATRICES
        ):
            print(f"Pinned export already exists: {output}")
            return 0

    output.mkdir(parents=True, exist_ok=True)
    patch_marker = output / ".patching"
    prepare_base_assets(output, restore_decoder=patch_marker.exists())
    adapter_path = (
        arguments.adapter_file.resolve()
        if arguments.adapter_file
        else output / ".source-adapter.gguf"
    )
    if not arguments.adapter_file:
        download(
            hub_url(SOURCE_MODEL, SOURCE_REVISION, SOURCE_ADAPTER),
            adapter_path,
            SOURCE_ADAPTER_BYTES,
        )
    if adapter_path.stat().st_size != SOURCE_ADAPTER_BYTES:
        raise RuntimeError(f"Unexpected adapter size: {adapter_path}")

    patch_marker.write_text("RhymeAI LoRA merge in progress\n", encoding="utf-8")
    try:
        patch_export(output, adapter_path)
        validate_sessions(output)
        if not arguments.adapter_file:
            adapter_path.unlink(missing_ok=True)
        patch_marker.unlink(missing_ok=True)
        write_manifest(output)
    except Exception:
        # The marker forces restoration of the pristine decoder data before a
        # retry, preventing the LoRA from being applied twice.
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
