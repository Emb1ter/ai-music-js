# Third-party notices

`ai-music-js` is distributed under the MIT License in [`LICENSE`](LICENSE).
It includes or interoperates with the third-party software and model artifacts
listed below. Their licenses apply to their respective components.

## Components included in the npm package

### Transformers.js 4.2.0

- Project: <https://github.com/huggingface/transformers.js>
- License: Apache License 2.0
- Use: tokenizer construction inside the bundled inference Worker
- License text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

### ONNX Runtime Web 1.27.0

- Project: <https://github.com/microsoft/onnxruntime>
- License: MIT
- Use: bundled ONNX Runtime Web/Common JavaScript runtime and redistributed
  WebAssembly runtime
- License text:
  [`licenses/onnxruntime-MIT.txt`](licenses/onnxruntime-MIT.txt)
- Upstream binary notices:
  [`licenses/onnxruntime-ThirdPartyNotices.txt`](licenses/onnxruntime-ThirdPartyNotices.txt)

### fflate 0.8.3

- Project: <https://github.com/101arrowz/fflate>
- License: MIT
- Use: decompression support bundled into the inference Worker
- License text: [`licenses/fflate-MIT.txt`](licenses/fflate-MIT.txt)

## Model artifacts downloaded at runtime

Model weights are not included in the npm package. The browser downloads them
from pinned Hugging Face revisions.

### ACE-Step 1.5 XL Turbo

- Model: <https://huggingface.co/ACE-Step/acestep-v15-xl-turbo>
- Implementation: <https://github.com/ace-step/ACE-Step-1.5>
- License: MIT
- Copyright: Copyright (c) 2026 ACEStep
- Use: original XL Turbo checkpoint and architecture from which the
  browser-specific ONNX graphs were exported
- License text: [`licenses/ACE-Step-MIT.txt`](licenses/ACE-Step-MIT.txt)

### ACE-Step 1.5 5 Hz language model

- Model:
  <https://huggingface.co/ACE-Step/acestep-5Hz-lm-4B>
- Browser ONNX export:
  <https://huggingface.co/emb1ter/ACE-Step-v1.5-5Hz-LM-4B-ONNX-WebGPU>
- Browser INT8-weight / FP32-compute and reference FP32 ONNX exports:
  <https://huggingface.co/emb1ter/ACE-Step-v1.5-5Hz-LM-4B-FP32-ONNX-WebGPU>
- License: MIT
- Copyright: Copyright (c) 2026 ACEStep
- Use: browser-local music metadata and semantic-code planning
- License text: [`licenses/ACE-Step-MIT.txt`](licenses/ACE-Step-MIT.txt)

### Qwen3.5 0.8B

- Browser model:
  <https://huggingface.co/onnx-community/Qwen3.5-0.8B-Text-ONNX>
- Original family: <https://huggingface.co/Qwen>
- License: Apache License 2.0
- Use: optional browser-local lyric writing
- License text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

### ACE-Step v1.5 ONNX conversion

- Model repository:
  <https://huggingface.co/shreyask/ACE-Step-v1.5-ONNX>
- License: Apache License 2.0
- Use: experimental conversion starting point and the shared text encoder,
  lyric embedding, and support assets downloaded by the browser
- License text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)

The XL condition encoder, XL Turbo DiT, and official-checkpoint-derived FP32
Oobleck VAE decoder in
<https://huggingface.co/emb1ter/ACE-Step-v1.5-XL-Turbo-ONNX-WebGPU> are fresh
checkpoint-specific exports. Attribution to the unofficial conversion is
retained because it provided the experimental ONNX starting point and shared
runtime assets.

## No endorsement

The third-party project names are used only for identification and
attribution. This project is unofficial and is not affiliated with or endorsed
by ACE-Step, Hugging Face, Microsoft, or the unofficial ONNX conversion
author.
