# Local lyric-model lab

The lab compares sixteen editable prompt strategies. Variant D is the detailed
narrative-fidelity control. F is a short neutral ablation, G treats the brief as
a story editor's evidence ledger, and H balances stronger songcraft against a
five-part fidelity gate. I is a compact small-model candidate derived from the
first eight-prompt evaluation: it combines C's strict contract with G's
agency/causality rules without copying any test-story content. J is a
RhymeAI-oriented fact-density experiment that assigns short lines explicit
story jobs and makes rhyme subordinate to fidelity. K tests the opposite
hypothesis with a minimal system prompt and lighter A-like user contract.
L uses the same lightweight user contract as K but explicitly blocks any rhyme
that requires changed facts, malformed grammar, vague imagery, or unnatural
word order.
M adds a hard verbatim contract for words or phrases the user explicitly marks
as required and reports missing quoted requirements in the basic validator.
O is an isolated single-pass long-form experiment. It keeps K's exact system
prompt, sampling, Worker path, and processing, changing only the structure
sentence in K's user prompt for durations over 45 seconds. K remains the locked
rollback baseline.
P uses O's exact long-form user prompt with L's natural-language system prompt.
It isolates whether L's anti-forced-rhyme and factual-language guardrails retain
more quality over additional sections. K, L, and O remain unchanged.
Q adds a stricter literal-referent gate and gives the model a 20% smaller word
ceiling for durations over 45 seconds while retaining the real ACE budget in
post-processing. Generated and retained word counts expose model overruns and
any resulting truncation. K, L, O, and P remain unchanged.
D/F/G/H/I/J deliberately share the same generated user
prompt, seed, model, and sampling settings so their system prompts are the only
experimental variable. Variant E remains the separate two-pass facts-then-lyrics
test.

Six selectable source briefs exercise different failure modes: awkward humor,
quiet grief, cause and effect, an intentionally mundane victory, conflicted
emotion, and place-specific celebration. Edit any preset to switch to a custom
brief. A useful prompt should work across several presets and seeds, not merely
the explicit comedy case.

## Automated model × prompt × story evaluation

The **Run the complete test matrix** panel lets you select any combination of
models, prompt variants, and predefined stories. It runs the Cartesian product
sequentially in model-major order. One model stays loaded while all of its cases
run, then its Worker and WebGPU resources are released before the next model is
loaded. This is intentionally sequential because concurrently resident 2–5 GB
models would make the comparison less reliable and can exhaust browser memory.

The default selection runs prompts A–G and I–M over all six stories with the default
model. Prompt H is available but starts unchecked. A model-load failure is
recorded and the remaining cases for that unavailable model are marked skipped;
the next selected model still runs. Cancelled batches retain exportable partial
results.

Download either:

- CSV for analysis in spreadsheets, Python, R, or a database; or
- JSON for a lossless structured archive.

Each row contains the model repository and pinned revision, story and focus,
system and effective user prompts, seed and sampling controls, raw/cleaned/final
lyrics, validation issues, token and word counts, timing, two-pass fact data,
and any error. Generated CSV cells are quoted and formula-prefixed output is
neutralized for safe spreadsheet opening.

The lyric lab compares prompt variants and six browser models:

- Qwen3.5 4B Q4F16;
- Qwen3.5 2B Q4F16;
- Qwen3.5 2B HauhauCS uncensored Q4F16;
- the production Qwen3.5 0.8B Q4 baseline; and
- RhymeAI Gemma 4 E2B Q4F16.
- RhymeAI Gemma 4 E4B v3 Q4F16.

The uncensored source repository contains BF16 Safetensors rather than ONNX.
Build its local, text-only WebGPU export before selecting it:

```bash
uv venv --python 3.11 .venv-qwen-export
uv pip install --python .venv-qwen-export/bin/python \
  -r scripts/requirements-qwen35-uncensored-export.txt
.venv-qwen-export/bin/python scripts/build-qwen35-uncensored-q4f16.py
```

The generated 1.40 GB model lives under `experiments/lyrics-lab/public/models`
and is intentionally ignored by Git. It preserves the qualified
`onnx-community/Qwen3.5-2B-ONNX-OPT` graph, patches only the 55 intentional
HauhauCS tensor edits, excludes documented BF16 conversion artefacts, and does
not include the vision encoder. The builder validates both generated graphs by
creating native ONNX Runtime sessions. HauhauCS has had its safety alignment
removed, so treat its output as unfiltered experimental content.

The RhymeAI repository contains GGUF models and F16 LoRA adapters rather than
ONNX. Build the local 3.13 GB E2B WebGPU export before selecting it:

```bash
uv venv --python 3.11 .venv-rhyme-export
uv pip install --python .venv-rhyme-export/bin/python \
  -r scripts/requirements-rhymeai-gemma-export.txt
.venv-rhyme-export/bin/python scripts/build-rhymeai-gemma-e2b-q4f16.py
```

To build the larger E4B v3 model instead:

```bash
.venv-rhyme-export/bin/python scripts/build-rhymeai-gemma-e2b-q4f16.py \
  --variant e4b-v3
```

This merges all 490 adapter tensors into 170 affected matrices in the pinned
Gemma 4 E2B or E4B Q4F16 ONNX graph. The model was trained primarily for constrained
line suggestions and lyric rewrites, so full-song prompt quality is itself an
experimental comparison.

Run the lab:

```bash
npm run lyrics:lab -- --host 0.0.0.0
```
