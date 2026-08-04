import { Tensor } from "@huggingface/transformers";

export const PLANNER_Q8_HEAD_WEIGHT_FILE = "lm_head_q8.bin";
export const PLANNER_Q8_HEAD_SCALE_FILE = "lm_head_q8_scales.f16";
export const PLANNER_Q8_HEAD_WEIGHT_BYTES = 556_042_240;
export const PLANNER_Q8_HEAD_SCALE_BYTES = 34_752_640;
export const PLANNER_Q8_HEAD_VOCAB_SIZE = 217_204;
export const PLANNER_Q8_HEAD_HIDDEN_SIZE = 2_560;
export const PLANNER_Q8_HEAD_BLOCK_SIZE = 32;

type GpuBuffer = {
  destroy(): void;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
};

type GpuDevice = {
  createBuffer(descriptor: Record<string, unknown>): GpuBuffer;
  createComputePipeline(descriptor: Record<string, unknown>): {
    getBindGroupLayout(index: number): unknown;
  };
  createShaderModule(descriptor: Record<string, unknown>): unknown;
  createBindGroup(descriptor: Record<string, unknown>): unknown;
  createCommandEncoder(): {
    beginComputePass(): {
      setPipeline(pipeline: unknown): void;
      setBindGroup(index: number, group: unknown): void;
      dispatchWorkgroups(x: number, y?: number): void;
      end(): void;
    };
    copyBufferToBuffer(
      source: GpuBuffer,
      sourceOffset: number,
      destination: GpuBuffer,
      destinationOffset: number,
      size: number,
    ): void;
    finish(): unknown;
  };
  queue: {
    writeBuffer(
      buffer: GpuBuffer,
      offset: number,
      data: ArrayBuffer,
      dataOffset?: number,
      size?: number,
    ): void;
    submit(commands: unknown[]): void;
  };
};

type PlannerHeadProgress = (
  file: string,
  loaded: number,
  total: number,
) => void;

const gpuFlags = () => {
  const scope = globalThis as typeof globalThis & {
    GPUBufferUsage: Record<string, number>;
    GPUMapMode: Record<string, number>;
  };
  return {
    buffer: scope.GPUBufferUsage,
    map: scope.GPUMapMode,
  };
};

const align4 = (value: number) => Math.ceil(value / 4) * 4;

const MODEL_HEAD_SHADER = /* wgsl */ `
struct Params {
  vocab_size: u32,
  row_count: u32,
  hidden_size: u32,
  blocks_per_row: u32,
  dispatch_x: u32,
  batch_size: u32,
  _padding0: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> hidden: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

fn signed_int8(value: u32) -> f32 {
  return f32(i32(value << 24u) >> 24u);
}

fn hidden_value(batch: u32, column: u32) -> f32 {
  let index = batch * params.hidden_size + column;
  let values = unpack2x16float(hidden[index / 2u]);
  return select(values.x, values.y, (index & 1u) == 1u);
}

fn scale_value(row: u32, block: u32) -> f32 {
  let index = row * params.blocks_per_row + block;
  let values = unpack2x16float(scales[index / 2u]);
  return select(values.x, values.y, (index & 1u) == 1u);
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let output_row = group_id.x + group_id.y * params.dispatch_x;
  let total_rows = params.row_count * params.batch_size;
  if (output_row >= total_rows) {
    return;
  }
  let batch = output_row / params.row_count;
  let vocab_row = output_row - batch * params.row_count;
  var sum = 0.0;
  var column = local_id.x;
  loop {
    if (column >= params.hidden_size) {
      break;
    }
    let weight_index = vocab_row * params.hidden_size + column;
    let packed = weights[weight_index / 4u];
    let shift = (weight_index & 3u) * 8u;
    let quantized = signed_int8((packed >> shift) & 255u);
    let scale = scale_value(vocab_row, column / ${PLANNER_Q8_HEAD_BLOCK_SIZE}u);
    sum += hidden_value(batch, column) * quantized * scale;
    column += 256u;
  }
  partial[local_id.x] = sum;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (local_id.x < stride) {
      partial[local_id.x] += partial[local_id.x + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride /= 2u;
  }
  if (local_id.x == 0u) {
    output[output_row] = partial[0];
  }
}
`;

const TOKEN_EMBEDDING_SHADER = /* wgsl */ `
struct Params {
  token_count: u32,
  hidden_size: u32,
  blocks_per_row: u32,
  _padding: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> token_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> embeddings: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn signed_int8(value: u32) -> f32 {
  return f32(i32(value << 24u) >> 24u);
}

fn scale_value(row: u32, block: u32) -> f32 {
  let index = row * params.blocks_per_row + block;
  let values = unpack2x16float(scales[index / 2u]);
  return select(values.x, values.y, (index & 1u) == 1u);
}

fn embedding_value(row: u32, column: u32) -> f32 {
  let weight_index = row * params.hidden_size + column;
  let packed = weights[weight_index / 4u];
  let shift = (weight_index & 3u) * 8u;
  let quantized = signed_int8((packed >> shift) & 255u);
  return quantized * scale_value(row, column / ${PLANNER_Q8_HEAD_BLOCK_SIZE}u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let pair_index = global_id.x;
  let pairs_per_token = params.hidden_size / 2u;
  let total_pairs = params.token_count * pairs_per_token;
  if (pair_index >= total_pairs) {
    return;
  }
  let token_index = pair_index / pairs_per_token;
  let column = (pair_index - token_index * pairs_per_token) * 2u;
  let row = token_ids[token_index];
  let values = vec2<f32>(
    embedding_value(row, column),
    embedding_value(row, column + 1u),
  );
  embeddings[pair_index] = pack2x16float(values);
}
`;

const fetchIntoGpuBuffer = async (
  device: GpuDevice,
  url: string,
  file: string,
  expectedBytes: number,
  progress: PlannerHeadProgress,
) => {
  const response = await fetch(url, {
    mode: "cors",
    credentials: "omit",
  });
  if (!response.ok || !response.body) {
    throw new Error(
      `Planner WebGPU head ${file} returned HTTP ${response.status}.`,
    );
  }
  const declaredBytes = Number(response.headers.get("content-length"));
  if (
    Number.isFinite(declaredBytes) &&
    declaredBytes > 0 &&
    declaredBytes !== expectedBytes
  ) {
    throw new Error(
      `Planner WebGPU head ${file} is ${declaredBytes} bytes; expected ${expectedBytes}.`,
    );
  }
  const { buffer: usage } = gpuFlags();
  const buffer = device.createBuffer({
    label: `ACE planner ${file}`,
    size: align4(expectedBytes),
    usage: usage.STORAGE | usage.COPY_DST,
  });
  const reader = response.body.getReader();
  let loaded = 0;
  let written = 0;
  let carry = new Uint8Array(0);
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      if (loaded + value.byteLength > expectedBytes) {
        throw new Error(`Planner WebGPU head ${file} exceeded its manifest size.`);
      }
      loaded += value.byteLength;

      let chunk = value;
      if (carry.byteLength > 0) {
        const combined = new Uint8Array(carry.byteLength + value.byteLength);
        combined.set(carry);
        combined.set(value, carry.byteLength);
        chunk = combined;
        carry = new Uint8Array(0);
      }
      const alignedBytes = chunk.byteLength - (chunk.byteLength % 4);
      if (alignedBytes > 0) {
        device.queue.writeBuffer(
          buffer,
          written,
          chunk.buffer,
          chunk.byteOffset,
          alignedBytes,
        );
        written += alignedBytes;
      }
      if (alignedBytes < chunk.byteLength) {
        carry = chunk.slice(alignedBytes);
      }
      progress(file, loaded, expectedBytes);
    }
  } catch (error) {
    buffer.destroy();
    await reader.cancel().catch(() => undefined);
    throw error;
  }
  if (loaded !== expectedBytes) {
    buffer.destroy();
    throw new Error(
      `Planner WebGPU head ${file} downloaded ${loaded} bytes; expected ${expectedBytes}.`,
    );
  }
  if (carry.byteLength !== 0 || written !== expectedBytes) {
    buffer.destroy();
    throw new Error(
      `Planner WebGPU head ${file} could not align ${expectedBytes} streamed bytes for WebGPU upload.`,
    );
  }
  return buffer;
};

export class PlannerQ8WebGpuHead {
  private constructor(
    private readonly device: GpuDevice,
    private readonly weights: GpuBuffer,
    private readonly scales: GpuBuffer,
    private readonly pipeline: {
      getBindGroupLayout(index: number): unknown;
    },
    private readonly embeddingPipeline: {
      getBindGroupLayout(index: number): unknown;
    },
  ) {}

  static async create(
    modelId: string,
    revision: string,
    progress: PlannerHeadProgress,
  ) {
    const navigatorWithGpu = self.navigator as Navigator & {
      gpu?: {
        requestAdapter(options?: Record<string, unknown>): Promise<{
          limits: {
            maxBufferSize: number;
            maxStorageBufferBindingSize: number;
          };
          requestDevice(options?: Record<string, unknown>): Promise<GpuDevice>;
        } | null>;
      };
    };
    const adapter = await navigatorWithGpu.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("Planner WebGPU output head could not acquire an adapter.");
    }
    if (
      Number(adapter.limits.maxBufferSize) < PLANNER_Q8_HEAD_WEIGHT_BYTES ||
      Number(adapter.limits.maxStorageBufferBindingSize) <
        PLANNER_Q8_HEAD_WEIGHT_BYTES
    ) {
      throw new Error(
        "Planner WebGPU output head requires a 556 MB storage-buffer binding; " +
          `this adapter reports ${Number(adapter.limits.maxStorageBufferBindingSize)} bytes.`,
      );
    }
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: PLANNER_Q8_HEAD_WEIGHT_BYTES,
        maxStorageBufferBindingSize: PLANNER_Q8_HEAD_WEIGHT_BYTES,
      },
    });
    const base =
      `https://huggingface.co/${modelId}/resolve/` +
      `${encodeURIComponent(revision)}/`;
    const weights = await fetchIntoGpuBuffer(
      device,
      `${base}${PLANNER_Q8_HEAD_WEIGHT_FILE}`,
      PLANNER_Q8_HEAD_WEIGHT_FILE,
      PLANNER_Q8_HEAD_WEIGHT_BYTES,
      progress,
    );
    let scales: GpuBuffer;
    try {
      scales = await fetchIntoGpuBuffer(
        device,
        `${base}${PLANNER_Q8_HEAD_SCALE_FILE}`,
        PLANNER_Q8_HEAD_SCALE_FILE,
        PLANNER_Q8_HEAD_SCALE_BYTES,
        progress,
      );
    } catch (error) {
      weights.destroy();
      throw error;
    }
    const pipeline = device.createComputePipeline({
      label: "ACE planner Q8 language-model head",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "ACE planner Q8 head shader",
          code: MODEL_HEAD_SHADER,
        }),
        entryPoint: "main",
      },
    });
    const embeddingPipeline = device.createComputePipeline({
      label: "ACE planner Q8 token embedding",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "ACE planner Q8 embedding shader",
          code: TOKEN_EMBEDDING_SHADER,
        }),
        entryPoint: "main",
      },
    });
    return new PlannerQ8WebGpuHead(
      device,
      weights,
      scales,
      pipeline,
      embeddingPipeline,
    );
  }

  async embed(inputIds: Tensor) {
    const [batchSize = 0, sequenceLength = 0] = inputIds.dims;
    const tokenCount = batchSize * sequenceLength;
    if (tokenCount < 1 || inputIds.data.length !== tokenCount) {
      throw new Error(
        `Planner input_ids has invalid shape ${JSON.stringify(inputIds.dims)}.`,
      );
    }
    const tokenData = new Uint32Array(tokenCount);
    for (let index = 0; index < tokenCount; index += 1) {
      const token = Number(inputIds.data[index]);
      if (
        !Number.isInteger(token) ||
        token < 0 ||
        token >= PLANNER_Q8_HEAD_VOCAB_SIZE
      ) {
        throw new Error(`Planner input token ${token} is outside the vocabulary.`);
      }
      tokenData[index] = token;
    }
    const { buffer: usage, map } = gpuFlags();
    const tokens = this.device.createBuffer({
      label: "ACE planner input token IDs",
      size: align4(tokenData.byteLength),
      usage: usage.STORAGE | usage.COPY_DST,
    });
    const embeddingBytes =
      tokenCount * PLANNER_Q8_HEAD_HIDDEN_SIZE * Uint16Array.BYTES_PER_ELEMENT;
    const embeddings = this.device.createBuffer({
      label: "ACE planner token embeddings",
      size: embeddingBytes,
      usage: usage.STORAGE | usage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      label: "ACE planner token embeddings readback",
      size: embeddingBytes,
      usage: usage.COPY_DST | usage.MAP_READ,
    });
    const paramsData = new Uint32Array([
      tokenCount,
      PLANNER_Q8_HEAD_HIDDEN_SIZE,
      PLANNER_Q8_HEAD_HIDDEN_SIZE / PLANNER_Q8_HEAD_BLOCK_SIZE,
      0,
    ]);
    const params = this.device.createBuffer({
      label: "ACE planner embedding parameters",
      size: paramsData.byteLength,
      usage: usage.UNIFORM | usage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      tokens,
      0,
      tokenData.buffer,
      tokenData.byteOffset,
      tokenData.byteLength,
    );
    this.device.queue.writeBuffer(
      params,
      0,
      paramsData.buffer,
      paramsData.byteOffset,
      paramsData.byteLength,
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.embeddingPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weights } },
        { binding: 1, resource: { buffer: this.scales } },
        { binding: 2, resource: { buffer: tokens } },
        { binding: 3, resource: { buffer: embeddings } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.embeddingPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(
        (tokenCount * PLANNER_Q8_HEAD_HIDDEN_SIZE) / 2 / 256,
      ),
    );
    pass.end();
    encoder.copyBufferToBuffer(
      embeddings,
      0,
      readback,
      0,
      embeddingBytes,
    );
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(map.READ);
      const values = new Uint16Array(
        readback.getMappedRange().slice(0),
      );
      readback.unmap();
      return new Tensor("float16", values, [
        batchSize,
        sequenceLength,
        PLANNER_Q8_HEAD_HIDDEN_SIZE,
      ]);
    } finally {
      tokens.destroy();
      embeddings.destroy();
      readback.destroy();
      params.destroy();
    }
  }

  async forward(lastHiddenState: Tensor) {
    const [batchSize = 0, sequenceLength = 0, hiddenSize = 0] =
      lastHiddenState.dims;
    if (
      batchSize < 1 ||
      sequenceLength < 1 ||
      hiddenSize !== PLANNER_Q8_HEAD_HIDDEN_SIZE
    ) {
      throw new Error(
        `Planner body returned ${JSON.stringify(lastHiddenState.dims)}; ` +
          `expected [batch, sequence, ${PLANNER_Q8_HEAD_HIDDEN_SIZE}].`,
      );
    }
    const source = lastHiddenState.data as ArrayBufferView;
    if (source.byteLength !== batchSize * sequenceLength * hiddenSize * 2) {
      throw new Error(
        "Planner body last_hidden_state is not a packed float16 tensor.",
      );
    }
    const finalRows = new Uint16Array(batchSize * hiddenSize);
    const sourceRows = new Uint16Array(
      source.buffer,
      source.byteOffset,
      source.byteLength / 2,
    );
    for (let batch = 0; batch < batchSize; batch += 1) {
      const rowStart =
        (batch * sequenceLength + sequenceLength - 1) * hiddenSize;
      finalRows.set(
        sourceRows.subarray(rowStart, rowStart + hiddenSize),
        batch * hiddenSize,
      );
    }

    const { buffer: usage, map } = gpuFlags();
    const hidden = this.device.createBuffer({
      label: "ACE planner final hidden state",
      size: finalRows.byteLength,
      usage: usage.STORAGE | usage.COPY_DST,
    });
    const outputBytes =
      batchSize * PLANNER_Q8_HEAD_VOCAB_SIZE * Float32Array.BYTES_PER_ELEMENT;
    const output = this.device.createBuffer({
      label: "ACE planner logits",
      size: outputBytes,
      usage: usage.STORAGE | usage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      label: "ACE planner logits readback",
      size: outputBytes,
      usage: usage.COPY_DST | usage.MAP_READ,
    });
    const dispatchX = Math.min(
      batchSize * PLANNER_Q8_HEAD_VOCAB_SIZE,
      65_535,
    );
    const paramsData = new Uint32Array([
      PLANNER_Q8_HEAD_VOCAB_SIZE,
      PLANNER_Q8_HEAD_VOCAB_SIZE,
      PLANNER_Q8_HEAD_HIDDEN_SIZE,
      PLANNER_Q8_HEAD_HIDDEN_SIZE / PLANNER_Q8_HEAD_BLOCK_SIZE,
      dispatchX,
      batchSize,
      0,
      0,
    ]);
    const params = this.device.createBuffer({
      label: "ACE planner head parameters",
      size: paramsData.byteLength,
      usage: usage.UNIFORM | usage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      hidden,
      0,
      finalRows.buffer,
      finalRows.byteOffset,
      finalRows.byteLength,
    );
    this.device.queue.writeBuffer(
      params,
      0,
      paramsData.buffer,
      paramsData.byteOffset,
      paramsData.byteLength,
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weights } },
        { binding: 1, resource: { buffer: this.scales } },
        { binding: 2, resource: { buffer: hidden } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      dispatchX,
      Math.ceil(
        (batchSize * PLANNER_Q8_HEAD_VOCAB_SIZE) / dispatchX,
      ),
    );
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(map.READ);
      const logits = new Float32Array(
        readback.getMappedRange().slice(0),
      );
      readback.unmap();
      return new Tensor("float32", logits, [
        batchSize,
        1,
        PLANNER_Q8_HEAD_VOCAB_SIZE,
      ]);
    } finally {
      hidden.destroy();
      output.destroy();
      readback.destroy();
      params.destroy();
    }
  }

  dispose() {
    this.weights.destroy();
    this.scales.destroy();
  }
}
