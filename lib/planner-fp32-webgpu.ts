import * as ort from "onnxruntime-web/webgpu";
import type { PlannerEmbeddingStats } from "./planner-profile";

export const FP32_PLANNER_VOCAB_SIZE = 217_204;
export const FP32_PLANNER_HIDDEN_SIZE = 2_560;
export const FP32_AUDIO_CODE_TOKEN_START = 151_669;
export const FP32_AUDIO_CODE_COUNT = 64_000;
export const FP32_HEAD_SHARD_ROWS = 8_192;

export const FP32_PLANNER_ROW_BYTES =
  FP32_PLANNER_HIDDEN_SIZE * Float32Array.BYTES_PER_ELEMENT;

export type Fp32PlannerEmbeddingRowStore = {
  load(tokenId: number): Promise<Float32Array | undefined>;
  save(tokenId: number, row: Float32Array): Promise<void>;
};

type GpuBuffer = {
  destroy(): void;
  getMappedRange(): ArrayBuffer;
  mapAsync(mode: number): Promise<void>;
  unmap(): void;
};

type ComputePipeline = {
  getBindGroupLayout(index: number): unknown;
};

type GpuDevice = {
  createBindGroup(descriptor: Record<string, unknown>): unknown;
  createBuffer(descriptor: Record<string, unknown>): GpuBuffer;
  createCommandEncoder(): {
    beginComputePass(): {
      dispatchWorkgroups(x: number, y?: number): void;
      end(): void;
      setBindGroup(index: number, group: unknown): void;
      setPipeline(pipeline: unknown): void;
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
  createComputePipeline(descriptor: Record<string, unknown>): ComputePipeline;
  createShaderModule(descriptor: Record<string, unknown>): unknown;
  queue: {
    submit(commands: unknown[]): void;
    writeBuffer(
      buffer: GpuBuffer,
      offset: number,
      data: ArrayBuffer,
      dataOffset?: number,
      size?: number,
    ): void;
  };
};

export type Fp32PlannerAssetLoader = (
  fileName: string,
  expectedBytes: number,
) => Promise<Uint8Array>;

type HeadProgress = (
  fileName: string,
  loaded: number,
  total: number,
) => void;

export type Fp32PlannerTimingEvent =
  | "embedding-range-fetch"
  | "embedding-cache-read"
  | "embedding-cache-write"
  | "embedding-pack"
  | "embedding-total"
  | "head-adapter-device"
  | "head-pipeline-create"
  | "head-weight-asset-load"
  | "head-weight-gpu-upload"
  | "head-hidden-copy"
  | "head-command-encode"
  | "head-gpu-compute-readback"
  | "head-result-copy"
  | "head-embedding-readback"
  | "head-forward-total";

export type Fp32PlannerTimingRecorder = (
  event: Fp32PlannerTimingEvent,
  milliseconds: number,
) => void;

type WeightShard = {
  buffer: GpuBuffer;
  outputStart: number;
  rowCount: number;
};

export const fp32AudioHeadRowLocation = (audioCode: number) => {
  if (
    !Number.isInteger(audioCode) ||
    audioCode < 0 ||
    audioCode >= FP32_AUDIO_CODE_COUNT
  ) {
    throw new RangeError(
      `Planner audio code ${audioCode} is outside 0-${FP32_AUDIO_CODE_COUNT - 1}.`,
    );
  }
  const shardIndex = Math.floor(audioCode / FP32_HEAD_SHARD_ROWS);
  return {
    shardIndex,
    rowInShard: audioCode - shardIndex * FP32_HEAD_SHARD_ROWS,
  };
};

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

const AUDIO_HEAD_SHADER = /* wgsl */ `
struct Params {
  output_start: u32,
  row_count: u32,
  hidden_size: u32,
  output_size: u32,
  dispatch_x: u32,
  batch_size: u32,
  _padding0: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> hidden: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let linear_row = group_id.x + group_id.y * params.dispatch_x;
  let total_rows = params.row_count * params.batch_size;
  if (linear_row >= total_rows) {
    return;
  }
  let batch = linear_row / params.row_count;
  let local_row = linear_row - batch * params.row_count;
  var sum = 0.0;
  var column = local_id.x;
  loop {
    if (column >= params.hidden_size) {
      break;
    }
    sum += hidden[batch * params.hidden_size + column] *
      weights[local_row * params.hidden_size + column];
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
    output[
      batch * params.output_size + params.output_start + local_row
    ] = partial[0];
  }
}
`;

const loadRange = async (
  url: string,
  start: number,
  byteLength: number,
) => {
  const response = await fetch(url, {
    cache: "no-store",
    credentials: "omit",
    headers: {
      Range: `bytes=${start}-${start + byteLength - 1}`,
    },
  });
  if (response.status !== 206) {
    throw new Error(
      `FP32 embedding range returned HTTP ${response.status}; expected 206.`,
    );
  }
  const payload = await response.arrayBuffer();
  if (payload.byteLength !== byteLength) {
    throw new Error(
      `FP32 embedding range returned ${payload.byteLength} bytes; expected ${byteLength}.`,
    );
  }
  return payload;
};

export class Fp32PlannerEmbeddingTable {
  private readonly rows = new Map<number, Float32Array>();
  private readonly pendingRows = new Map<number, Promise<Float32Array>>();
  private readonly statistics: PlannerEmbeddingStats = {
    requestedRows: 0,
    fetchedRows: 0,
    memoryHits: 0,
    persistentHits: 0,
    persistentWrites: 0,
    injectedRows: 0,
    fetchedBytes: 0,
    rangeRequests: 0,
  };

  constructor(
    private readonly weightUrl: string,
    private readonly recordTiming: Fp32PlannerTimingRecorder = () => undefined,
    private readonly persistentStore?: Fp32PlannerEmbeddingRowStore,
  ) {}

  private validateTokenId(tokenId: number) {
    if (
      !Number.isInteger(tokenId) ||
      tokenId < 0 ||
      tokenId >= FP32_PLANNER_VOCAB_SIZE
    ) {
      throw new RangeError(`Planner token ${tokenId} is outside the vocabulary.`);
    }
  }

  private validateRow(tokenId: number, row: Float32Array) {
    if (row.length !== FP32_PLANNER_HIDDEN_SIZE) {
      throw new Error(
        `Planner embedding row ${tokenId} contains ${row.length} values; ` +
          `expected ${FP32_PLANNER_HIDDEN_SIZE}.`,
      );
    }
  }

  private async loadMissingRow(tokenId: number) {
    if (this.persistentStore) {
      const cacheStart = performance.now();
      const stored = await this.persistentStore.load(tokenId);
      this.recordTiming(
        "embedding-cache-read",
        performance.now() - cacheStart,
      );
      if (stored) {
        this.validateRow(tokenId, stored);
        this.statistics.persistentHits += 1;
        this.rows.set(tokenId, stored);
        return stored;
      }
    }

    const fetchStart = performance.now();
    const payload = await loadRange(
      this.weightUrl,
      tokenId * FP32_PLANNER_ROW_BYTES,
      FP32_PLANNER_ROW_BYTES,
    );
    this.recordTiming(
      "embedding-range-fetch",
      performance.now() - fetchStart,
    );
    this.statistics.fetchedRows += 1;
    this.statistics.fetchedBytes += payload.byteLength;
    this.statistics.rangeRequests += 1;
    const row = new Float32Array(payload);
    this.rows.set(tokenId, row);
    if (this.persistentStore) {
      const cacheStart = performance.now();
      await this.persistentStore.save(tokenId, row);
      this.recordTiming(
        "embedding-cache-write",
        performance.now() - cacheStart,
      );
      this.statistics.persistentWrites += 1;
    }
    return row;
  }

  private async loadRow(tokenId: number) {
    this.statistics.requestedRows += 1;
    this.validateTokenId(tokenId);
    const existing = this.rows.get(tokenId);
    if (existing) {
      this.statistics.memoryHits += 1;
      return existing;
    }
    const pending = this.pendingRows.get(tokenId);
    if (pending) return pending;
    const loading = this.loadMissingRow(tokenId);
    this.pendingRows.set(tokenId, loading);
    try {
      return await loading;
    } finally {
      this.pendingRows.delete(tokenId);
    }
  }

  async prefetch(tokenIds: readonly number[]) {
    const requested = [...new Set(tokenIds.map(Number))];
    for (let start = 0; start < requested.length; start += 16) {
      await Promise.all(
        requested
          .slice(start, start + 16)
          .map((tokenId) => this.loadRow(tokenId)),
      );
    }
  }

  setRow(tokenId: number, row: Float32Array) {
    this.validateTokenId(tokenId);
    this.validateRow(tokenId, row);
    if (!this.rows.has(tokenId)) this.statistics.injectedRows += 1;
    this.rows.set(tokenId, row);
  }

  async embed(
    tokenIds: readonly number[],
    dims: readonly [number, number],
  ) {
    const totalStart = performance.now();
    const [batchSize, sequenceLength] = dims;
    if (batchSize * sequenceLength !== tokenIds.length) {
      throw new Error(
        `Embedding input ${JSON.stringify(dims)} contains ${tokenIds.length} values.`,
      );
    }
    await this.prefetch(tokenIds);

    const packStart = performance.now();
    const values = new Float32Array(
      tokenIds.length * FP32_PLANNER_HIDDEN_SIZE,
    );
    for (let index = 0; index < tokenIds.length; index += 1) {
      const row = this.rows.get(Number(tokenIds[index]));
      if (!row) {
        throw new Error(`Planner embedding row ${tokenIds[index]} was not loaded.`);
      }
      values.set(row, index * FP32_PLANNER_HIDDEN_SIZE);
    }
    const tensor = new ort.Tensor("float32", values, [
      batchSize,
      sequenceLength,
      FP32_PLANNER_HIDDEN_SIZE,
    ]);
    this.recordTiming("embedding-pack", performance.now() - packStart);
    this.recordTiming("embedding-total", performance.now() - totalStart);
    return tensor;
  }

  /**
   * The ACE planner ties its input embedding and language-model head. Score a
   * constrained set of normal text tokens without materializing the complete
   * 217,204-row FP32 head in browser memory.
   */
  async scoreTokenIds(
    lastHiddenState: ort.Tensor,
    tokenIds: readonly number[],
  ) {
    const [batchSize = 0, sequenceLength = 0, hiddenSize = 0] =
      lastHiddenState.dims;
    if (
      batchSize !== 1 ||
      sequenceLength < 1 ||
      hiddenSize !== FP32_PLANNER_HIDDEN_SIZE ||
      lastHiddenState.type !== "float32"
    ) {
      throw new Error(
        `Sparse planner head requires one FP32 hidden row, received ` +
          `${lastHiddenState.type} ${JSON.stringify(lastHiddenState.dims)}.`,
      );
    }
    const uniqueTokenIds = [...new Set(tokenIds.map(Number))];
    await Promise.all(
      uniqueTokenIds.map((tokenId) => this.loadRow(tokenId)),
    );
    const hidden = lastHiddenState.data as Float32Array;
    const hiddenOffset =
      (sequenceLength - 1) * FP32_PLANNER_HIDDEN_SIZE;
    const scores = new Float32Array(tokenIds.length);
    for (let tokenIndex = 0; tokenIndex < tokenIds.length; tokenIndex += 1) {
      const tokenId = Number(tokenIds[tokenIndex]);
      const row = this.rows.get(tokenId);
      if (!row) {
        throw new Error(
          `Planner output row ${tokenId} was not loaded.`,
        );
      }
      let score = 0;
      for (
        let column = 0;
        column < FP32_PLANNER_HIDDEN_SIZE;
        column += 1
      ) {
        score += hidden[hiddenOffset + column]! * row[column]!;
      }
      scores[tokenIndex] = score;
    }
    return scores;
  }

  clear() {
    this.rows.clear();
    this.pendingRows.clear();
  }

  stats(): PlannerEmbeddingStats {
    return { ...this.statistics };
  }
}

export class Fp32PlannerAudioCodeHead {
  private constructor(
    private readonly device: GpuDevice,
    private readonly pipeline: ComputePipeline,
    private readonly shards: WeightShard[],
    private readonly recordTiming: Fp32PlannerTimingRecorder,
  ) {}

  static async create(
    files: readonly {
      fileName: string;
      bytes: number;
      rowCount: number;
    }[],
    load: Fp32PlannerAssetLoader,
    progress: HeadProgress,
    recordTiming: Fp32PlannerTimingRecorder = () => undefined,
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
    const adapterStart = performance.now();
    const adapter = await navigatorWithGpu.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("FP32 audio-code head could not acquire a WebGPU adapter.");
    }
    const largestShardBytes = Math.max(...files.map((file) => file.bytes));
    if (
      Number(adapter.limits.maxBufferSize) < largestShardBytes ||
      Number(adapter.limits.maxStorageBufferBindingSize) < largestShardBytes
    ) {
      throw new Error(
        `FP32 audio-code head requires an ${largestShardBytes}-byte storage binding; ` +
          `the adapter reports ${Number(adapter.limits.maxStorageBufferBindingSize)} bytes.`,
      );
    }
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: largestShardBytes,
        maxStorageBufferBindingSize: largestShardBytes,
      },
    });
    recordTiming("head-adapter-device", performance.now() - adapterStart);
    const pipelineStart = performance.now();
    const pipeline = device.createComputePipeline({
      label: "ACE planner sharded FP32 audio-code head",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "ACE planner FP32 audio-code head shader",
          code: AUDIO_HEAD_SHADER,
        }),
        entryPoint: "main",
      },
    });
    recordTiming(
      "head-pipeline-create",
      performance.now() - pipelineStart,
    );

    const { buffer: usage } = gpuFlags();
    const totalBytes = files.reduce((sum, file) => sum + file.bytes, 0);
    const shards: WeightShard[] = [];
    let loadedBytes = 0;
    let outputStart = 0;
    try {
      for (const file of files) {
        const loadStart = performance.now();
        const payload = await load(file.fileName, file.bytes);
        recordTiming(
          "head-weight-asset-load",
          performance.now() - loadStart,
        );
        const uploadStart = performance.now();
        const buffer = device.createBuffer({
          label:
            `ACE planner FP32 audio rows ${outputStart}-` +
            `${outputStart + file.rowCount - 1}`,
          mappedAtCreation: true,
          size: file.bytes,
          usage: usage.STORAGE | usage.COPY_SRC,
        });
        new Uint8Array(buffer.getMappedRange()).set(payload);
        buffer.unmap();
        recordTiming(
          "head-weight-gpu-upload",
          performance.now() - uploadStart,
        );
        shards.push({
          buffer,
          outputStart,
          rowCount: file.rowCount,
        });
        loadedBytes += file.bytes;
        progress(file.fileName, loadedBytes, totalBytes);
        outputStart += file.rowCount;
      }
    } catch (error) {
      for (const shard of shards) shard.buffer.destroy();
      throw error;
    }
    if (outputStart !== FP32_AUDIO_CODE_COUNT) {
      for (const shard of shards) shard.buffer.destroy();
      throw new Error(
        `FP32 audio head contains ${outputStart} rows; expected ${FP32_AUDIO_CODE_COUNT}.`,
      );
    }
    return new Fp32PlannerAudioCodeHead(
      device,
      pipeline,
      shards,
      recordTiming,
    );
  }

  async forward(lastHiddenState: ort.Tensor) {
    const totalStart = performance.now();
    const [batchSize = 0, sequenceLength = 0, hiddenSize = 0] =
      lastHiddenState.dims;
    if (
      batchSize < 1 ||
      sequenceLength < 1 ||
      hiddenSize !== FP32_PLANNER_HIDDEN_SIZE ||
      lastHiddenState.type !== "float32"
    ) {
      throw new Error(
        `FP32 planner body returned ${lastHiddenState.type} ` +
          `${JSON.stringify(lastHiddenState.dims)}.`,
      );
    }
    const hiddenCopyStart = performance.now();
    const source = lastHiddenState.data as Float32Array;
    const finalRows = new Float32Array(
      batchSize * FP32_PLANNER_HIDDEN_SIZE,
    );
    for (let batch = 0; batch < batchSize; batch += 1) {
      const start =
        (batch * sequenceLength + sequenceLength - 1) *
        FP32_PLANNER_HIDDEN_SIZE;
      finalRows.set(
        source.subarray(start, start + FP32_PLANNER_HIDDEN_SIZE),
        batch * FP32_PLANNER_HIDDEN_SIZE,
      );
    }
    this.recordTiming(
      "head-hidden-copy",
      performance.now() - hiddenCopyStart,
    );

    const commandStart = performance.now();
    const { buffer: usage, map } = gpuFlags();
    const hidden = this.device.createBuffer({
      label: "ACE planner FP32 final hidden states",
      size: finalRows.byteLength,
      usage: usage.STORAGE | usage.COPY_DST,
    });
    const outputBytes =
      batchSize *
      FP32_AUDIO_CODE_COUNT *
      Float32Array.BYTES_PER_ELEMENT;
    const output = this.device.createBuffer({
      label: "ACE planner FP32 audio logits",
      size: outputBytes,
      usage: usage.STORAGE | usage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      label: "ACE planner FP32 audio logits readback",
      size: outputBytes,
      usage: usage.COPY_DST | usage.MAP_READ,
    });
    this.device.queue.writeBuffer(
      hidden,
      0,
      finalRows.buffer,
      finalRows.byteOffset,
      finalRows.byteLength,
    );

    const parameterBuffers: GpuBuffer[] = [];
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    try {
      for (const shard of this.shards) {
        const totalRows = batchSize * shard.rowCount;
        const dispatchX = Math.min(totalRows, 65_535);
        const parameters = new Uint32Array([
          shard.outputStart,
          shard.rowCount,
          FP32_PLANNER_HIDDEN_SIZE,
          FP32_AUDIO_CODE_COUNT,
          dispatchX,
          batchSize,
          0,
          0,
        ]);
        const params = this.device.createBuffer({
          label: "ACE planner FP32 head parameters",
          size: parameters.byteLength,
          usage: usage.UNIFORM | usage.COPY_DST,
        });
        parameterBuffers.push(params);
        this.device.queue.writeBuffer(
          params,
          0,
          parameters.buffer,
          parameters.byteOffset,
          parameters.byteLength,
        );
        const bindGroup = this.device.createBindGroup({
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: shard.buffer } },
            { binding: 1, resource: { buffer: hidden } },
            { binding: 2, resource: { buffer: output } },
            { binding: 3, resource: { buffer: params } },
          ],
        });
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
          dispatchX,
          Math.ceil(totalRows / dispatchX),
        );
      }
      pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
      this.recordTiming(
        "head-command-encode",
        performance.now() - commandStart,
      );
      const gpuStart = performance.now();
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(map.READ);
      this.recordTiming(
        "head-gpu-compute-readback",
        performance.now() - gpuStart,
      );
      const resultStart = performance.now();
      const values = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      this.recordTiming(
        "head-result-copy",
        performance.now() - resultStart,
      );
      this.recordTiming(
        "head-forward-total",
        performance.now() - totalStart,
      );
      return values;
    } finally {
      hidden.destroy();
      output.destroy();
      readback.destroy();
      for (const params of parameterBuffers) params.destroy();
    }
  }

  /**
   * The planner ties input embeddings to the output head. The selected audio
   * code's exact FP32 embedding is already resident in these head shards, so
   * copy that 10 KiB row back instead of issuing another HTTP range request.
   */
  async readEmbeddingRow(audioCode: number) {
    const { shardIndex, rowInShard } =
      fp32AudioHeadRowLocation(audioCode);
    const shard = this.shards[shardIndex];
    if (!shard || rowInShard >= shard.rowCount) {
      throw new Error(`No FP32 audio-head shard contains code ${audioCode}.`);
    }

    const { buffer: usage, map } = gpuFlags();
    const readback = this.device.createBuffer({
      label: `ACE planner embedding row for audio code ${audioCode}`,
      size: FP32_PLANNER_ROW_BYTES,
      usage: usage.COPY_DST | usage.MAP_READ,
    });
    const start = performance.now();
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(
        shard.buffer,
        rowInShard * FP32_PLANNER_ROW_BYTES,
        readback,
        0,
        FP32_PLANNER_ROW_BYTES,
      );
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(map.READ);
      const row = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      this.recordTiming(
        "head-embedding-readback",
        performance.now() - start,
      );
      return row;
    } finally {
      readback.destroy();
    }
  }

  dispose() {
    for (const shard of this.shards) shard.buffer.destroy();
  }
}
