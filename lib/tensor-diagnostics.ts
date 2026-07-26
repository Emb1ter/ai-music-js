export type TensorSummary = {
  name: string;
  dims: readonly number[];
  count: number;
  min: number;
  max: number;
  mean: number;
  rms: number;
  checksum: string;
  probes: Array<{ index: number; value: number }>;
};

const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

export function tensorSummary(
  name: string,
  dims: readonly number[],
  data: ArrayLike<number>,
  probeCount = 16,
): TensorSummary {
  if (data.length === 0) {
    return {
      name,
      dims,
      count: 0,
      min: 0,
      max: 0,
      mean: 0,
      rms: 0,
      checksum: "00000000",
      probes: [],
    };
  }

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let sumSquares = 0;
  let hash = FNV_OFFSET;
  const view = new DataView(new ArrayBuffer(4));

  for (let index = 0; index < data.length; index += 1) {
    const value = Number(data[index]);
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    sumSquares += value * value;
    if (index % Math.max(1, Math.floor(data.length / 4096)) === 0) {
      view.setFloat32(0, value, true);
      hash ^= view.getUint32(0, true);
      hash = Math.imul(hash, FNV_PRIME) >>> 0;
    }
  }

  const probes = Array.from(
    { length: Math.min(probeCount, data.length) },
    (_, probeIndex) => {
      const index =
        probeCount === 1
          ? 0
          : Math.round(
              (probeIndex * (data.length - 1)) /
                (Math.min(probeCount, data.length) - 1),
            );
      return { index, value: Number(data[index]) };
    },
  );

  return {
    name,
    dims,
    count: data.length,
    min,
    max,
    mean: sum / data.length,
    rms: Math.sqrt(sumSquares / data.length),
    checksum: hash.toString(16).padStart(8, "0"),
    probes,
  };
}

export function assertShape(
  actual: readonly number[],
  expected: readonly number[],
  name: string,
) {
  if (
    actual.length !== expected.length ||
    actual.some((dimension, index) => dimension !== expected[index])
  ) {
    throw new Error(
      `${name} shape mismatch: expected [${expected.join(", ")}], received [${actual.join(", ")}].`,
    );
  }
}
