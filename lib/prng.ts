const UINT32_SCALE = 4_294_967_296;
const FALLBACK_SEED = 0x6d2b79f5;

export class XorShift32 {
  private state: number;
  private spareNormal: number | null = null;

  constructor(seed: number) {
    const normalized = Number.isFinite(seed) ? Math.trunc(seed) >>> 0 : 0;
    this.state = normalized || FALLBACK_SEED;
  }

  nextUint32() {
    let value = this.state;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.state = value >>> 0;
    return this.state;
  }

  uniformOpen() {
    return (this.nextUint32() + 0.5) / UINT32_SCALE;
  }

  normal() {
    if (this.spareNormal !== null) {
      const result = this.spareNormal;
      this.spareNormal = null;
      return Math.fround(result);
    }

    const radius = Math.sqrt(-2 * Math.log(this.uniformOpen()));
    const angle = 2 * Math.PI * this.uniformOpen();
    this.spareNormal = radius * Math.sin(angle);
    return Math.fround(radius * Math.cos(angle));
  }
}

export function deterministicNormal(
  length: number,
  seed: number,
): Float32Array {
  if (!Number.isSafeInteger(length) || length < 0) {
    throw new RangeError("Tensor length must be a non-negative safe integer.");
  }
  const rng = new XorShift32(seed);
  const values = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    values[index] = rng.normal();
  }
  return values;
}
