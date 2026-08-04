export const PLANNER_CACHE_MIN_WRITE_HEADROOM_BYTES = 128_000_000;

export type PlannerCacheDecision = {
  cache: boolean;
  availableBytes: number;
  missingAudioBytes: number;
  assetBytes: number;
  writeHeadroomBytes: number;
  requiredBytes: number;
  remainingAfterWriteBytes: number;
};

/**
 * Keep enough origin quota for every audio asset that is still missing while
 * deciding whether one planner file can be persisted. Cache API writes may
 * temporarily retain both the response being consumed and the stored entry,
 * so large planner files use their own size as transaction headroom.
 */
export const decidePlannerAssetCache = ({
  availableBytes,
  missingAudioBytes,
  assetBytes,
  minimumWriteHeadroomBytes = PLANNER_CACHE_MIN_WRITE_HEADROOM_BYTES,
}: {
  availableBytes: number;
  missingAudioBytes: number;
  assetBytes: number;
  minimumWriteHeadroomBytes?: number;
}): PlannerCacheDecision => {
  for (const [name, value] of Object.entries({
    availableBytes,
    missingAudioBytes,
    assetBytes,
    minimumWriteHeadroomBytes,
  })) {
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError(`${name} must be a finite non-negative number.`);
    }
  }
  const writeHeadroomBytes = Math.max(
    minimumWriteHeadroomBytes,
    assetBytes,
  );
  const requiredBytes =
    missingAudioBytes + assetBytes + writeHeadroomBytes;
  return {
    cache: availableBytes >= requiredBytes,
    availableBytes,
    missingAudioBytes,
    assetBytes,
    writeHeadroomBytes,
    requiredBytes,
    remainingAfterWriteBytes: Math.max(
      0,
      availableBytes - assetBytes,
    ),
  };
};

export const missingAssetBytes = (
  assets: readonly { bytes: number; cached: boolean }[],
) =>
  assets.reduce(
    (total, asset) => total + (asset.cached ? 0 : asset.bytes),
    0,
  );
