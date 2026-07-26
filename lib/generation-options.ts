export const SAMPLER_MODES = [
  "euler",
  "heun",
  "euler-sde",
] as const;

export type SamplerMode = (typeof SAMPLER_MODES)[number];

export const DCW_MODES = ["low", "high", "double", "pix"] as const;

export type DcwMode = (typeof DCW_MODES)[number];

export type DcwOptions = {
  /** Opt-in to preserve the package's pre-DCW deterministic default. */
  enabled?: boolean;
  /** Haar low band, high band, both bands, or direct latent correction. */
  mode?: DcwMode;
  /** Low-band strength, or the single strength for high/pix modes. */
  scaler?: number;
  /** Independent high-band strength used by double mode. */
  highScaler?: number;
};

export type ResolvedDcwOptions = {
  enabled: boolean;
  mode: DcwMode;
  scaler: number;
  highScaler: number;
};

export const DEFAULT_DCW_OPTIONS: Readonly<ResolvedDcwOptions> = {
  enabled: false,
  mode: "double",
  scaler: 0.05,
  highScaler: 0.02,
};

export const resolveDcwOptions = (
  options: DcwOptions = {},
): ResolvedDcwOptions => {
  const resolved = {
    enabled: options.enabled ?? DEFAULT_DCW_OPTIONS.enabled,
    mode: options.mode ?? DEFAULT_DCW_OPTIONS.mode,
    scaler: options.scaler ?? DEFAULT_DCW_OPTIONS.scaler,
    highScaler:
      options.highScaler ?? DEFAULT_DCW_OPTIONS.highScaler,
  };
  if (!DCW_MODES.includes(resolved.mode)) {
    throw new RangeError(`Unsupported DCW mode: ${resolved.mode}`);
  }
  for (const [name, value] of [
    ["scaler", resolved.scaler],
    ["highScaler", resolved.highScaler],
  ] as const) {
    if (!Number.isFinite(value) || value < 0 || value > 0.1) {
      throw new RangeError(`DCW ${name} must be from 0 through 0.1.`);
    }
  }
  return resolved;
};

export const validateSamplerMode = (
  value: string,
): SamplerMode => {
  if (!SAMPLER_MODES.includes(value as SamplerMode)) {
    throw new RangeError(`Unsupported sampler mode: ${value}`);
  }
  return value as SamplerMode;
};
