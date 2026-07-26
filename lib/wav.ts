const writeAscii = (view: DataView, offset: number, value: string) => {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
};

export function encodeStereoWav(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number,
) {
  if (left.length !== right.length) {
    throw new RangeError("Stereo channel lengths must match.");
  }
  const channels = 2;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const dataBytes = left.length * channels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, bitsPerSample, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let index = 0; index < left.length; index += 1) {
    const leftSample = Math.max(-1, Math.min(1, left[index]));
    const rightSample = Math.max(-1, Math.min(1, right[index]));
    view.setInt16(
      offset,
      leftSample < 0 ? leftSample * 0x8000 : leftSample * 0x7fff,
      true,
    );
    view.setInt16(
      offset + 2,
      rightSample < 0 ? rightSample * 0x8000 : rightSample * 0x7fff,
      true,
    );
    offset += 4;
  }

  return buffer;
}
