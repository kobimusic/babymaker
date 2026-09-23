/**
 * Minimal RIFF/WAVE reader: PCM 16/24/32-bit and IEEE float32, any channel
 * count (mixed down to mono). Written rather than decodeAudioData because
 * the morph needs the samples exactly as stored, at the file's own rate —
 * the browser would resample them to the AudioContext's rate.
 */
export interface Wav { sampleRate: number; samples: Float64Array }

export function decodeWav(buf: ArrayBuffer): Wav {
  const v = new DataView(buf);
  const tag = (o: number) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("not a WAV file");

  let fmt = 0, channels = 1, sampleRate = 44100, bits = 16;
  let dataOff = -1, dataLen = 0;
  for (let o = 12; o + 8 <= buf.byteLength; ) {
    const id = tag(o);
    const len = v.getUint32(o + 4, true);
    if (id === "fmt ") {
      fmt = v.getUint16(o + 8, true);
      channels = v.getUint16(o + 10, true);
      sampleRate = v.getUint32(o + 12, true);
      bits = v.getUint16(o + 22, true);
      if (fmt === 0xfffe) fmt = v.getUint16(o + 32, true); // WAVE_FORMAT_EXTENSIBLE: sub-format
    } else if (id === "data") {
      dataOff = o + 8;
      dataLen = Math.min(len, buf.byteLength - dataOff);
    }
    o += 8 + len + (len & 1);
  }
  if (dataOff < 0) throw new Error("WAV has no data chunk");

  const bytes = bits >> 3;
  const frames = Math.floor(dataLen / (bytes * channels));
  const samples = new Float64Array(frames);
  const read = (o: number): number => {
    if (fmt === 3) return v.getFloat32(o, true);
    if (bits === 16) return v.getInt16(o, true) / 32768;
    if (bits === 24) return ((v.getUint8(o) | (v.getUint8(o + 1) << 8) | (v.getInt8(o + 2) << 16)) / 8388608);
    if (bits === 32) return v.getInt32(o, true) / 2147483648;
    throw new Error(`unsupported WAV: format ${fmt}, ${bits} bits`);
  };
  for (let i = 0; i < frames; i++) {
    let s = 0;
    for (let c = 0; c < channels; c++) s += read(dataOff + (i * channels + c) * bytes);
    samples[i] = s / channels;
  }
  return { sampleRate, samples };
}
