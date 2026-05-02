export class AudioRingBuffer {
  private buffer: Float32Array;
  private targetSr: number;
  private targetSamples: number;

  constructor(targetSr: number, targetSamples: number) {
    this.targetSr = targetSr;
    this.targetSamples = targetSamples;
    this.buffer = new Float32Array(targetSamples);
  }

  /** 线性插值重采样：足够用于流派分类的频谱包络 */
  private linearResample(input: Float32Array, origSr: number): Float32Array {
    if (origSr === this.targetSr) return input;
    const ratio = origSr / this.targetSr;
    const outLen = Math.ceil(input.length / ratio);
    const output = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const idx = i * ratio;
      const f = Math.floor(idx);
      const c = Math.min(f + 1, input.length - 1);
      const frac = idx - f;
      output[i] = input[f] * (1 - frac) + input[c] * frac;
    }
    return output;
  }

  append(chunk: Float32Array, chunkSr: number) {
    const resampled = this.linearResample(chunk, chunkSr);
    const newLen = this.buffer.length + resampled.length;
    const concatenated = new Float32Array(newLen);
    concatenated.set(this.buffer);
    concatenated.set(resampled, this.buffer.length);

    if (newLen > this.targetSamples) {
      this.buffer = concatenated.slice(newLen - this.targetSamples);
    } else {
      this.buffer = new Float32Array(this.targetSamples);
      this.buffer.set(concatenated, this.targetSamples - newLen);
    }
  }

  getSnapshot(): Float32Array {
    return this.buffer.slice();
  }

  clear() {
    this.buffer.fill(0);
  }
}