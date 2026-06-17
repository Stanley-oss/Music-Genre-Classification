export class AudioRingBuffer {
  private buffer: Float32Array;
  private targetSr: number;
  private targetSamples: number;
  private writeHead: number = 0;
  private filled: number = 0;
  private rsBuf: Float32Array | null = null; // Pooled resample buffer (grows on demand)

  constructor(targetSr: number, targetSamples: number) {
    this.targetSr = targetSr;
    this.targetSamples = targetSamples;
    this.buffer = new Float32Array(targetSamples);
  }

  /**
   * 线性插值重采样：足够用于流派分类的频谱包络。
   * Uses a pooled buffer to avoid per-call allocations.
   */
  private linearResample(input: Float32Array, origSr: number): { data: Float32Array; length: number } {
    if (origSr === this.targetSr) return { data: input, length: input.length };

    const ratio = origSr / this.targetSr;
    const outLen = Math.ceil(input.length / ratio);

    // Grow pooled resample buffer on demand (amortized O(1) allocation)
    if (!this.rsBuf || this.rsBuf.length < outLen) {
      this.rsBuf = new Float32Array(Math.max(outLen, 8192));
    }

    for (let i = 0; i < outLen; i++) {
      const idx = i * ratio;
      const f = Math.floor(idx);
      const c = Math.min(f + 1, input.length - 1);
      const frac = idx - f;
      this.rsBuf[i] = input[f] * (1 - frac) + input[c] * frac;
    }

    return { data: this.rsBuf, length: outLen };
  }

  /** Write resampled data directly into the circular buffer (zero allocation). */
  private writeChunk(data: Float32Array, len: number) {
    if (len >= this.targetSamples) {
      // Chunk larger than buffer: keep only the newest targetSamples
      this.buffer.set(data.subarray(len - this.targetSamples, len));
      this.writeHead = 0;
      this.filled = this.targetSamples;
      return;
    }

    const spaceToEnd = this.targetSamples - this.writeHead;
    if (len <= spaceToEnd) {
      // No wrap-around needed
      this.buffer.set(data.subarray(0, len), this.writeHead);
    } else {
      // Wrap around the end of the buffer
      this.buffer.set(data.subarray(0, spaceToEnd), this.writeHead);
      this.buffer.set(data.subarray(spaceToEnd, len), 0);
    }

    this.writeHead = (this.writeHead + len) % this.targetSamples;
    this.filled = Math.min(this.filled + len, this.targetSamples);
  }

  append(chunk: Float32Array, chunkSr: number) {
    const { data, length } = this.linearResample(chunk, chunkSr);
    this.writeChunk(data, length);
  }

  /**
   * Return a stable, chronologically-ordered copy of the buffer.
   * Zeros are left-padded when the buffer is not yet full, matching the
   * original concat-and-slice behaviour.
   */
  getSnapshot(): Float32Array {
    const out = new Float32Array(this.targetSamples);
    if (this.filled === 0) return out;

    if (this.filled < this.targetSamples) {
      // Not yet full: data sits at [0..filled), pad zeros on left
      out.set(this.buffer.subarray(0, this.filled), this.targetSamples - this.filled);
    } else {
      // Full: read chronologically from writeHead (oldest) → end, then 0 → writeHead (newest)
      const tailLen = this.targetSamples - this.writeHead;
      out.set(this.buffer.subarray(this.writeHead, this.writeHead + tailLen), 0);
      if (this.writeHead > 0) {
        out.set(this.buffer.subarray(0, this.writeHead), tailLen);
      }
    }

    return out;
  }

  clear() {
    this.buffer.fill(0);
    this.writeHead = 0;
    this.filled = 0;
  }
}