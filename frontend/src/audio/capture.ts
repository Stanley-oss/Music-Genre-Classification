export interface AudioCapture {
  start(...args: any[]): Promise<void>;
  stop(): void;
  pause?(): void;
  resume?(): void;
  seek?(percentage: number): void;
  readonly isActive: boolean;
  analyser?: AnalyserNode | null;
}

export class MicrophoneCapture implements AudioCapture {
  private stream: MediaStream | null = null;
  private audioCtx: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  public analyser: AnalyserNode | null = null;
  private _isActive = false;

  get isActive() { return this._isActive; }

  async start(callback: (chunk: Float32Array, sr: number) => void) {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    }});

    this.audioCtx = new AudioContext();
    const sr = this.audioCtx.sampleRate;
    const source = this.audioCtx.createMediaStreamSource(this.stream);

    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.8;
    source.connect(this.analyser);

    this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (e) => {
      callback(new Float32Array(e.inputBuffer.getChannelData(0)), sr);
    };
    this.analyser.connect(this.processor);

    const silentGain = this.audioCtx.createGain();
    silentGain.gain.value = 0;
    this.processor.connect(silentGain);
    silentGain.connect(this.audioCtx.destination);

    this._isActive = true;
  }

  stop() {
    this._isActive = false;
    this.processor?.disconnect();
    this.stream?.getTracks().forEach(t => t.stop());
    this.audioCtx?.close();
  }
}

/** 文件流式播放：修复 Seek 缓冲区噪音，提速混音 */
export class FileCapture implements AudioCapture {
  private audioCtx: AudioContext | null = null;
  private source: AudioBufferSourceNode | null = null;
  public buffer: AudioBuffer | null = null;
  private interval: number | null = null;
  private _isActive = false;

  private startTime = 0;
  private startOffsetSec = 0;
  private lastProcessedOffset = 0;

  public duration = 0;
  private mono: Float32Array | null = null;
  private sr = 22050;
  private dataCallback: ((chunk: Float32Array, sr: number) => void) | null = null;

  get isActive() { return this._isActive; }

  get currentTime() {
    if (!this.audioCtx || !this._isActive) return this.startOffsetSec;
    return this.startOffsetSec + (this.audioCtx.currentTime - this.startTime);
  }

  async start(
    file: File, 
    callback: (chunk: Float32Array, sr: number) => void,
    onDecoded?: (fullWaveform: Float32Array, duration: number) => void
  ) {
    const arrayBuffer = await file.arrayBuffer();
    this.audioCtx = new AudioContext();
    this.buffer = await this.audioCtx.decodeAudioData(arrayBuffer);

    // 优化：针对常见立体声进行高速降维混音，大幅降低 JS 循环耗时
    const len = this.buffer.length;
    const channels = this.buffer.numberOfChannels;
    this.mono = new Float32Array(len);
    
    if (channels === 1) {
      this.mono.set(this.buffer.getChannelData(0));
    } else if (channels === 2) {
      const ch0 = this.buffer.getChannelData(0);
      const ch1 = this.buffer.getChannelData(1);
      for (let i = 0; i < len; i++) {
        this.mono[i] = (ch0[i] + ch1[i]) * 0.5;
      }
    } else {
      for (let c = 0; c < channels; c++) {
        const ch = this.buffer.getChannelData(c);
        for (let i = 0; i < len; i++) this.mono[i] += ch[i];
      }
      const inv = 1.0 / channels;
      for (let i = 0; i < len; i++) this.mono[i] *= inv;
    }

    this.duration = this.buffer.duration;
    this.sr = this.buffer.sampleRate;
    this.dataCallback = callback;

    if (onDecoded) onDecoded(this.mono, this.duration);

    this._isActive = true;
    this.playFrom(0);
    this.startInterval();
  }

  private startInterval() {
    if (this.interval) clearInterval(this.interval);
    this.interval = window.setInterval(() => {
      const currTime = this.currentTime;
      const targetOffset = Math.floor(currTime * this.sr);
      if (targetOffset >= this.mono!.length) {
        this.pause();
        return;
      }
      if (targetOffset > this.lastProcessedOffset) {
        this.dataCallback!(this.mono!.slice(this.lastProcessedOffset, targetOffset), this.sr);
        this.lastProcessedOffset = targetOffset;
      }
    }, 100);
  }

  private playFrom(timeSec: number) {
    if (this.source) {
      try { this.source.stop(); } catch(e) {}
      this.source.disconnect();
    }
    this.source = this.audioCtx!.createBufferSource();
    this.source.buffer = this.buffer;
    this.source.connect(this.audioCtx!.destination);
    this.source.start(0, timeSec);

    this.startTime = this.audioCtx!.currentTime;
    this.startOffsetSec = timeSec;
    
    // 核心修复：预加载前 2.3 秒（约 50688 个采样点 22050Hz）的数据送入队列
    // 彻底解决寻址后由于 RingBuffer 缺水补 0 导致的白噪声被判定为 Metal/Rock 的 Bug！
    const preloadSec = 2.3; 
    this.lastProcessedOffset = Math.max(0, Math.floor((timeSec - preloadSec) * this.sr));
  }

  seek(percentage: number) {
    if (!this.buffer) return;
    const targetSec = Math.max(0, Math.min(percentage * this.duration, this.duration));
    this.startOffsetSec = targetSec;
    if (this._isActive) {
      this.playFrom(targetSec);
    } else {
      const preloadSec = 2.3; 
      this.lastProcessedOffset = Math.max(0, Math.floor((targetSec - preloadSec) * this.sr));
    }
  }

  pause() {
    this._isActive = false;
    if (this.interval) clearInterval(this.interval);
    try { this.source?.stop(); } catch(e) {}
    this.source?.disconnect();
    this.source = null;
    if (this.audioCtx) {
      this.startOffsetSec += (this.audioCtx.currentTime - this.startTime);
    }
  }

  resume() {
    if (!this.buffer || !this.audioCtx) return;
    if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
    this._isActive = true;
    this.playFrom(this.startOffsetSec);
    this.startInterval();
  }

  stop() {
    this.pause();
    this.audioCtx?.close();
    this.buffer = null;
    this.mono = null;
    this.startOffsetSec = 0;
  }
}