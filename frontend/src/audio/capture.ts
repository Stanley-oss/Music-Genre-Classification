// File: ./frontend/src/audio/capture.ts
export interface AudioCapture {
  start(...args: any[]): Promise<void>;
  stop(): void;
  seek?(percentage: number): void;
  readonly isActive: boolean;
}

export class MicrophoneCapture implements AudioCapture {
  private stream: MediaStream | null = null;
  private audioCtx: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private _isActive = false;

  get isActive() { return this._isActive; }

  async start(callback: (chunk: Float32Array, sr: number) => void) {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    }});

    this.audioCtx = new AudioContext();
    const sr = this.audioCtx.sampleRate;

    const source = this.audioCtx.createMediaStreamSource(this.stream);
    this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (e) => {
      callback(new Float32Array(e.inputBuffer.getChannelData(0)), sr);
    };

    source.connect(this.processor);
    this.processor.connect(this.audioCtx.destination);
    this._isActive = true;
  }

  stop() {
    this._isActive = false;
    this.processor?.disconnect();
    this.stream?.getTracks().forEach(t => t.stop());
    this.audioCtx?.close();
  }
}

export class FileCapture implements AudioCapture {
  private audioCtx: AudioContext | null = null;
  private source: AudioBufferSourceNode | null = null;
  private buffer: AudioBuffer | null = null;
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
    onDecoded?: (fullWaveform: Float32Array) => void,
    startTimeSec: number = 0 // 新增：支持传入起始时间
  ) {
    const arrayBuffer = await file.arrayBuffer();
    this.audioCtx = new AudioContext();
    this.buffer = await this.audioCtx.decodeAudioData(arrayBuffer);

    const len = this.buffer.length;
    this.mono = new Float32Array(len);
    for (let c = 0; c < this.buffer.numberOfChannels; c++) {
      const ch = this.buffer.getChannelData(c);
      for (let i = 0; i < len; i++) this.mono[i] += ch[i];
    }
    const inv = 1.0 / this.buffer.numberOfChannels;
    for (let i = 0; i < len; i++) this.mono[i] *= inv;

    if (onDecoded) onDecoded(this.mono);

    this.duration = this.buffer.duration;
    this.sr = this.buffer.sampleRate;
    this.dataCallback = callback;

    this._isActive = true;
    this.playFrom(startTimeSec);

    this.interval = window.setInterval(() => {
      const currTime = this.currentTime;
      const targetOffset = Math.floor(currTime * this.sr);
      if (targetOffset >= this.mono!.length) {
        this.stop();
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
    this.lastProcessedOffset = Math.floor(timeSec * this.sr);
  }

  seek(percentage: number) {
    if (!this._isActive || !this.buffer) return;
    const targetSec = Math.max(0, Math.min(percentage * this.duration, this.duration));
    this.playFrom(targetSec);
  }

  stop() {
    this._isActive = false;
    if (this.interval) clearInterval(this.interval);
    try { this.source?.stop(); } catch(e) {}
    this.source?.disconnect();
    this.audioCtx?.close();
  }
}