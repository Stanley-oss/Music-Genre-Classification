// File: ./frontend/src/engine/backend-engine.ts
import { InferenceEngine } from './interface';

export class BackendEngine extends InferenceEngine {
  private url: string;
  private ws: WebSocket | null;
  private reqId: number;
  private pending: Map<number, { resolve: Function, reject: Function }>;
  private streamStarted: boolean; // 新增：流状态标识

  constructor() {
    super();
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = (import.meta as any).env?.DEV ? 'localhost:8000' : window.location.host;
    this.url = `${protocol}//${host}/ws/inference`;
    this.ws = null;
    this.reqId = 0;
    this.pending = new Map();
    this.streamStarted = false;
  }

  get name() { return 'Server GPU — ResNet18'; }
  get mode(): 'local' | 'remote' { return 'remote'; }
  get genres() {
    return ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'];
  }
  get sampleRate() { return 22050; }
  get patchSamples() { return 50688; }

  async init(): Promise<void> {
    this.ws = new WebSocket(this.url);
    await new Promise<void>((resolve, reject) => {
      this.ws!.onopen = () => resolve();
      this.ws!.onerror = (e) => reject(e);
    });

    this.ws!.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'patch' && msg.request_id !== undefined) {
        const p = this.pending.get(msg.request_id);
        if (p) {
          this.pending.delete(msg.request_id);
          p.resolve(new Float32Array(msg.probabilities));
        }
      }
    };
  }

  async predict(audioPatch: Float32Array, sr = 22050): Promise<Float32Array> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket disconnected');
    }
    const id = ++this.reqId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });

      // 修复 3：在第一次推流前，先发送 start_stream 指令唤醒后端
      if (!this.streamStarted) {
        this.ws!.send(JSON.stringify({ command: 'start_stream' }));
        this.streamStarted = true;
      }

      this.ws!.send(JSON.stringify({
        command: 'audio_patch',
        request_id: id,
        data: Array.from(audioPatch),
        sr,
        timestamp: Date.now()
      }));

      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error('Backend inference timeout'));
        }
      }, 8000);
    });
  }

  async finalize(patchProbs: Float32Array[]): Promise<{ top5: any[], distribution: Record<string, number> }> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket disconnected');
    }
    this.streamStarted = false; // 重置标识

    return new Promise((resolve, reject) => {
      const handler = (ev: MessageEvent) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'final') {
          this.ws!.removeEventListener('message', handler);
          resolve({ top5: msg.top5, distribution: msg.distribution });
        }
        if (msg.type === 'error') {
          this.ws!.removeEventListener('message', handler);
          reject(new Error(msg.message));
        }
      };
      this.ws!.addEventListener('message', handler);
      this.ws!.send(JSON.stringify({ command: 'stop' }));
      
      setTimeout(() => {
        this.ws!.removeEventListener('message', handler);
        reject(new Error('Finalize timeout'));
      }, 10000);
    });
  }

  dispose() {
    this.pending.clear();
    this.ws?.close();
  }
}