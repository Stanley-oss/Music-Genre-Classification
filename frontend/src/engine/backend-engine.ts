import { InferenceEngine } from './interface';

export class BackendEngine extends InferenceEngine {
  private url: string;
  private ws: WebSocket | null;
  private reqId: number;
  private pending: Map<number, { resolve: Function, reject: Function }>;
  private streamStarted: boolean;

  constructor() {
    super();
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Remove leading slash from BASE_URL if we are appending to host, or keep it depending on how it's formed
    // import.meta.env.BASE_URL always starts and ends with '/' (e.g., '/music-genre/')
    const base = (import.meta as any).env.BASE_URL;
    this.url = `${protocol}//${window.location.host}${base}ws/inference`;
    // const host = (import.meta as any).env?.DEV ? 'localhost:8000' : window.location.host;
    // this.url = `${protocol}//${host}${base}ws/inference`;
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
    // 声明接收类型（尽管后端发来的是JSON字符串，规范写上不影响）
    this.ws.binaryType = 'arraybuffer';
    
    await new Promise<void>((resolve, reject) => {
      this.ws!.onopen = () => resolve();
      this.ws!.onerror = (e) => reject(e);
    });

    this.ws!.onmessage = (ev) => {
      // 后端返回的概率依然是极小的 JSON
      if (typeof ev.data === 'string') {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'patch' && msg.request_id !== undefined) {
          const p = this.pending.get(msg.request_id);
          if (p) {
            this.pending.delete(msg.request_id);
            p.resolve({
              probs: new Float32Array(msg.probabilities),
              mel: new Float32Array(0),
              activations: new Float32Array(0)
            });
          }
        }
      }
    };
  }

  async predict(audioPatch: Float32Array, sr = 22050): Promise<{
    probs: Float32Array;
    mel: Float32Array;
    shallowMap?: Float32Array;
    deepMap?: Float32Array;
    shallowShape?: number[];
    deepShape?: number[];
    hiddenState?: Float32Array;
    cellState?: Float32Array;
    identityEnergy?: Float32Array;
    residualEnergy?: Float32Array;
    freqMap?: Float32Array;
    timeMap?: Float32Array;
    freqShape?: number[];
    timeShape?: number[];
    cleanProbs?: Float32Array;
    noisyProbs?: Float32Array;
    denoisedProbs?: Float32Array;
    cleanEmb?: Float32Array;
    noisyEmb?: Float32Array;
    denoisedEmb?: Float32Array;
  }> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket disconnected');
    }
    const id = ++this.reqId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });

      // 控制信令依然走 JSON（体积极小）
      if (!this.streamStarted) {
        this.ws!.send(JSON.stringify({ command: 'start_stream' }));
        this.streamStarted = true;
      }

      // 核心修复：纯二进制传输
      // Header 长度: 16 Bytes
      // [0-3]: request_id (Uint32)
      //[4-7]: sample_rate (Uint32)
      // [8-15]: timestamp (Float64)
      const buffer = new ArrayBuffer(16 + audioPatch.byteLength);
      const view = new DataView(buffer);
      view.setUint32(0, id, true); // true = Little-Endian
      view.setUint32(4, sr, true);
      view.setFloat64(8, Date.now(), true);
      
      // 数据段：直接将 Float32 注入共享内存
      const audioView = new Float32Array(buffer, 16);
      audioView.set(audioPatch);

      // 直接发送二进制 Buffer
      this.ws!.send(buffer);

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
    this.streamStarted = false;

    return new Promise((resolve, reject) => {
      const handler = (ev: MessageEvent) => {
        if (typeof ev.data === 'string') {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'final') {
            this.ws!.removeEventListener('message', handler);
            resolve({ top5: msg.top5, distribution: msg.distribution });
          }
          if (msg.type === 'error') {
            this.ws!.removeEventListener('message', handler);
            reject(new Error(msg.message));
          }
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