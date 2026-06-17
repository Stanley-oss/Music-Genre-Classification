import * as ort from 'onnxruntime-web';
import { InferenceEngine } from './interface';

export class OnnxEngine extends InferenceEngine {
  private modelUrl: string;
  private session: ort.InferenceSession | null;
  private temporalKind: string;

  constructor(temporalKind = 'cnn') {
    super();
    this.temporalKind = temporalKind;
    this.modelUrl = `${(import.meta as any).env.BASE_URL}models/emfv1_${temporalKind}_e2e.onnx`;
    this.session = null;
  }

  get name() { return `Browser ONNX (WASM) — ${this.temporalKind.toUpperCase()}`; }
  get mode(): 'local' | 'remote' { return 'local'; }
  get genres() {
    return['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'];
  }
  get sampleRate() { return 22050; }
  get patchSamples() { return 50688; } 

  async init(): Promise<void> {
    // ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    // 限制单线程，避免由于浏览器不支持 SharedArrayBuffer 引起的跨域隔离报错
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false; 

    this.session = await ort.InferenceSession.create(this.modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      intraOpNumThreads: 1
    });
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
    if (!this.session) throw new Error('Model not initialized');

    let patch = audioPatch;
    if (patch.length < this.patchSamples) {
      const padded = new Float32Array(this.patchSamples);
      padded.set(patch);
      patch = padded;
    } else if (patch.length > this.patchSamples) {
      patch = patch.slice(0, this.patchSamples);
    }

    const tensor = new ort.Tensor('float32', patch, [1, this.patchSamples]);
    const feeds: Record<string, ort.Tensor> = { [this.session.inputNames[0]]: tensor };
    const results = await this.session.run(feeds);
    
    return {
      probs: results['probabilities']?.data as Float32Array || new Float32Array(0),
      mel: results['mel']?.data as Float32Array || new Float32Array(0),
      freqMap: results['freq_map']?.data as Float32Array,
      timeMap: results['time_map']?.data as Float32Array,
      freqShape: results['freq_map']?.dims ? Array.from(results['freq_map'].dims as readonly number[]) : undefined,
      timeShape: results['time_map']?.dims ? Array.from(results['time_map'].dims as readonly number[]) : undefined,
      cleanProbs: results['clean_probs']?.data as Float32Array,
      noisyProbs: results['noisy_probs']?.data as Float32Array,
      denoisedProbs: results['denoised_probs']?.data as Float32Array,
      cleanEmb: results['clean_emb']?.data as Float32Array,
      noisyEmb: results['noisy_emb']?.data as Float32Array,
      denoisedEmb: results['denoised_emb']?.data as Float32Array,
    };
  }

  async finalize(patchProbs: Float32Array[]): Promise<{ top5: any[], distribution: Record<string, number> }> {
    const mean = new Float32Array(this.genres.length);
    for (const p of patchProbs) {
      for (let i = 0; i < mean.length; i++) mean[i] += p[i];
    }
    for (let i = 0; i < mean.length; i++) mean[i] /= patchProbs.length;

    const indexed = Array.from(mean).map((p, i) => ({ genre: this.genres[i], probability: p }));
    indexed.sort((a, b) => b.probability - a.probability);

    const distribution: Record<string, number> = {};
    for (let i = 0; i < this.genres.length; i++) distribution[this.genres[i]] = mean[i];

    return { top5: indexed.slice(0, 5), distribution };
  }
}