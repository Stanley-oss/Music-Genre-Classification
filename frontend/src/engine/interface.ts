export abstract class InferenceEngine {
  abstract get name(): string;
  abstract get mode(): 'local' | 'remote';
  abstract get genres(): string[];
  abstract get sampleRate(): number;
  abstract get patchSamples(): number;

  abstract init(): Promise<void>;
  
  /**
   * @param audioPatch 音频切片数据
   * @param sr 实际采样率
   * @returns 分类概率分布
   */
  abstract predict(audioPatch: Float32Array, sr?: number): Promise<{
    probs: Float32Array;
    mel: Float32Array;
    // CNN
    shallowMap?: Float32Array;
    deepMap?: Float32Array;
    shallowShape?: number[];
    deepShape?: number[];
    // LSTM
    hiddenState?: Float32Array;
    cellState?: Float32Array;
    // ResNet
    identityEnergy?: Float32Array;
    residualEnergy?: Float32Array;

    // Feature Viz & UMAP
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
  }>;

  /**
   * 停止流式预测，结合历史数据产生最终结论
   * @param patchProbs 历史 Patch 概率数组
   */
  abstract finalize(patchProbs: Float32Array[]): Promise<{ top5: any[], distribution: Record<string, number> }>;

  dispose(): void {}
}