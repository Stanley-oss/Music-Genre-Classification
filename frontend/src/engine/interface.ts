// File: ./frontend/src/engine/interface.ts
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
  abstract predict(audioPatch: Float32Array, sr?: number): Promise<Float32Array>;

  /**
   * 停止流式预测，结合历史数据产生最终结论
   * @param patchProbs 历史 Patch 概率数组
   */
  abstract finalize(patchProbs: Float32Array[]): Promise<{ top5: any[], distribution: Record<string, number> }>;

  dispose(): void {}
}