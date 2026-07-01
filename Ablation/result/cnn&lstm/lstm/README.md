# LSTM 第三点专项证据

这个文件夹用于对应老师第三条要求：gradient、slicing、sampling。

## 文件说明

- `gradient_diagnostics.csv`：若干训练 batch 的 loss 分量和各模块梯度范数。
- `gradient_norm_curve.png`：backbone、classifier、denoise 等模块的梯度范数曲线。
- `loss_components_curve.png`：classification CE、denoise MSE、denoised CE 和 total loss。
- `slice_level_predictions.csv`：测试集中每个 3 秒 slice 的预测结果。
- `slicing_song_aggregation.csv`：每首歌由多个 slice 聚合成 song-level 预测的过程。
- `segment_vs_song_accuracy.png`：segment-level 与 song-level accuracy 对比。
- `slice_prediction_example.png`：单首歌的逐 slice 预测示例。
- `sampling_noise_ablation_repeats.csv`：不同噪声采样强度、不同 repeat 的原始结果。
- `sampling_noise_ablation.csv`：按噪声强度汇总后的均值和标准差。
- `denoise_accuracy_vs_noise.png`：噪声增强后 noisy / denoised 表征的分类准确率变化。
- `denoise_mse_vs_noise.png`：noisy / denoised 表征到 clean embedding 的 MSE 对比。
- `summary.json`：本文件夹关键指标摘要。

## 可以写进报告的话

本实验对 LSTM 主干进行了额外诊断。首先，在训练 batch 上记录了总损失、分类损失、降噪重构损失以及不同模块的梯度范数，用于说明模型通过反向传播进行梯度优化。其次，模型输入由整首音乐切分为固定长度的 3 秒片段，片段级预测进一步聚合为歌曲级预测，因此实验同时报告 segment-level 和 song-level 结果。最后，在 denoise 分支中对 latent embedding 施加不同强度的随机噪声采样，并比较 noisy embedding 与 denoised embedding 的分类准确率和到 clean embedding 的重构距离，用于验证采样扰动和降噪恢复过程。

## 关键摘要

```json
{
  "model": "lstm",
  "device": "cuda",
  "num_gradient_batches": 30,
  "num_test_segments": 2997,
  "temperature": 1.15,
  "slicing": {
    "segment": {
      "acc": 0.7620954287620955,
      "balanced_acc": 0.7620540803423792,
      "macro_precision": 0.7913994259824616,
      "macro_recall": 0.7620540803423792,
      "macro_f1": 0.7713568715589981
    },
    "song_probability_average": {
      "acc": 0.8333333333333334,
      "balanced_acc": 0.8333333333333334,
      "macro_precision": 0.8592086834733893,
      "macro_recall": 0.8333333333333334,
      "macro_f1": 0.8398191970007876,
      "num_songs": 150.0
    },
    "song_majority_vote": {
      "acc": 0.8266666666666667,
      "balanced_acc": 0.8266666666666668,
      "macro_precision": 0.8513845872797278,
      "macro_recall": 0.8266666666666668,
      "macro_f1": 0.8336147240707131,
      "num_songs": 150.0
    },
    "temperature": 1.15
  },
  "sampling_first_row": {
    "model": "lstm",
    "t_value": 1.0,
    "noise_ratio_1_minus_t": 0.0,
    "repeats": 3,
    "clean_segment_acc_mean": 0.7620954287620955,
    "clean_segment_acc_std": 0.0,
    "clean_song_acc_mean": 0.8333333333333334,
    "clean_song_acc_std": 0.0,
    "noisy_segment_acc_mean": 0.7620954287620955,
    "noisy_segment_acc_std": 0.0,
    "denoised_segment_acc_mean": 0.7654320987654321,
    "denoised_segment_acc_std": 0.0,
    "noisy_song_acc_mean": 0.8333333333333334,
    "noisy_song_acc_std": 0.0,
    "denoised_song_acc_mean": 0.84,
    "denoised_song_acc_std": 0.0,
    "noisy_segment_macro_f1_mean": 0.7713568715589981,
    "noisy_segment_macro_f1_std": 0.0,
    "denoised_segment_macro_f1_mean": 0.7738421060272054,
    "denoised_segment_macro_f1_std": 0.0,
    "noisy_song_macro_f1_mean": 0.8398191970007876,
    "noisy_song_macro_f1_std": 0.0,
    "denoised_song_macro_f1_mean": 0.8439079209115361,
    "denoised_song_macro_f1_std": 0.0,
    "noisy_mse_to_clean_mean": 0.0,
    "noisy_mse_to_clean_std": 0.0,
    "denoised_mse_to_clean_mean": 0.6154974699020386,
    "denoised_mse_to_clean_std": 0.0,
    "noisy_cosine_to_clean_mean": 1.0,
    "noisy_cosine_to_clean_std": 0.0,
    "denoised_cosine_to_clean_mean": 0.8569588661193848,
    "denoised_cosine_to_clean_std": 0.0
  },
  "sampling_last_row": {
    "model": "lstm",
    "t_value": 0.1,
    "noise_ratio_1_minus_t": 0.9,
    "repeats": 3,
    "clean_segment_acc_mean": 0.7620954287620955,
    "clean_segment_acc_std": 0.0,
    "clean_song_acc_mean": 0.8333333333333334,
    "clean_song_acc_std": 0.0,
    "noisy_segment_acc_mean": 0.23990657323990658,
    "noisy_segment_acc_std": 0.008258960389606041,
    "denoised_segment_acc_mean": 0.2895117339561784,
    "denoised_segment_acc_std": 0.004689838425605162,
    "noisy_song_acc_mean": 0.6422222222222221,
    "noisy_song_acc_std": 0.021998877636914816,
    "denoised_song_acc_mean": 0.6244444444444445,
    "denoised_song_acc_std": 0.011331154474650657,
    "noisy_segment_macro_f1_mean": 0.24214279469363856,
    "noisy_segment_macro_f1_std": 0.007818139977301004,
    "denoised_segment_macro_f1_mean": 0.28273147102361257,
    "denoised_segment_macro_f1_std": 0.006020488159390152,
    "noisy_song_macro_f1_mean": 0.6664448365859637,
    "noisy_song_macro_f1_std": 0.019755582339705122,
    "denoised_song_macro_f1_mean": 0.6223220993935382,
    "denoised_song_macro_f1_std": 0.014116480947191153,
    "noisy_mse_to_clean_mean": 1.6258580684661865,
    "noisy_mse_to_clean_std": 0.002089437103719326,
    "denoised_mse_to_clean_mean": 0.9693496028582255,
    "denoised_mse_to_clean_std": 0.0016151445142871734,
    "noisy_cosine_to_clean_mean": 0.10949860513210297,
    "noisy_cosine_to_clean_std": 0.0016042975553807704,
    "denoised_cosine_to_clean_mean": 0.22438386579354605,
    "denoised_cosine_to_clean_std": 0.0030437134706574557
  }
}
```
