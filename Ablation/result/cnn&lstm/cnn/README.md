# CNN 第三点专项证据

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

本实验对 CNN 主干进行了额外诊断。首先，在训练 batch 上记录了总损失、分类损失、降噪重构损失以及不同模块的梯度范数，用于说明模型通过反向传播进行梯度优化。其次，模型输入由整首音乐切分为固定长度的 3 秒片段，片段级预测进一步聚合为歌曲级预测，因此实验同时报告 segment-level 和 song-level 结果。最后，在 denoise 分支中对 latent embedding 施加不同强度的随机噪声采样，并比较 noisy embedding 与 denoised embedding 的分类准确率和到 clean embedding 的重构距离，用于验证采样扰动和降噪恢复过程。

## 关键摘要

```json
{
  "model": "cnn",
  "device": "cuda",
  "num_gradient_batches": 30,
  "num_test_segments": 2997,
  "temperature": 1.15,
  "slicing": {
    "segment": {
      "acc": 0.7497497497497497,
      "balanced_acc": 0.7497799525637285,
      "macro_precision": 0.7624125014497127,
      "macro_recall": 0.7497799525637285,
      "macro_f1": 0.7466858574012438
    },
    "song_probability_average": {
      "acc": 0.8,
      "balanced_acc": 0.8,
      "macro_precision": 0.8055065548486601,
      "macro_recall": 0.8,
      "macro_f1": 0.7967254820784886,
      "num_songs": 150.0
    },
    "song_majority_vote": {
      "acc": 0.7933333333333333,
      "balanced_acc": 0.7933333333333333,
      "macro_precision": 0.8024491710559822,
      "macro_recall": 0.7933333333333333,
      "macro_f1": 0.7907824283559577,
      "num_songs": 150.0
    },
    "temperature": 1.15
  },
  "sampling_first_row": {
    "model": "cnn",
    "t_value": 1.0,
    "noise_ratio_1_minus_t": 0.0,
    "repeats": 3,
    "clean_segment_acc_mean": 0.7497497497497497,
    "clean_segment_acc_std": 0.0,
    "clean_song_acc_mean": 0.8000000000000002,
    "clean_song_acc_std": 1.1102230246251565e-16,
    "noisy_segment_acc_mean": 0.7497497497497497,
    "noisy_segment_acc_std": 0.0,
    "denoised_segment_acc_mean": 0.736403069736403,
    "denoised_segment_acc_std": 0.0,
    "noisy_song_acc_mean": 0.8000000000000002,
    "noisy_song_acc_std": 1.1102230246251565e-16,
    "denoised_song_acc_mean": 0.7866666666666666,
    "denoised_song_acc_std": 0.0,
    "noisy_segment_macro_f1_mean": 0.7466858574012439,
    "noisy_segment_macro_f1_std": 1.1102230246251565e-16,
    "denoised_segment_macro_f1_mean": 0.7327276622827835,
    "denoised_segment_macro_f1_std": 1.1102230246251565e-16,
    "noisy_song_macro_f1_mean": 0.7967254820784886,
    "noisy_song_macro_f1_std": 0.0,
    "denoised_song_macro_f1_mean": 0.7830522737911177,
    "denoised_song_macro_f1_std": 1.1102230246251565e-16,
    "noisy_mse_to_clean_mean": 0.0,
    "noisy_mse_to_clean_std": 0.0,
    "denoised_mse_to_clean_mean": 0.6438510417938232,
    "denoised_mse_to_clean_std": 0.0,
    "noisy_cosine_to_clean_mean": 1.0,
    "noisy_cosine_to_clean_std": 0.0,
    "denoised_cosine_to_clean_mean": 0.864500105381012,
    "denoised_cosine_to_clean_std": 0.0
  },
  "sampling_last_row": {
    "model": "cnn",
    "t_value": 0.1,
    "noise_ratio_1_minus_t": 0.9,
    "repeats": 3,
    "clean_segment_acc_mean": 0.7497497497497497,
    "clean_segment_acc_std": 0.0,
    "clean_song_acc_mean": 0.8000000000000002,
    "clean_song_acc_std": 1.1102230246251565e-16,
    "noisy_segment_acc_mean": 0.2513624735846958,
    "noisy_segment_acc_std": 0.006178598337151011,
    "denoised_segment_acc_mean": 0.2909576242909577,
    "denoised_segment_acc_std": 0.0029468671559318834,
    "noisy_song_acc_mean": 0.6977777777777777,
    "noisy_song_acc_std": 0.0062853936105470775,
    "denoised_song_acc_mean": 0.6977777777777777,
    "denoised_song_acc_std": 0.027932900199947843,
    "noisy_segment_macro_f1_mean": 0.2526643499340177,
    "noisy_segment_macro_f1_std": 0.005826174033491893,
    "denoised_segment_macro_f1_mean": 0.2886366721064111,
    "denoised_segment_macro_f1_std": 0.0030006081257141813,
    "noisy_song_macro_f1_mean": 0.699642041893972,
    "noisy_song_macro_f1_std": 0.0031870911563087493,
    "denoised_song_macro_f1_mean": 0.6964343397979724,
    "denoised_song_macro_f1_std": 0.027684129540011326,
    "noisy_mse_to_clean_mean": 1.6306006113688152,
    "noisy_mse_to_clean_std": 0.002799665481542345,
    "denoised_mse_to_clean_mean": 0.9961836139361063,
    "denoised_mse_to_clean_std": 0.0028600329162740685,
    "noisy_cosine_to_clean_mean": 0.11184705793857574,
    "noisy_cosine_to_clean_std": 0.0008822819561932264,
    "denoised_cosine_to_clean_mean": 0.2028184880812963,
    "denoised_cosine_to_clean_std": 0.003396815630686535
  }
}
```
