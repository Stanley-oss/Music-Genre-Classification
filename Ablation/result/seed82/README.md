# seed82 去噪消融结果

这个目录收录 `seed=82` 下六个 backbone 的去噪消融结果。每个模型都包含两种设置：

- `full_dn`：启用 denoise 分支。
- `no_dn`：关闭 denoise 分支，只保留主分类训练。

六个模型分别是 `CNN`、`ResNet`、`LSTM`、`RNN`、`MLP`、`Transformer`。本目录只保留这六个模型的 seed82 正式结果，不放其它历史 seed 或不在本轮消融范围内的模型。

## 目录内容

- `runs/full_dn/`：六个模型启用 denoise 后的训练输出。
- `runs/no_dn/`：六个模型关闭 denoise 后的训练输出。
- 每个 run 目录内包含：
  - `history.json`：训练和验证过程指标。
  - `test_metrics.json`：测试集 segment/song 级别指标。
  - `test_predictions_v1.csv`：测试集逐片段预测结果，已去掉本机绝对路径。
  - `config.json`：路径脱敏后的训练配置。
  - `denoise_ablation_config.json`：本轮消融配置。
  - `label_to_id.json`：类别映射。
- `tables/train_summary.csv`：12 组实验的汇总表。
- `tables/denoise_delta_summary.csv`：同一 backbone 下 `full_dn - no_dn` 的差值表。
- `tables/epoch_history_all_runs.csv`：12 组实验逐 epoch 历史指标。
- `tables/seed82_run_inventory.csv`：结果文件清单。
- `tables/mel_items_seed82_sanitized.csv`：脱敏后的数据索引，只保留一份，避免每个 run 重复提交。
- `figures/tsne_unified_noisy_vs_denoised_seed82.png`：六模型 denoise 前后 t-SNE 合成图。
- `tsne/tsne_unified_noisy_vs_denoised_seed82.png`：同一张图的 t-SNE 目录副本。

## 测试集结果摘要

| 模型 | full_dn song acc | no_dn song acc | song acc 差值 | full_dn segment acc | no_dn segment acc | segment acc 差值 |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.8333 | 0.8533 | -0.0200 | 0.7528 | 0.7724 | -0.0197 |
| ResNet | 0.8933 | 0.8400 | +0.0533 | 0.8115 | 0.7774 | +0.0340 |
| LSTM | 0.8200 | 0.8000 | +0.0200 | 0.7401 | 0.7291 | +0.0110 |
| RNN | 0.7800 | 0.8600 | -0.0800 | 0.7304 | 0.7754 | -0.0450 |
| MLP | 0.8467 | 0.8600 | -0.0133 | 0.7855 | 0.7701 | +0.0153 |
| Transformer | 0.7467 | 0.8067 | -0.0600 | 0.6777 | 0.7514 | -0.0737 |

从这批 seed82 结果看，`ResNet + full_dn` 是整体最好的组合，测试集 song acc 为 `0.8933`，segment acc 为 `0.8115`。但 denoise 的收益和 backbone 有明显关系：ResNet、LSTM 在 song/segment 上都有正增益，CNN、RNN、Transformer 在这组 seed 下反而下降，MLP 则是 segment 提升但 song 略降。写报告时建议按具体表格解释，不要笼统写“去噪模块对所有模型都有效”。
