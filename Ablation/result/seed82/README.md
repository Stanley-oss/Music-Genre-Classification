# seed82 实验结果目录

这个目录只放 `seed=82` 协议下已经确认存在的真实输出。当前本机可恢复的 seed82 结果不是完整 12 组消融：原始日志显示计划是 6 个模型分别跑 `full_dn` 和 `no_dn`，但实际只留下了 `CNN + full_dn` 的训练目录，以及一张 seed82 的 t-SNE 合成图。旧的其它 seed 结果和不在六模型范围内的结果没有放进这个正式目录。

## 当前已收录

- `runs/full_dn/cnn_cnn_w0p75_d1_full_dn_60ep_s82/`：CNN + denoise 的 seed82 训练输出，包含 `best_emf_v1.pt`、`history.json`、路径脱敏后的 `config.json`、`denoise_ablation_config.json` 和标签映射。
- `tables/cnn_full_dn_epoch_history.csv`：从 `history.json` 导出的 50 个 epoch 逐轮指标。
- `tables/seed82_run_inventory.csv`：原计划 12 个实验的完成情况清单，明确标出哪些结果缺失。
- `figures/tsne_unified_noisy_vs_denoised_seed82.png`：当前可找到的 seed82 六模型 t-SNE 合成图。
- `tsne/tsne_unified_noisy_vs_denoised_seed82.png`：同一张 t-SNE 图，放在 t-SNE 子目录下便于查找。注意，本机没有找到这张图对应的逐模型 t-SNE 点数据或完整训练目录，所以这里先只收录 PNG。
为了让公开仓库干净，原始日志和 `mel_items.csv` 这类带本机路径的运行缓存没有提交。逐 epoch 指标已经从 `history.json` 导出到 `tables/cnn_full_dn_epoch_history.csv`。

## 已知指标

CNN + denoise 这一组计划训练 60 个 epoch，但本机留下的 `history.json` 只有前 50 个 epoch。

| 模型 | 模式 | seed | 参数 | 已完成 epoch | 最好 val segment acc | 最好 val song acc |
|---|---|---:|---|---:|---:|---:|
| CNN | full_dn | 82 | `width_mult=0.75, depth=1` | 50/60 | 0.8565, epoch 49 | 0.9267, epoch 45 |

## 当前缺失

以下结果在本机没有找到，因此没有放入仓库：

- `CNN + no_dn`
- `ResNet + full_dn / no_dn`
- `LSTM + full_dn / no_dn`
- `RNN + full_dn / no_dn`
- `MLP + full_dn / no_dn`
- `Transformer + full_dn / no_dn`

所以，这个目录现在可以作为“已恢复的真实 seed82 产物”，但还不能当作完整的六模型去噪消融训练结果。要写正式消融结论，还需要补跑缺失的 11 组，并重新生成完整的对比表和配套 t-SNE 数据。
