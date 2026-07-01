# 消融实验说明

本目录只整理模型和实验相关内容，重点是固定 `seed=82` 下的六个 backbone 对比，以及 denoise 模块的开关消融。

## 实验目标

本项目最终比较六个主干结构：

| 模型 | 主要参数 |
|---|---|
| CNN | `width_mult=0.75`, `depth=1` |
| ResNet | `width_mult=1.0`, `depth=3` |
| LSTM | `hidden_size=64`, `num_layers=1`, `bidirectional=True` |
| RNN | `hidden_size=80`, `num_layers=1`, `bidirectional=True` |
| MLP | `hidden_dims=[192,192,192,192,192]`, `dropout=0.15` |
| Transformer | `d_model=160`, `nhead=4`, `num_layers=1`, `dim_feedforward=320` |

每个模型跑两组：

- `full_dn`：分类损失 + embedding denoise MSE + denoised embedding 分类损失。
- `no_dn`：只保留普通分类损失，用来判断 denoise 模块是否带来收益。

## 目录结构

```text
Ablation/
├── code/
│   ├── run_all_seed82_ablation.py
│   ├── run_cnn_ablation.py
│   ├── run_resnet_ablation.py
│   ├── run_lstm_ablation.py
│   ├── run_rnn_ablation.py
│   ├── run_mlp_ablation.py
│   ├── run_transformer_ablation.py
│   └── run_seed82_ablation_common.py
└── result/
    └── seed82/
        ├── runs/
        ├── tables/
        ├── tsne/
        └── figures/
```

`code/` 里六个单模型脚本可以分别运行；`run_all_seed82_ablation.py` 会按 CNN、ResNet、LSTM、RNN、MLP、Transformer 的顺序全部跑完。

## 如何运行

先完成主项目的数据预处理和 log-mel 缓存。训练脚本默认读取：

```text
model/preprocessed/seg_3s_base
model/mel_cache/lm3_base_v1
```

如果只跑一个模型，例如 CNN：

```bash
python Ablation/code/run_cnn_ablation.py
```

如果一次跑完六个模型：

```bash
python Ablation/code/run_all_seed82_ablation.py
```

输出会自动写入：

```text
Ablation/result/seed82/
```

训练完成后会生成：

- `tables/train_summary.csv`：每个模型、每个模式的 segment/song 指标。
- `tables/denoise_delta_summary.csv`：同一模型下 `full_dn - no_dn` 的差值。
- `tsne/<model>/`：每个模型 denoise 前后的 t-SNE 图和坐标。
- `figures/tsne_unified_noisy_vs_denoised_seed82.png`：六个模型合并图。

## 写论文时的口径

这组实验用于回答两个问题：

1. 在相同数据划分、相同训练轮数、相同 seed 下，不同 backbone 的表现是否有差异。
2. 在同一个 backbone 内，加入 denoise 分支后，分类指标和 embedding 可视化是否发生变化。

注意不要把历史不同 seed 的结果混进这张正式消融表。GitHub 仓库只保留 `seed82` 目录下的正式结果；旧调参过程可以在本地留档，但不再作为仓库正式结果提交。
