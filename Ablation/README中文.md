# MusicFlowNet 消融实验整理说明

这个文件夹是给组内写消融实验用的整理包，里面放了训练代码快照、统一 seed 的结果、历史跑出来但不能严格横比的结果、表格和 t-SNE 图。

重点先说清楚：论文主表请优先使用 `results/controlled_seed2025`。`results/historical_not_strictly_comparable` 只能当作背景参考，不能和主表混在一起排名。

## 先看这个结论

在固定 `seed=2025`、训练轮数和超参数一致、并且都保留 denoise 模块的条件下，原始 Conformer 主干整体最好：

- 原始 Conformer + denoise：song accuracy `0.8667`，song macro-F1 `0.8737`
- 替换主干里最接近的是 MLP 和 ResNet，二者 song accuracy 都是 `0.8467`
- ResNet 的 denoised embedding 分离度在替换主干里最好，适合拿来解释“表征分布更清楚”
- CNN 早期不同 seed 跑到过 `0.88`，但在统一 `seed=2025` 后是 `0.80`，所以不能直接写“CNN 最好”

一句话给论文用：在本实验设置下，原始 Conformer 仍然保持最优整体分类性能；若仅比较替换主干，MLP 与 ResNet 表现最接近原模型，其中 ResNet 在降噪后表征空间的类别分离度更突出。

## 文件夹结构

```text
ablation/
├── code/
│   ├── train_musicflownet.py
│   ├── train_backbone_suite_seed2025.py
│   ├── plot_denoise_tsne.py
│   ├── predict_audio.py
│   └── reference_emf_fast_ablation.py
└── results/
    ├── controlled_seed2025/
    │   ├── backbone_runs/
    │   ├── original_conformer_s2025/
    │   ├── paper_tables/
    │   ├── figures/
    │   └── tsne/
    └── historical_not_strictly_comparable/
```

## 严格可比实验

`results/controlled_seed2025` 是主实验，建议论文里的消融表和可视化都从这里取。

统一设置如下：

- seed：`2025`
- epochs：`60`
- learning rate：`0.0008`
- weight decay：`0.0001`
- denoise 模块：全部开启
- 数据缓存：`seg_3s_base` + `lm3_base_v1`
- 测试集：150 首歌，2997 个片段

比较的模型如下：

- `original_conformer`：原始 MusicFlowNet temporal 模块 + denoise
- `cnn`：CNN 替换主干 + denoise
- `resnet`：ResNet 替换主干 + denoise
- `lstm`：LSTM 替换主干 + denoise
- `rnn`：RNN 替换主干 + denoise
- `mlp`：MLP 替换主干 + denoise
- `transformer`：Transformer 替换主干 + denoise

## 主结果表

主表位置：

```text
results/controlled_seed2025/paper_tables/classification_summary_all7.csv
```

核心结果如下：

| 模型 | segment acc | segment macro-F1 | song acc | song macro-F1 |
|---|---:|---:|---:|---:|
| Original Conformer | 0.7985 | 0.8042 | 0.8667 | 0.8737 |
| MLP | 0.7738 | 0.7721 | 0.8467 | 0.8461 |
| ResNet | 0.7988 | 0.7926 | 0.8467 | 0.8458 |
| LSTM | 0.7621 | 0.7714 | 0.8333 | 0.8398 |
| RNN | 0.7624 | 0.7605 | 0.8333 | 0.8312 |
| Transformer | 0.7714 | 0.7726 | 0.8267 | 0.8251 |
| CNN | 0.7494 | 0.7464 | 0.8000 | 0.7967 |

论文里如果只放一个分类结果表，建议就放这个表。排序可以按 `song acc` 或 `song macro-F1`，因为最终是按歌曲级别判断，song-level 指标比 segment-level 更贴近任务目标。

## t-SNE 和表征指标

主要图片：

```text
results/controlled_seed2025/figures/tsne_unified_original_plus_six.png
results/controlled_seed2025/figures/embedding_metrics_bars_original_plus_six.png
```

单模型 t-SNE 在这里：

```text
results/controlled_seed2025/tsne/
```

表征指标表在这里：

```text
results/controlled_seed2025/paper_tables/embedding_metrics_all7.csv
results/controlled_seed2025/paper_tables/denoise_movement_all7.csv
results/controlled_seed2025/paper_tables/denoise_movement_six_replacements.csv
```

这里的指标主要用于解释 denoise 前后的 embedding 变化，不建议拿它直接替代最终分类准确率。比较有用的说法是：denoise 后，多个模型的 silhouette score 和 inter/intra 距离比都有提升，说明 denoise 分支确实把表征往更可分的方向拉。

在 denoised embedding 上，几个代表性结果如下：

| 模型 | denoised classifier acc | silhouette | inter/intra |
|---|---:|---:|---:|
| ResNet | 0.794 | 0.4097 | 2.6674 |
| Original Conformer | 0.795 | 0.4040 | 2.5782 |
| MLP | 0.756 | 0.3532 | 2.3653 |
| Transformer | 0.767 | 0.3521 | 2.3038 |
| RNN | 0.754 | 0.2865 | 2.1372 |
| CNN | 0.723 | 0.2728 | 2.1183 |
| LSTM | 0.747 | 0.2908 | 2.0902 |

注意：ResNet 在表征分离度上很好，但最终 song-level 分类没有超过原始 Conformer，所以论文里可以写“ResNet 展现出较好的表征聚类效果”，不要写成“ResNet 整体最好”。

## 关于 seed 的写法

这个任务对随机种子非常敏感。最明显的例子是 CNN：

- 历史不同 seed 结果：CNN song accuracy 曾达到 `0.88`
- 统一 `seed=2025` 后：CNN song accuracy 是 `0.80`

这说明历史结果和统一 seed 结果不能直接混在一起比较。内部讨论时可以说 seed 影响很大，甚至可能大过一部分结构调整；但论文里不要直接写“seed 的影响大于模型结构”，因为这个结论需要多 seed 的均值和方差来证明。

论文建议写法：

```text
为保证不同主干结构之间的比较尽量公平，本文在相同数据划分、相同训练轮数、相同优化器超参数和固定随机种子的条件下进行消融实验。实验结果显示，原始 Conformer 主干在歌曲级分类指标上保持最优表现；替换主干中，MLP 与 ResNet 表现最接近原模型。同时，我们观察到模型训练对随机初始化和数据顺序存在一定敏感性，因此后续工作可进一步采用多随机种子重复实验，并报告均值与标准差，以获得更稳健的统计结论。
```

不建议这样写：

- 不要写“CNN 是最好的主干”，因为这个只在历史不同 seed 的单次运行里成立
- 不要写“seed 影响一定大于模型结构”，除非后面补跑多 seed 实验
- 不要把 `historical_not_strictly_comparable` 里的结果放进主表和 `controlled_seed2025` 一起排名
- 不要只按 t-SNE 图肉眼观感下结论，t-SNE 只能辅助解释

如果篇幅有限，主文只放统一 seed 的表。历史 CNN 的高分可以不写，或者放到内部讨论和补充说明里，避免老师追问“为什么不同表里的最优模型不一致”。

## 代码说明

代码快照在 `code/`：

- `train_musicflownet.py`：统一训练脚本，已经支持 `conformer / cnn / resnet / lstm / rnn / mlp / transformer`
- `train_backbone_suite_seed2025.py`：一键串行跑六个替换主干，顺序是 CNN、ResNet、LSTM、RNN、MLP、Transformer
- `plot_denoise_tsne.py`：生成 denoise 前后 t-SNE 和 embedding 指标
- `predict_audio.py`：推理脚本，兼容这些 backbone 的 checkpoint
- `reference_emf_fast_ablation.py`：朋友给的 MLP/RNN 等参考代码，保留作来源说明

如果只是写论文，不一定要重新跑训练。优先使用 `results/controlled_seed2025/paper_tables` 里的 CSV 和 `results/controlled_seed2025/figures` 里的图。

## 历史结果怎么处理

`results/historical_not_strictly_comparable` 里是之前不同 seed 或不同整理阶段跑出来的结果。它们有参考价值，但不适合当主消融表。

它最适合用来说明一件事：这个数据集和训练流程存在明显随机性，因此固定 seed 的对照实验是必要的；如果要做更严格的论文版本，最好每个模型补跑 3 到 5 个 seed，然后报告 `mean ± std`。

如果没有算力补跑，当前最稳妥的处理方式是：

1. 主实验只报告 `seed=2025` 的统一对照结果
2. 结论写成“在当前控制实验设置下”，不要写成“绝对优于”
3. 局限性里补一句“单 seed 结果可能受到随机初始化影响”

## GitHub 上传注意

这个包里包含 `.pt` checkpoint。单个 checkpoint 没超过 GitHub 的单文件限制，但整个文件夹比较大。如果仓库想轻一点，可以删掉各个结果目录里的 `best_emf_v1.pt`，只保留 CSV、JSON 和 PNG。写论文时最重要的是：

- `paper_tables/` 里的 CSV
- `figures/` 里的总图
- `tsne/` 里的单模型图
- `code/` 里的训练和画图代码快照

