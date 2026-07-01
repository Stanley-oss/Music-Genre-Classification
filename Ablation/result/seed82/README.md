# seed82 消融结果目录

这里保留 `seed=82` 的正式消融输出。运行 `Ablation/code/run_all_seed82_ablation.py` 后会生成以下内容：

```text
seed82/
├── runs/
│   ├── full_dn/      # 六个模型，开启 denoise
│   └── no_dn/        # 六个模型，关闭 denoise
├── tables/
│   ├── train_summary.csv
│   └── denoise_delta_summary.csv
├── tsne/             # 每个模型的 denoise 前后 t-SNE
└── figures/          # 六模型合并图
```

为保证 GitHub 仓库清爽，本目录只放 seed82 协议下的结果。旧的调参过程和其他历史结果不再放入正式结果目录。
