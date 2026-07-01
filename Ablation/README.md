# Ablation Study Documentation

This directory organizes content related specifically to models and experiments, focusing on the comparison of six backbones under a fixed `seed=82`, alongside an ON/OFF ablation study of the denoise module.

## Experimental Objectives

The project evaluates and compares six backbone architectures with the following specifications:

| Model | Key Parameters |
|---|---|
| CNN | `width_mult=0.75`, `depth=1` |
| ResNet | `width_mult=1.0`, `depth=3` |
| LSTM | `hidden_size=64`, `num_layers=1`, `bidirectional=True` |
| RNN | `hidden_size=80`, `num_layers=1`, `bidirectional=True` |
| MLP | `hidden_dims=[192,192,192,192,192]`, `dropout=0.15` |
| Transformer | `d_model=160`, `nhead=4`, `num_layers=1`, `dim_feedforward=320` |

The configures are adjusted 9 times for the parameter tuning of each backbone before getting this set.

Each model is trained under two configurations:

- `full_dn`: Classification Loss + Embedding Denoise MSE + Denoised Embedding Classification Loss.
- `no_dn`: Standard Classification Loss only, used to determine the performance gains brought by the denoise module.

## Directory Structure

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

The six single-model scripts in `code/` can be executed independently. Alternatively, `run_all_seed82_ablation.py` will run all of them sequentially in the following order: CNN, ResNet, LSTM, RNN, MLP, and Transformer.

## How to Run

Ensure that data preprocessing and log-mel caching for the main project are completed first. The training scripts default to reading from:

```text
model/preprocessed/seg_3s_base
model/mel_cache/lm3_base_v1
```

To run a single model ablation (e.g., CNN):

```bash
python Ablation/code/run_cnn_ablation.py
```

To run all six models in sequence:

```bash
python Ablation/code/run_all_seed82_ablation.py
```

Outputs will be automatically saved to:

```text
Ablation/result/seed82/
```

Upon completion of the training, the following files will be generated:

- `tables/train_summary.csv`：egment-level and song-level metrics for each model and mode.
- `tables/denoise_delta_summary.csv`：The metric improvements `full_dn - no_dn` within the same model backbone.
- `tsne/<model>/`：t-SNE plots and coordinates before and after denoising for each model.
- `figures/tsne_unified_noisy_vs_denoised_seed82.png`：A unified composite plot for all six models.


This set of experiments is designed to answer two main research questions:

1. Whether performance variances exist across different backbones given identical data splits, training epochs, and the same random seed.
2. Within the same backbone, how the classification metrics and embedding visualizations change after incorporating the denoise branch.
