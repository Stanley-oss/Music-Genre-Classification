# MusicFlowNet Ablation Package

This folder collects the code snapshots and experiment artifacts for the
MusicFlowNet temporal-backbone ablation study.

## What Is Strictly Comparable

Use `results/controlled_seed2025` as the main ablation evidence for the paper.

All runs in this folder use:

- seed: `2025`
- epochs: `60`
- learning rate: `0.0008`
- weight decay: `0.0001`
- denoising auxiliary branch: enabled
- dataset/cache: `seg_3s_base` + `lm3_base_v1`
- test split: 150 songs / 2997 clips

Models compared:

- `original_conformer`: the original MusicFlowNet temporal module + denoise
- `cnn`: CNN replacement backbone + denoise
- `resnet`: ResNet replacement backbone + denoise
- `lstm`: LSTM replacement backbone + denoise
- `rnn`: RNN replacement backbone + denoise
- `mlp`: MLP replacement backbone + denoise
- `transformer`: Transformer replacement backbone + denoise

Main table:

- `results/controlled_seed2025/paper_tables/classification_summary_all7.csv`

Main figures:

- `results/controlled_seed2025/figures/tsne_unified_original_plus_six.png`
- `results/controlled_seed2025/figures/embedding_metrics_bars_original_plus_six.png`

Full t-SNE outputs:

- `results/controlled_seed2025/tsne`

## Recommended Paper Conclusion

Under the controlled seed-2025 setting, the original Conformer temporal module
with denoising performs best overall:

- song accuracy: `0.8667`
- song macro-F1: `0.8737`
- segment accuracy: `0.7985`
- segment macro-F1: `0.8042`

Among replacement backbones, MLP and ResNet are the closest competitors:

- MLP: song accuracy `0.8467`, song macro-F1 `0.8461`
- ResNet: song accuracy `0.8467`, song macro-F1 `0.8458`

ResNet has the strongest denoised embedding separation among the replacement
models, but it does not exceed the original model in final song-level accuracy.

## Code

Code snapshots are in `code/`:

- `train_musicflownet.py`: current unified training script with all supported backbones.
- `train_backbone_suite_seed2025.py`: one-click serial runner for the six replacement backbones.
- `plot_denoise_tsne.py`: t-SNE and embedding-separation analysis script.
- `predict_audio.py`: inference script with checkpoint compatibility for all backbones.
- `reference_emf_fast_ablation.py`: teammate-provided fast ablation reference code.

## Results Layout

`results/controlled_seed2025/backbone_runs` contains the six replacement
backbone runs. Each run contains:

- `best_emf_v1.pt`
- `config.json`
- `history.json`
- `label_to_id.json`
- `mel_items.csv`
- `test_metrics.json`
- `test_predictions_v1.csv`

`results/controlled_seed2025/original_conformer_s2025` contains the matching
original Conformer baseline.

`results/controlled_seed2025/paper_tables` contains compact CSV tables intended
for the paper:

- `classification_summary_all7.csv`
- `embedding_metrics_all7.csv`
- `denoise_movement_all7.csv`
- `denoise_movement_six_replacements.csv`

## Historical Runs

`results/historical_not_strictly_comparable` contains early runs with different
random seeds. These are useful for context, but should not be used as the main
controlled ablation table.

Important example:

- CNN previously reached `0.88` song accuracy with seed `293276`.
- Under the controlled seed-2025 protocol, CNN reaches `0.80`.

This indicates noticeable seed sensitivity, so the paper should avoid claiming
that a single high-scoring run proves one backbone is generally superior.

## Upload Note

The package includes `.pt` checkpoint files. Each checkpoint is below GitHub's
single-file size limit, but the folder is relatively large. If the repository
should stay lightweight, remove `best_emf_v1.pt` files and keep the CSV/JSON/PNG
artifacts.
