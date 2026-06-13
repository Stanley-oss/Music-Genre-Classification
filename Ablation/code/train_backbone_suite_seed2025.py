from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import train_musicflownet as tm


# Best original-model seed from E:\dl\emf_v1_sweep:
# v1_full_s2025 ties for top song accuracy and has the best song macro-F1.
SEED = 2025
EPOCHS = 60
RUN_ORDER = ["cnn", "resnet", "lstm", "rnn", "mlp", "transformer"]
RUN_ROOT = ROOT_DIR / "emf_train_runs_seed2025_backbones"
SKIP_FINISHED = True


SUMMARY_FIELDS = [
    "seed",
    "structure",
    "temporal_kind",
    "use_denoising",
    "segment_acc",
    "segment_macro_f1",
    "song_acc",
    "song_macro_f1",
    "temperature",
    "out_dir",
]


def read_summary(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(tm.csv.DictReader(f))


def upsert_summary(path: Path, row: Dict[str, object]) -> None:
    rows = read_summary(path)
    out_dir = str(row.get("out_dir", ""))
    rows = [old for old in rows if str(old.get("out_dir", "")) != out_dir]
    rows.append(row)
    tm.write_csv(path, rows, SUMMARY_FIELDS)


def main() -> None:
    tm.EPOCHS = EPOCHS
    tm.FORCE_CUDA = True
    tm.BUILD_MEL_CACHE = True
    tm.EXPORT_EXPLANATIONS = False

    tm.ensure_dir(RUN_ROOT)
    summary_path = RUN_ROOT / "train_summary.csv"

    print("=" * 72)
    print(f"[suite] seed={SEED}")
    print(f"[suite] epochs={tm.EPOCHS} lr={tm.LEARNING_RATE} weight_decay={tm.WEIGHT_DECAY}")
    print(f"[suite] denoise={tm.USE_DENOISING}")
    print(f"[suite] order={', '.join(RUN_ORDER)}")
    print(f"[suite] out={RUN_ROOT}")
    print("=" * 72)

    for temporal_kind in RUN_ORDER:
        out_dir = RUN_ROOT / f"{temporal_kind}_{tm.EPOCHS}ep_s{SEED}"
        if SKIP_FINISHED and (out_dir / "best_emf_v1.pt").exists() and (out_dir / "test_metrics.json").exists():
            print(f"[skip] already finished: {out_dir}")
            row = tm.read_test_summary(out_dir, SEED, temporal_kind, temporal_kind, True)
            upsert_summary(summary_path, row)
            continue

        row = tm.run_experiment_once(
            seed=SEED,
            structure=temporal_kind,
            temporal_kind=temporal_kind,
            use_denoising=True,
            out_dir=out_dir,
            build_mel_cache_flag=True,
            export_explanations_flag=False,
        )
        upsert_summary(summary_path, row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 72)
    print(f"[suite done] {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
