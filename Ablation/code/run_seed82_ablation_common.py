from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
RESULT_ROOT = REPO_ROOT / "Ablation" / "result" / "seed82"
RUNS_DIR = RESULT_ROOT / "runs"
TABLES_DIR = RESULT_ROOT / "tables"
TSNE_DIR = RESULT_ROOT / "tsne"
FIGURES_DIR = RESULT_ROOT / "figures"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import train_musicflownet as tm  # noqa: E402


SEED = 82
EPOCHS = 60
RUN_MODES: List[Tuple[str, bool]] = [("full_dn", True), ("no_dn", False)]
MODEL_ORDER = ["cnn", "resnet", "lstm", "rnn", "mlp", "transformer"]

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "cnn": {
        "param_tag": "cnn_w0p75_d1",
        "knobs": {"CNN_WIDTH_MULT": 0.75, "CNN_DEPTH": 1},
    },
    "resnet": {
        "param_tag": "resnet_w1p0_d3",
        "knobs": {"RESNET_WIDTH_MULT": 1.0, "RESNET_DEPTH": 3},
    },
    "lstm": {
        "param_tag": "lstm_h64_l1",
        "knobs": {"LSTM_HIDDEN_SIZE": 64, "LSTM_NUM_LAYERS": 1, "LSTM_DROPOUT": 0.0},
    },
    "rnn": {
        "param_tag": "rnn_h80_l1",
        "knobs": {
            "RNN_HIDDEN_SIZE": 80,
            "RNN_NUM_LAYERS": 1,
            "RNN_BIDIRECTIONAL": True,
            "RNN_DROPOUT": 0.0,
        },
    },
    "mlp": {
        "param_tag": "mlp_d5_h192",
        "knobs": {
            "MLP_HIDDEN_DIMS": [192, 192, 192, 192, 192],
            "MLP_OUTPUT_DIM": 160,
            "MLP_DROPOUT": 0.15,
        },
    },
    "transformer": {
        "param_tag": "tf_d160_h4_ff320_l1",
        "knobs": {
            "TRANSFORMER_D_MODEL": 160,
            "TRANSFORMER_NHEAD": 4,
            "TRANSFORMER_NUM_LAYERS": 1,
            "TRANSFORMER_DIM_FEEDFORWARD": 320,
            "TRANSFORMER_DROPOUT": 0.15,
        },
    },
}

SUMMARY_FIELDS = [
    "seed",
    "mode",
    "structure",
    "temporal_kind",
    "use_denoising",
    "param_tag",
    "segment_acc",
    "segment_macro_f1",
    "song_acc",
    "song_macro_f1",
    "temperature",
    "out_dir",
]

DELTA_FIELDS = [
    "temporal_kind",
    "param_tag",
    "full_song_acc",
    "no_dn_song_acc",
    "delta_song_acc",
    "full_song_macro_f1",
    "no_dn_song_macro_f1",
    "delta_song_macro_f1",
    "full_segment_acc",
    "no_dn_segment_acc",
    "delta_segment_acc",
    "full_out_dir",
    "no_dn_out_dir",
]


def ensure_dirs() -> None:
    for path in [RESULT_ROOT, RUNS_DIR, TABLES_DIR, TSNE_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def reset_training_defaults() -> None:
    tm.EPOCHS = EPOCHS
    tm.FORCE_CUDA = True
    tm.BUILD_MEL_CACHE = False
    tm.EXPORT_EXPLANATIONS = False
    tm.NUM_WORKERS = 0


def apply_model_config(model_name: str) -> str:
    config = MODEL_CONFIGS[model_name]
    for key, value in dict(config["knobs"]).items():
        setattr(tm, key, value)
    return str(config["param_tag"])


def run_dir_for(model_name: str, mode: str, param_tag: str) -> Path:
    return RUNS_DIR / mode / f"{model_name}_{param_tag}_{mode}_{EPOCHS}ep_s{SEED}"


def read_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert_summary(row: Dict[str, object]) -> None:
    path = TABLES_DIR / "train_summary.csv"
    rows = read_rows(path)
    out_dir = str(row.get("out_dir", ""))
    rows = [old for old in rows if str(old.get("out_dir", "")) != out_dir]
    rows.append(row)
    write_rows(path, rows, SUMMARY_FIELDS)


def as_float(row: Dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def write_delta_summary() -> None:
    rows = read_rows(TABLES_DIR / "train_summary.csv")
    by_model: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["temporal_kind"]), {})[str(row["mode"])] = row

    out_rows: List[Dict[str, object]] = []
    for model_name in MODEL_ORDER:
        pair = by_model.get(model_name, {})
        if "full_dn" not in pair or "no_dn" not in pair:
            continue
        full = pair["full_dn"]
        nodn = pair["no_dn"]
        out_rows.append(
            {
                "temporal_kind": model_name,
                "param_tag": full.get("param_tag", ""),
                "full_song_acc": as_float(full, "song_acc"),
                "no_dn_song_acc": as_float(nodn, "song_acc"),
                "delta_song_acc": as_float(full, "song_acc") - as_float(nodn, "song_acc"),
                "full_song_macro_f1": as_float(full, "song_macro_f1"),
                "no_dn_song_macro_f1": as_float(nodn, "song_macro_f1"),
                "delta_song_macro_f1": as_float(full, "song_macro_f1") - as_float(nodn, "song_macro_f1"),
                "full_segment_acc": as_float(full, "segment_acc"),
                "no_dn_segment_acc": as_float(nodn, "segment_acc"),
                "delta_segment_acc": as_float(full, "segment_acc") - as_float(nodn, "segment_acc"),
                "full_out_dir": full.get("out_dir", ""),
                "no_dn_out_dir": nodn.get("out_dir", ""),
            }
        )
    write_rows(TABLES_DIR / "denoise_delta_summary.csv", out_rows, DELTA_FIELDS)


def run_tsne(model_name: str, full_dir: Path, no_dn_dir: Path) -> None:
    full_ckpt = full_dir / "best_emf_v1.pt"
    no_dn_ckpt = no_dn_dir / "best_emf_v1.pt"
    if not full_ckpt.exists() or not no_dn_ckpt.exists():
        print(f"[tsne skip] missing checkpoints for {model_name}")
        return
    out_dir = TSNE_DIR / model_name
    cmd = [
        sys.executable,
        str(MODEL_DIR / "plot_denoise_tsne.py"),
        "--root_dir",
        str(MODEL_DIR),
        "--ckpt",
        str(full_ckpt),
        "--nodn_ckpt",
        str(no_dn_ckpt),
        "--out_dir",
        str(out_dir),
        "--seed",
        str(SEED),
        "--max_samples",
        "1000",
        "--t_value",
        "0.55",
    ]
    print(f"[tsne] {model_name}: {out_dir}")
    subprocess.run(cmd, check=True)


def combine_tsne_grid_if_ready() -> None:
    try:
        from combine_tsne_grid import combine_tsne_grid

        combine_tsne_grid(TSNE_DIR, FIGURES_DIR / f"tsne_unified_noisy_vs_denoised_seed{SEED}.png")
    except Exception as exc:
        print(f"[combined tsne warn] {exc}")


def run_model(model_name: str, make_tsne: bool = True) -> None:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}")

    ensure_dirs()
    reset_training_defaults()
    param_tag = apply_model_config(model_name)
    pair_dirs: Dict[str, Path] = {}

    print("=" * 72)
    print(f"[seed82 ablation] model={model_name} param={param_tag}")
    print("=" * 72)

    for mode, use_denoising in RUN_MODES:
        out_dir = run_dir_for(model_name, mode, param_tag)
        pair_dirs[mode] = out_dir
        if (out_dir / "best_emf_v1.pt").exists() and (out_dir / "test_metrics.json").exists():
            print(f"[skip] already finished: {out_dir}")
            row = tm.read_test_summary(out_dir, SEED, f"{model_name}_{mode}", model_name, use_denoising)
        else:
            row = tm.run_experiment_once(
                seed=SEED,
                structure=f"{model_name}_{mode}",
                temporal_kind=model_name,
                use_denoising=use_denoising,
                out_dir=out_dir,
                build_mel_cache_flag=False,
                export_explanations_flag=False,
            )
        row["mode"] = mode
        row["param_tag"] = param_tag
        upsert_summary(row)
        write_delta_summary()

    if make_tsne:
        run_tsne(model_name, pair_dirs["full_dn"], pair_dirs["no_dn"])
        combine_tsne_grid_if_ready()


def run_all_models() -> None:
    for model_name in MODEL_ORDER:
        run_model(model_name, make_tsne=False)
    for model_name in MODEL_ORDER:
        param_tag = str(MODEL_CONFIGS[model_name]["param_tag"])
        run_tsne(
            model_name,
            run_dir_for(model_name, "full_dn", param_tag),
            run_dir_for(model_name, "no_dn", param_tag),
        )
    combine_tsne_grid_if_ready()
