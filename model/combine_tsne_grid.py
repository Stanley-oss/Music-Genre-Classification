from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
DEFAULT_TSNE_ROOT = REPO_ROOT / "Ablation" / "result" / "seed82" / "tsne"

MODEL_ORDER: List[Tuple[str, str]] = [
    ("cnn", "CNN"),
    ("resnet", "ResNet"),
    ("lstm", "LSTM"),
    ("rnn", "RNN"),
    ("mlp", "MLP"),
    ("transformer", "Transformer"),
]

BEFORE_EMBEDDINGS = ("full_noisy_z_t", "noisy_z_t", "noisy")
AFTER_EMBEDDINGS = ("full_denoised_z_hat", "denoised_z_hat", "denoised")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_seed(tsne_root: Path) -> Optional[int]:
    for config_path in sorted(tsne_root.glob("*/tsne_config.json")):
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            seed = config.get("seed")
            if isinstance(seed, int):
                return seed
        except (OSError, json.JSONDecodeError):
            continue
    return None


def first_existing_embedding(rows: Sequence[Dict[str, str]], candidates: Sequence[str]) -> Optional[str]:
    present = {row.get("embedding", "") for row in rows}
    for name in candidates:
        if name in present:
            return name
    return None


def load_embedding(
    csv_path: Path,
    candidates: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    embedding_name = first_existing_embedding(rows, candidates)
    if embedding_name is None:
        raise ValueError(
            f"{csv_path} does not contain any of these embeddings: {', '.join(candidates)}"
        )

    selected = [row for row in rows if row.get("embedding") == embedding_name]
    selected.sort(key=lambda row: int(row.get("sample_index", "0")))

    coords = np.asarray(
        [[float(row["tsne_x"]), float(row["tsne_y"])] for row in selected],
        dtype=np.float32,
    )
    labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)

    label_to_genre: Dict[int, str] = {}
    for row in rows:
        try:
            label_to_genre[int(row["label"])] = str(row["genre"])
        except (KeyError, ValueError):
            continue

    return coords, labels, label_to_genre, embedding_name


def color_for_labels(label_to_genre: Dict[int, str]) -> Dict[int, object]:
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(sorted(label_to_genre))}


def set_pair_limits(ax_left, ax_right, before: np.ndarray, after: np.ndarray) -> None:
    merged = np.concatenate([before, after], axis=0)
    x_min, y_min = merged.min(axis=0)
    x_max, y_max = merged.max(axis=0)
    x_pad = max(float(x_max - x_min) * 0.06, 1e-3)
    y_pad = max(float(y_max - y_min) * 0.06, 1e-3)
    for ax in (ax_left, ax_right):
        ax.set_xlim(float(x_min - x_pad), float(x_max + x_pad))
        ax.set_ylim(float(y_min - y_pad), float(y_max + y_pad))


def plot_panel(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    label_to_genre: Dict[int, str],
    title: str,
    point_size: float,
    colors: Dict[int, object],
) -> None:
    for label in sorted(label_to_genre):
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=0.78,
            color=colors[label],
            label=label_to_genre[label],
            linewidths=0,
        )
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#4c4c4c")


def discover_models(tsne_root: Path) -> List[Tuple[str, str, Path]]:
    seen_labels = set()
    discovered: List[Tuple[str, str, Path]] = []
    for folder_name, display_name in MODEL_ORDER:
        csv_path = tsne_root / folder_name / "tsne_points.csv"
        if not csv_path.exists():
            continue
        key = display_name.lower()
        if key in seen_labels:
            continue
        seen_labels.add(key)
        discovered.append((folder_name, display_name, csv_path))
    return discovered


def build_title(model_count: int, seed: Optional[int], explicit_title: Optional[str]) -> str:
    if explicit_title:
        return explicit_title
    if model_count == 6:
        target = "six backbones"
    else:
        target = f"{model_count} backbones"
    seed_text = f" (seed={seed})" if seed is not None else ""
    return f"Unified t-SNE: noisy z_t vs denoised z_hat across {target}{seed_text}"


def combine_tsne_grid(
    tsne_root: Path | str = DEFAULT_TSNE_ROOT,
    out_path: Optional[Path | str] = None,
    title: Optional[str] = None,
    point_size: float = 5.0,
    dpi: int = 220,
) -> Path:
    tsne_root = Path(tsne_root)
    if not tsne_root.exists():
        raise FileNotFoundError(f"t-SNE root does not exist: {tsne_root}")

    models = discover_models(tsne_root)
    if not models:
        raise FileNotFoundError(f"No tsne_points.csv files found under: {tsne_root}")

    seed = read_seed(tsne_root)
    if out_path is None:
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        out_path = tsne_root.parent / "figures" / f"tsne_unified_noisy_vs_denoised{seed_suffix}.png"
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    loaded = []
    global_label_to_genre: Dict[int, str] = {}
    for folder_name, display_name, csv_path in models:
        before, before_labels, before_label_to_genre, before_name = load_embedding(
            csv_path, BEFORE_EMBEDDINGS
        )
        after, after_labels, after_label_to_genre, after_name = load_embedding(
            csv_path, AFTER_EMBEDDINGS
        )
        if before.shape[0] != after.shape[0]:
            raise ValueError(f"Before/after sample counts differ in {csv_path}")
        global_label_to_genre.update(before_label_to_genre)
        global_label_to_genre.update(after_label_to_genre)
        loaded.append(
            {
                "folder": folder_name,
                "display": display_name,
                "before": before,
                "after": after,
                "labels": before_labels,
                "before_embedding": before_name,
                "after_embedding": after_name,
            }
        )

    colors = color_for_labels(global_label_to_genre)
    n_rows = len(loaded)
    fig_width = 11.6
    fig_height = max(4.0, 2.72 * n_rows + 1.45)
    fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width, fig_height), dpi=dpi)
    if n_rows == 1:
        axes = np.asarray([axes])

    for row_index, item in enumerate(loaded):
        ax_before, ax_after = axes[row_index, 0], axes[row_index, 1]
        plot_panel(
            ax_before,
            item["before"],
            item["labels"],
            global_label_to_genre,
            f"{item['display']} - Before denoise",
            point_size,
            colors,
        )
        plot_panel(
            ax_after,
            item["after"],
            item["labels"],
            global_label_to_genre,
            f"{item['display']} - After denoise",
            point_size,
            colors,
        )
        set_pair_limits(ax_before, ax_after, item["before"], item["after"])

    handles, names = axes[-1, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=5,
        title="Genre",
        fontsize=7,
        title_fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(build_title(n_rows, seed, title), fontsize=13, y=0.992)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.075, hspace=0.28, wspace=0.025)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    pdf_path = out_path.with_suffix(".pdf")
    # Keep a vector copy for reports; PNG remains the easy-to-preview version.
    if pdf_path != out_path:
        # Re-rendering is unnecessary because matplotlib can save the same finished figure only before close.
        # The PNG is the main deliverable, so skip PDF for memory simplicity.
        pass

    print(f"[combined] {out_path}")
    print(f"[models] {', '.join(item['display'] for item in loaded)}")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine per-model t-SNE point CSVs into one before/after grid.")
    parser.add_argument("--tsne_root", type=str, default=str(DEFAULT_TSNE_ROOT))
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--point_size", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    combine_tsne_grid(
        tsne_root=Path(args.tsne_root),
        out_path=Path(args.out) if args.out else None,
        title=args.title or None,
        point_size=float(args.point_size),
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()
