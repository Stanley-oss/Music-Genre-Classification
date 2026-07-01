from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
except ImportError as exc:
    raise ImportError("This script needs scikit-learn: pip install scikit-learn") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = SCRIPT_DIR
DEFAULT_CKPT = DEFAULT_ROOT_DIR / "emf_v1_out" / "best_emf_v1.pt"
DEFAULT_NODN_CKPT = DEFAULT_ROOT_DIR / "emf_train_runs" / "cnn_no_dn" / "best_emf_v1.pt"
DEFAULT_OUT_DIR = DEFAULT_ROOT_DIR / "tsne_out"

LEGACY_ROOTS: Tuple[str, ...] = ()


def relocate_known_data_dir(path: Path, root_dir: Path) -> Optional[Path]:
    """Map flat run paths from another machine onto this repo's data layout."""
    name = path.name
    candidates = []
    if name.startswith("seg_"):
        candidates.append(root_dir / "preprocessed" / name)
    if name.startswith("lm") or name.startswith("logmel"):
        candidates.append(root_dir / "mel_cache" / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def import_train_module(root_dir: Path):
    """Import train_musicflownet without requiring librosa for cached-mel analysis."""
    if importlib.util.find_spec("librosa") is None:
        sys.modules.setdefault("librosa", types.ModuleType("librosa"))
    root = str(root_dir)
    if root not in sys.path:
        sys.path.insert(0, root)
    import train_musicflownet as tm  # noqa: E402

    return tm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def remap_path(value: object, root_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.exists():
        return candidate

    normalized = text.replace("/", "\\")
    for legacy_root in LEGACY_ROOTS:
        legacy_normalized = legacy_root.replace("/", "\\")
        if normalized.lower().startswith(legacy_normalized.lower()):
            suffix = normalized[len(legacy_normalized) :].lstrip("\\/")
            mapped = root_dir / suffix
            if mapped.exists():
                return mapped
            relocated = relocate_known_data_dir(mapped, root_dir)
            if relocated is not None:
                return relocated
            return mapped

    if not candidate.is_absolute():
        mapped = root_dir / candidate
        relocated = relocate_known_data_dir(mapped, root_dir)
        if relocated is not None:
            return relocated
        return mapped
    relocated = relocate_known_data_dir(candidate, root_dir)
    if relocated is not None:
        return relocated
    return candidate


def apply_checkpoint_paths(tm, ckpt: Dict[str, object], root_dir: Path) -> Dict[str, str]:
    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}

    tm.ROOT_DIR = root_dir
    applied: Dict[str, str] = {"ROOT_DIR": str(root_dir)}

    for attr in ("SEG_DIR", "MEL_DIR", "OUT_DIR"):
        mapped = remap_path(cfg.get(attr), root_dir)
        if mapped is not None:
            setattr(tm, attr, mapped)
            applied[attr] = str(mapped)

    if "TARGET_FRAMES" in cfg:
        try:
            tm.TARGET_FRAMES = int(cfg["TARGET_FRAMES"])
            applied["TARGET_FRAMES"] = str(tm.TARGET_FRAMES)
        except (TypeError, ValueError):
            pass

    return applied


def infer_temporal_kind(ckpt: Dict[str, object]) -> str:
    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        return "cnn"
    raw = str(cfg.get("TEMPORAL_REFINER", cfg.get("temporal_kind", "cnn"))).lower()
    if "transformer" in raw:
        return "transformer"
    if "lstm" in raw:
        return "lstm"
    if "rnn" in raw:
        return "rnn"
    if "cnn" in raw:
        return "cnn"
    if "resnet" in raw:
        return "resnet"
    return "mlp" if "mlp" in raw else "cnn"


def infer_use_denoising(ckpt: Dict[str, object], fallback: bool) -> bool:
    cfg = ckpt.get("config", {})
    if isinstance(cfg, dict) and "USE_DENOISING" in cfg:
        return bool(cfg["USE_DENOISING"])
    return fallback


def load_model(
    tm,
    ckpt_path: Path,
    device: torch.device,
    root_dir: Path,
    fallback_use_denoising: bool,
) -> Tuple[torch.nn.Module, Dict[str, int], float, float, Dict[str, object], Dict[str, str]]:
    ckpt = load_checkpoint(ckpt_path, device)
    applied_paths = apply_checkpoint_paths(tm, ckpt, root_dir)

    label_to_id = ckpt.get("label_to_id")
    if not isinstance(label_to_id, dict):
        raise KeyError("Checkpoint does not contain a valid label_to_id dictionary.")
    label_to_id = {str(k): int(v) for k, v in label_to_id.items()}

    model = tm.EMFv1(
        num_classes=len(label_to_id),
        temporal_kind=infer_temporal_kind(ckpt),
        use_denoising=infer_use_denoising(ckpt, fallback=fallback_use_denoising),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    mean = float(ckpt.get("mean", 0.0))
    std = float(ckpt.get("std", 1.0))
    return model, label_to_id, mean, std, ckpt, applied_paths


def build_selected_indices(dataset, max_samples: int, seed: int) -> List[int]:
    if max_samples < 0 or len(dataset) <= max_samples:
        return list(range(len(dataset)))

    by_class: Dict[int, List[int]] = {}
    for idx, row in enumerate(dataset.rows):
        label = int(row["label"])
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    per_class = max(1, max_samples // max(len(by_class), 1))
    selected: List[int] = []
    leftovers: List[int] = []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        selected.extend(indices[:per_class])
        leftovers.extend(indices[per_class:])

    rng.shuffle(leftovers)
    selected.extend(leftovers[: max(0, max_samples - len(selected))])
    selected = selected[:max_samples]
    selected.sort()
    return selected


def make_loader(tm, rows: List[Dict[str, object]], split: str, mean: float, std: float, indices: List[int], batch_size: int, device: torch.device) -> DataLoader:
    dataset = tm.MelGenreDataset(rows, split, mean, std)
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


@torch.no_grad()
def extract_full_model(
    model,
    loader: DataLoader,
    device: torch.device,
    t_value: float,
) -> Dict[str, object]:
    clean_list: List[np.ndarray] = []
    noisy_list: List[np.ndarray] = []
    denoised_list: List[np.ndarray] = []
    labels: List[int] = []
    song_ids: List[str] = []
    genres: List[str] = []
    mel_paths: List[str] = []
    pred_clean: List[int] = []
    pred_noisy: List[int] = []
    pred_denoised: List[int] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        z, _, _ = model.encode(x)
        t = torch.full((z.shape[0], 1), float(t_value), device=device)
        eps = torch.randn_like(z)
        z_noisy = t * z + (1.0 - t) * eps
        z_hat = model.denoise(z_noisy, t)

        logits_clean = model.classify(z)
        logits_noisy = model.classify(z_noisy)
        logits_denoised = model.classify(z_hat)

        clean_list.append(z.detach().cpu().numpy())
        noisy_list.append(z_noisy.detach().cpu().numpy())
        denoised_list.append(z_hat.detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy().astype(int).tolist())
        song_ids.extend(list(batch["song_id"]))
        genres.extend(list(batch["genre"]))
        mel_paths.extend(list(batch["mel_path"]))
        pred_clean.extend(logits_clean.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
        pred_noisy.extend(logits_noisy.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
        pred_denoised.extend(logits_denoised.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())

    return {
        "clean": np.concatenate(clean_list, axis=0),
        "noisy": np.concatenate(noisy_list, axis=0),
        "denoised": np.concatenate(denoised_list, axis=0),
        "labels": np.asarray(labels, dtype=np.int64),
        "song_ids": song_ids,
        "genres": genres,
        "mel_paths": mel_paths,
        "pred_clean": np.asarray(pred_clean, dtype=np.int64),
        "pred_noisy": np.asarray(pred_noisy, dtype=np.int64),
        "pred_denoised": np.asarray(pred_denoised, dtype=np.int64),
    }


@torch.no_grad()
def extract_clean_model(model, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    clean_list: List[np.ndarray] = []
    pred_list: List[int] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        z, _, _ = model.encode(x)
        logits = model.classify(z)
        clean_list.append(z.detach().cpu().numpy())
        pred_list.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
    return {
        "clean": np.concatenate(clean_list, axis=0),
        "pred": np.asarray(pred_list, dtype=np.int64),
    }


def standardize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return (arr - mean) / np.maximum(std, 1e-6)


def run_tsne(arrays: Sequence[np.ndarray], seed: int, perplexity: float) -> List[np.ndarray]:
    sizes = [a.shape[0] for a in arrays]
    merged = standardize(np.concatenate(arrays, axis=0))
    n = merged.shape[0]
    safe_perplexity = min(float(perplexity), max(2.0, (n - 1) / 3.0))
    safe_perplexity = max(2.0, safe_perplexity)

    print(f"[tsne] points={n}, dim={merged.shape[1]}, perplexity={safe_perplexity:.1f}")
    kwargs = dict(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    try:
        coords = TSNE(max_iter=1200, **kwargs).fit_transform(merged)
    except TypeError:
        coords = TSNE(n_iter=1200, **kwargs).fit_transform(merged)

    out: List[np.ndarray] = []
    start = 0
    for size in sizes:
        out.append(coords[start : start + size])
        start += size
    return out


def color_for_labels(id_to_label: Dict[int, str]) -> Dict[int, object]:
    cmap = plt.get_cmap("tab10")
    return {class_id: cmap(i % 10) for i, class_id in enumerate(sorted(id_to_label))}


def plot_embedding(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    id_to_label: Dict[int, str],
    title: str,
    point_size: float,
    colors: Dict[int, object],
) -> None:
    for class_id in sorted(id_to_label.keys()):
        mask = labels == class_id
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=0.78,
            color=colors[class_id],
            label=id_to_label[class_id],
            linewidths=0,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def set_common_limits(axes: Iterable[object], arrays: Sequence[np.ndarray]) -> None:
    merged = np.concatenate(arrays, axis=0)
    x_min, y_min = merged.min(axis=0)
    x_max, y_max = merged.max(axis=0)
    x_pad = max((x_max - x_min) * 0.06, 1e-3)
    y_pad = max((y_max - y_min) * 0.06, 1e-3)
    for ax in axes:
        ax.set_xlim(float(x_min - x_pad), float(x_max + x_pad))
        ax.set_ylim(float(y_min - y_pad), float(y_max + y_pad))


def save_panels(
    path: Path,
    arrays: Sequence[np.ndarray],
    labels: np.ndarray,
    id_to_label: Dict[int, str],
    titles: Sequence[str],
    point_size: float,
) -> None:
    ensure_dir(path.parent)
    colors = color_for_labels(id_to_label)
    fig, axes = plt.subplots(1, len(arrays), figsize=(6.2 * len(arrays), 5.8), dpi=180)
    if len(arrays) == 1:
        axes = [axes]
    for ax, coords, title in zip(axes, arrays, titles):
        plot_embedding(ax, coords, labels, id_to_label, title, point_size, colors)
    set_common_limits(axes, arrays)
    handles, names = axes[-1].get_legend_handles_labels()
    fig.legend(handles, names, loc="center right", fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {path}")


def save_arrow_plot(
    path: Path,
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    id_to_label: Dict[int, str],
    max_arrows: int,
    seed: int,
) -> None:
    ensure_dir(path.parent)
    colors = color_for_labels(id_to_label)
    n = before.shape[0]
    rng = np.random.default_rng(seed)
    chosen = np.arange(n)
    if n > max_arrows:
        chosen = rng.choice(chosen, size=max_arrows, replace=False)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    for class_id in sorted(id_to_label.keys()):
        mask = labels == class_id
        if not np.any(mask):
            continue
        ax.scatter(
            after[mask, 0],
            after[mask, 1],
            s=16,
            alpha=0.76,
            color=colors[class_id],
            label=id_to_label[class_id],
            linewidths=0,
        )
    for i in chosen:
        ax.annotate(
            "",
            xy=(after[i, 0], after[i, 1]),
            xytext=(before[i, 0], before[i, 1]),
            arrowprops={"arrowstyle": "->", "lw": 0.45, "alpha": 0.35, "color": "#333333"},
        )
    set_common_limits([ax], [before, after])
    ax.set_title("Movement from noisy z_t to denoised z_hat", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {path}")


def mean_intra_class_distance(emb: np.ndarray, labels: np.ndarray) -> float:
    vals: List[float] = []
    for cls in sorted(set(labels.tolist())):
        x = emb[labels == cls]
        if x.shape[0] < 2:
            continue
        center = x.mean(axis=0, keepdims=True)
        vals.append(float(np.linalg.norm(x - center, axis=1).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def mean_inter_class_centroid_distance(emb: np.ndarray, labels: np.ndarray) -> float:
    centers: List[np.ndarray] = []
    for cls in sorted(set(labels.tolist())):
        x = emb[labels == cls]
        if x.shape[0] > 0:
            centers.append(x.mean(axis=0))
    if len(centers) < 2:
        return float("nan")
    vals: List[float] = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            vals.append(float(np.linalg.norm(centers[i] - centers[j])))
    return float(np.mean(vals))


def safe_silhouette(emb: np.ndarray, labels: np.ndarray, seed: int) -> float:
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2 or emb.shape[0] <= len(unique):
        return float("nan")
    x = standardize(emb)
    y = labels
    max_for_sil = min(1500, x.shape[0])
    if x.shape[0] > max_for_sil:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(x.shape[0]), size=max_for_sil, replace=False)
        x = x[idx]
        y = y[idx]
    try:
        return float(silhouette_score(x, y, metric="euclidean"))
    except Exception:
        return float("nan")


def accuracy(pred: Optional[np.ndarray], labels: np.ndarray) -> float:
    if pred is None:
        return float("nan")
    return float((pred == labels).mean())


def mse_to_ref(emb: np.ndarray, ref: Optional[np.ndarray]) -> float:
    if ref is None:
        return float("nan")
    return float(np.mean((emb - ref) ** 2))


def cosine_to_ref(emb: np.ndarray, ref: Optional[np.ndarray]) -> float:
    if ref is None:
        return float("nan")
    a = torch.from_numpy(emb.astype(np.float32))
    b = torch.from_numpy(ref.astype(np.float32))
    return float(F.cosine_similarity(a, b, dim=1).mean().item())


def build_metrics_rows(
    embeddings: Dict[str, np.ndarray],
    labels: np.ndarray,
    seed: int,
    predictions: Optional[Dict[str, np.ndarray]] = None,
    clean_ref: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    predictions = predictions or {}
    rows: List[Dict[str, object]] = []
    for name, emb in embeddings.items():
        intra = mean_intra_class_distance(emb, labels)
        inter = mean_inter_class_centroid_distance(emb, labels)
        rows.append(
            {
                "embedding": name,
                "num_samples": int(emb.shape[0]),
                "dim": int(emb.shape[1]),
                "classifier_acc_on_selected": accuracy(predictions.get(name), labels),
                "silhouette_score": safe_silhouette(emb, labels, seed),
                "mean_intra_class_distance": intra,
                "mean_inter_class_centroid_distance": inter,
                "inter_over_intra": float(inter / intra) if intra and np.isfinite(intra) else float("nan"),
                "mse_to_full_clean_z": mse_to_ref(emb, clean_ref),
                "cosine_to_full_clean_z": cosine_to_ref(emb, clean_ref),
            }
        )
    return rows


def save_point_csv(
    path: Path,
    labels: np.ndarray,
    song_ids: Sequence[str],
    genres: Sequence[str],
    coords_by_name: Dict[str, np.ndarray],
) -> None:
    rows: List[Dict[str, object]] = []
    for name, coords in coords_by_name.items():
        for i in range(coords.shape[0]):
            rows.append(
                {
                    "embedding": name,
                    "sample_index": i,
                    "song_id": song_ids[i],
                    "genre": genres[i],
                    "label": int(labels[i]),
                    "tsne_x": float(coords[i, 0]),
                    "tsne_y": float(coords[i, 1]),
                }
            )
    write_csv(path, rows, ["embedding", "sample_index", "song_id", "genre", "label", "tsne_x", "tsne_y"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate comparable t-SNE plots for MusicFlowNet denoising embeddings."
    )
    parser.add_argument("--root_dir", type=str, default=str(DEFAULT_ROOT_DIR), help="Project root containing train_musicflownet.py.")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT), help="Full denoising model checkpoint.")
    parser.add_argument("--nodn_ckpt", type=str, default=str(DEFAULT_NODN_CKPT), help="Optional no-denoising ablation checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to visualize.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder for figures and metrics.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum clips used for t-SNE. Use -1 for all clips.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding extraction.")
    parser.add_argument("--t_value", type=float, default=0.55, help="Noise mix coefficient in z_t = t*z + (1-t)*eps.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity; reduced automatically if needed.")
    parser.add_argument("--seed", type=int, default=3407, help="Seed for sample selection, noise, and t-SNE.")
    parser.add_argument("--point_size", type=float, default=12.0, help="Scatter point size.")
    parser.add_argument("--max_arrows", type=int, default=180, help="Maximum arrows in movement plot.")
    parser.add_argument("--force_cpu", action="store_true", help="Run embedding extraction on CPU.")
    parser.add_argument("--skip_nodn", action="store_true", help="Skip full-vs-no-denoise comparison.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(int(args.seed))

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    tm = import_train_module(root_dir)
    device = get_device(bool(args.force_cpu))
    print(f"[device] {device}")
    print(f"[root] {root_dir}")

    ckpt_path = Path(args.ckpt)
    model, label_to_id, mean, std, ckpt, applied_paths = load_model(
        tm,
        ckpt_path=ckpt_path,
        device=device,
        root_dir=root_dir,
        fallback_use_denoising=True,
    )
    id_to_label = {int(v): str(k) for k, v in label_to_id.items()}
    print(f"[ckpt] {ckpt_path}")
    print(f"[paths] {applied_paths}")

    rows, discovered_label_to_id = tm.list_wav_rows()
    if {str(k): int(v) for k, v in discovered_label_to_id.items()} != label_to_id:
        print("[warn] discovered label_to_id differs from checkpoint label_to_id; using checkpoint order for plots.")

    full_dataset = tm.MelGenreDataset(rows, args.split, mean, std)
    selected_indices = build_selected_indices(full_dataset, int(args.max_samples), int(args.seed))
    full_loader = make_loader(tm, rows, args.split, mean, std, selected_indices, int(args.batch_size), device)
    print(f"[data] split={args.split}, total_clips={len(full_dataset)}, selected_clips={len(selected_indices)}")

    extracted = extract_full_model(
        model=model,
        loader=full_loader,
        device=device,
        t_value=float(args.t_value),
    )
    clean = extracted["clean"]
    noisy = extracted["noisy"]
    denoised = extracted["denoised"]
    labels = extracted["labels"]

    clean_2d, noisy_2d, denoised_2d = run_tsne(
        [clean, noisy, denoised],
        seed=int(args.seed),
        perplexity=float(args.perplexity),
    )
    save_panels(
        out_dir / "tsne_noisy_vs_denoised.png",
        [noisy_2d, denoised_2d],
        labels,
        id_to_label,
        ["Before denoising: noisy z_t", "After denoising: z_hat"],
        float(args.point_size),
    )
    save_panels(
        out_dir / "tsne_clean_noisy_denoised.png",
        [clean_2d, noisy_2d, denoised_2d],
        labels,
        id_to_label,
        ["Clean embedding z", "Noisy embedding z_t", "Denoised embedding z_hat"],
        float(args.point_size),
    )
    save_arrow_plot(
        out_dir / "tsne_noisy_to_denoised_arrows.png",
        noisy_2d,
        denoised_2d,
        labels,
        id_to_label,
        max_arrows=int(args.max_arrows),
        seed=int(args.seed),
    )

    metric_rows = build_metrics_rows(
        {
            "full_clean_z": clean,
            "full_noisy_z_t": noisy,
            "full_denoised_z_hat": denoised,
        },
        labels,
        seed=int(args.seed),
        predictions={
            "full_clean_z": extracted["pred_clean"],
            "full_noisy_z_t": extracted["pred_noisy"],
            "full_denoised_z_hat": extracted["pred_denoised"],
        },
        clean_ref=clean,
    )

    coords_for_csv = {
        "full_clean_z": clean_2d,
        "full_noisy_z_t": noisy_2d,
        "full_denoised_z_hat": denoised_2d,
    }

    nodn_info: Dict[str, object] = {"used": False}
    nodn_path = Path(args.nodn_ckpt) if str(args.nodn_ckpt).strip() else None
    if not args.skip_nodn and nodn_path is not None and nodn_path.exists():
        nodn_model, nodn_label_to_id, nodn_mean, nodn_std, _, nodn_paths = load_model(
            tm,
            ckpt_path=nodn_path,
            device=device,
            root_dir=root_dir,
            fallback_use_denoising=False,
        )
        if nodn_label_to_id != label_to_id:
            print("[warn] no-denoise label_to_id differs; full-vs-nodn plot may be invalid.")
        comparable_keys = ("SEG_DIR", "MEL_DIR", "TARGET_FRAMES")
        same_input_setup = all(nodn_paths.get(k) == applied_paths.get(k) for k in comparable_keys)
        if not same_input_setup:
            print(
                "[warn] no-denoise checkpoint uses different input/cache settings; "
                "skip full-vs-nodn comparison to avoid mixing incompatible embeddings."
            )
            print(f"[warn] full paths: {applied_paths}")
            print(f"[warn] nodn paths: {nodn_paths}")
            nodn_info = {"used": False, "path": str(nodn_path), "reason": "input/cache settings differ"}
        else:
            nodn_loader = make_loader(tm, rows, args.split, nodn_mean, nodn_std, selected_indices, int(args.batch_size), device)
            nodn_extracted = extract_clean_model(nodn_model, nodn_loader, device)
            nodn_clean = nodn_extracted["clean"]

            full_clean_cmp_2d, nodn_clean_2d = run_tsne(
                [clean, nodn_clean],
                seed=int(args.seed),
                perplexity=float(args.perplexity),
            )
            save_panels(
                out_dir / "tsne_full_vs_nodn_clean.png",
                [full_clean_cmp_2d, nodn_clean_2d],
                labels,
                id_to_label,
                ["With denoising auxiliary branch", "Without denoising branch"],
                float(args.point_size),
            )
            metric_rows.extend(
                build_metrics_rows(
                    {
                        "full_model_clean_z": clean,
                        "no_denoise_model_clean_z": nodn_clean,
                    },
                    labels,
                    seed=int(args.seed),
                    predictions={
                        "full_model_clean_z": extracted["pred_clean"],
                        "no_denoise_model_clean_z": nodn_extracted["pred"],
                    },
                    clean_ref=None,
                )
            )
            coords_for_csv["full_model_clean_z_for_nodn_tsne"] = full_clean_cmp_2d
            coords_for_csv["no_denoise_model_clean_z"] = nodn_clean_2d
            nodn_info = {"used": True, "path": str(nodn_path), "applied_paths": nodn_paths}
    elif not args.skip_nodn:
        print(f"[warn] no-denoise checkpoint not found, skip full-vs-nodn plot: {nodn_path}")

    np.savez_compressed(
        out_dir / f"embeddings_{args.split}.npz",
        clean=clean,
        noisy=noisy,
        denoised=denoised,
        labels=labels,
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
        t_value=np.asarray([float(args.t_value)], dtype=np.float32),
    )
    write_csv(
        out_dir / "embedding_separation_metrics.csv",
        metric_rows,
        [
            "embedding",
            "num_samples",
            "dim",
            "classifier_acc_on_selected",
            "silhouette_score",
            "mean_intra_class_distance",
            "mean_inter_class_centroid_distance",
            "inter_over_intra",
            "mse_to_full_clean_z",
            "cosine_to_full_clean_z",
        ],
    )
    save_point_csv(
        out_dir / "tsne_points.csv",
        labels,
        extracted["song_ids"],
        extracted["genres"],
        coords_for_csv,
    )
    save_json(
        out_dir / "tsne_config.json",
        {
            "root_dir": str(root_dir),
            "ckpt": str(ckpt_path),
            "nodn": nodn_info,
            "split": args.split,
            "max_samples": int(args.max_samples),
            "selected_clips": len(selected_indices),
            "t_value": float(args.t_value),
            "perplexity": float(args.perplexity),
            "seed": int(args.seed),
            "out_dir": str(out_dir),
            "mean": mean,
            "std": std,
            "applied_paths": applied_paths,
        },
    )

    print("[done] t-SNE figures and metrics saved to:")
    print(out_dir)


if __name__ == "__main__":
    main()
