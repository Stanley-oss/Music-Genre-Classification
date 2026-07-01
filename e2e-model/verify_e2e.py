"""
End-to-end verification: compare training-pipeline mel vs export-pipeline mel,
and compare PyTorch model predictions vs ONNX model predictions.
"""

import os, sys, json, glob
import numpy as np
import torch
import librosa
import onnxruntime as ort

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
model_dir = os.path.join(repo_root, "model")
sys.path.append(model_dir)

from train_musicflownet import EMFv1

# ── Constants from train_musicflownet.py ──
SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 20.0
FMAX = 11025.0
TARGET_FRAMES = 130
TOP_DB_FLOOR = -80.0
TOP_DB_CEIL = 20.0

stats = json.load(
    open(
        os.path.join(model_dir, "mel_cache", "lm3_base_v1", "stats_train.json")
    )
)
GLOBAL_MEAN = stats["mean"]
GLOBAL_STD = stats["std"]

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

# ── Step 1: Find a real test audio file ──
seg_dir = os.path.join(
    current_dir, "..", "Ablation", "code", "preprocessed", "seg_3s_base", "test"
)
test_files = []
for genre in GENRES:
    genre_dir = os.path.join(seg_dir, genre)
    if os.path.isdir(genre_dir):
        wavs = sorted(glob.glob(os.path.join(genre_dir, "*.wav")))
        if wavs:
            test_files.append((genre, wavs[0]))

print(f"Found {len(test_files)} test files")
genre_name, wav_path = test_files[0]
print(f"Using: {genre_name} -> {wav_path}")

# ── Step 2: Training pipeline mel (librosa) ──
y, _ = librosa.load(wav_path, sr=SR, mono=True)
print(f"Audio length: {len(y)} samples ({len(y) / SR:.2f}s)")

mel = librosa.feature.melspectrogram(
    y=y,
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX,
    power=2.0,
)
logmel = librosa.power_to_db(mel, ref=1.0)
logmel = np.clip(logmel, TOP_DB_FLOOR, TOP_DB_CEIL).astype(np.float32)

if logmel.shape[1] < TARGET_FRAMES:
    pad = TARGET_FRAMES - logmel.shape[1]
    logmel = np.pad(
        logmel, ((0, 0), (0, pad)), mode="constant", constant_values=TOP_DB_FLOOR
    )
elif logmel.shape[1] > TARGET_FRAMES:
    logmel = logmel[:, :TARGET_FRAMES]

logmel_norm = (logmel - GLOBAL_MEAN) / GLOBAL_STD
logmel_norm = np.clip(logmel_norm, -6.0, 6.0)

print(f"\n=== Training Pipeline Mel ===")
print(
    f"logmel (before norm) shape={logmel.shape}, min={logmel.min():.4f}, max={logmel.max():.4f}, mean={logmel.mean():.4f}"
)
print(
    f"logmel_norm shape={logmel_norm.shape}, min={logmel_norm.min():.4f}, max={logmel_norm.max():.4f}, mean={logmel_norm.mean():.4f}"
)

# ── Step 3: Run through PyTorch model ──
x_torch = torch.from_numpy(logmel_norm).unsqueeze(0).unsqueeze(0)  # [1,1,128,130]
print(f"PyTorch input shape: {x_torch.shape}")

for tk in ["cnn", "resnet", "lstm"]:
    candidates = glob.glob(
        os.path.join(
            repo_root,
            "Ablation",
            "result",
            "seed82",
            "runs",
            "full_dn",
            f"{tk}_*_full_dn_60ep_s82",
            "best_emf_v1.pt",
        )
    )
    ckpt_path = candidates[0] if candidates else os.path.join(model_dir, "emf_v1_out", "best_emf_v1.pt")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] {tk}: checkpoint not found")
        continue

    model = EMFv1(num_classes=10, temporal_kind=tk, use_denoising=True)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    sd = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(sd, strict=True)
    model.eval()

    with torch.no_grad():
        logits, emb, attn = model(x_torch)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    top3_idx = np.argsort(probs)[::-1][:3]
    print(f"\n[PyTorch {tk}] Predictions on {genre_name}:")
    for i in top3_idx:
        print(f"  {GENRES[i]:12s} {probs[i] * 100:.1f}%")

# ── Step 4: Run through ONNX model ──
print(f"\n=== ONNX E2E Pipeline ===")

# Load raw waveform for ONNX (the export model takes raw audio)
from export_onnx import WaveformToMelExact

mel_module = WaveformToMelExact(
    sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, frames=96
)
mel_module.eval()

# Use same raw audio, but trim/pad to samples_per_patch
samples_per_patch = mel_module.samples_per_patch
print(f"ONNX samples_per_patch: {samples_per_patch} (= {samples_per_patch / SR:.2f}s)")

if len(y) >= samples_per_patch:
    audio_patch = y[:samples_per_patch]
else:
    audio_patch = np.pad(y, (0, samples_per_patch - len(y)))

audio_tensor = torch.from_numpy(audio_patch).unsqueeze(0)  # [1, samples_per_patch]

with torch.no_grad():
    mel_from_export = mel_module(audio_tensor)  # [1,1,128,T]

print(f"Export mel shape: {mel_from_export.shape}")
print(
    f"Export mel min={mel_from_export.min():.4f}, max={mel_from_export.max():.4f}, mean={mel_from_export.mean():.4f}"
)

# Compare with librosa pipeline on the same audio segment
y_segment = audio_patch
mel_seg = librosa.feature.melspectrogram(
    y=y_segment,
    sr=SR,
    n_fft=N_FFT,
    hop_length=HOP,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX,
    power=2.0,
)
logmel_seg = librosa.power_to_db(mel_seg, ref=1.0)
logmel_seg = np.clip(logmel_seg, TOP_DB_FLOOR, TOP_DB_CEIL).astype(np.float32)
logmel_seg_norm = (logmel_seg - GLOBAL_MEAN) / GLOBAL_STD
logmel_seg_norm = np.clip(logmel_seg_norm, -6.0, 6.0)

print(
    f"\nLibrosa mel on same segment: shape={logmel_seg_norm.shape}, min={logmel_seg_norm.min():.4f}, max={logmel_seg_norm.max():.4f}, mean={logmel_seg_norm.mean():.4f}"
)

# Trim to same time dimension for comparison
export_np = mel_from_export.numpy()[0, 0]  # [128, T]
T_min = min(export_np.shape[1], logmel_seg_norm.shape[1])
diff = np.abs(export_np[:, :T_min] - logmel_seg_norm[:, :T_min])
print(f"Mel diff (export vs librosa): max={diff.max():.4f}, mean={diff.mean():.4f}")

# ── Step 5: ONNX Runtime inference ──
for tk in ["cnn", "resnet", "lstm"]:
    onnx_path = os.path.join(current_dir, "exported_models", f"emfv1_{tk}_e2e.onnx")
    if not os.path.exists(onnx_path):
        print(f"[SKIP] ONNX {tk} not found at {onnx_path}")
        continue

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    onnx_probs = sess.run(
        [output_name], {input_name: audio_patch.reshape(1, -1).astype(np.float32)}
    )[0][0]

    top3_idx = np.argsort(onnx_probs)[::-1][:3]
    print(f"\n[ONNX {tk}] Predictions on {genre_name}:")
    for i in top3_idx:
        print(f"  {GENRES[i]:12s} {onnx_probs[i] * 100:.1f}%")

# ── Step 6: Check weight loading ──
print("\n=== Weight Loading Check ===")
for tk in ["cnn"]:
    candidates = glob.glob(
        os.path.join(
            repo_root,
            "Ablation",
            "result",
            "seed82",
            "runs",
            "full_dn",
            f"{tk}_*_full_dn_60ep_s82",
            "best_emf_v1.pt",
        )
    )
    ckpt_path = candidates[0] if candidates else os.path.join(model_dir, "emf_v1_out", "best_emf_v1.pt")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] {tk}: checkpoint not found")
        continue
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    sd = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    model = EMFv1(num_classes=10, temporal_kind=tk, use_denoising=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{tk}] Missing keys: {missing}")
    print(f"[{tk}] Unexpected keys: {unexpected}")
