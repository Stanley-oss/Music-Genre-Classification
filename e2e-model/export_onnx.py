import os
import sys
import torch
import torch.nn as nn
import onnx

# Add required paths
current_dir = os.path.dirname(os.path.abspath(__file__))
ablation_code_dir = os.path.join(current_dir, '..', 'Ablation', 'code')
resnet_e2e_dir = os.path.join(current_dir, '..', 'resnet-e2e-model')
sys.path.append(ablation_code_dir)
sys.path.append(resnet_e2e_dir)

from train_musicflownet import EMFv1
import torchaudio

# Global mean/std from Ablation/code/mel_cache/lm3_base_v1/stats_train.json
GLOBAL_MEAN = -12.98087773158693
GLOBAL_STD = 16.636949104826435

class WaveformToMelExact(nn.Module):
    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mels=128, frames=96):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.samples_per_patch = n_fft + hop_length * (frames - 1)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))
        
        fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=20.0,
            f_max=11025.0,
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("mel_filterbank", fbanks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stft_out = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window, center=True, pad_mode="reflect",
            normalized=False, onesided=True, return_complex=False
        )
        power = stft_out[..., 0] ** 2 + stft_out[..., 1] ** 2
        mel_power = torch.matmul(power.transpose(1, 2), self.mel_filterbank).transpose(1, 2)
        mel_db = 10.0 * torch.log10(mel_power.clamp(min=1e-10))
        mel_db = mel_db.clamp(min=-80.0, max=20.0)
        mel_db = (mel_db - GLOBAL_MEAN) / GLOBAL_STD
        mel_db = mel_db.clamp(min=-6.0, max=6.0)
        return mel_db.unsqueeze(1)

class E2E_EMFv1(nn.Module):
    def __init__(self, temporal_kind, sr=22050, n_fft=2048, hop_length=512, n_mels=128, frames=96):
        super().__init__()
        self.waveform_to_mel = WaveformToMelExact(sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, frames=frames)
        self.emfv1 = EMFv1(num_classes=10, temporal_kind=temporal_kind, use_denoising=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self.waveform_to_mel(x)
        logits, emb, attn = self.emfv1(mel)
        return torch.softmax(logits, dim=1)

def export_model(temporal_kind, out_path):
    device = torch.device("cpu")
    model = E2E_EMFv1(temporal_kind=temporal_kind).to(device)
    
    ckpt_path = os.path.join(ablation_code_dir, "emf_train_runs_seed2025_backbones", f"{temporal_kind}_60ep_s2025", "best_emf_v1.pt")
    if os.path.exists(ckpt_path):
        print(f"[INFO] Loading trained weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        # The checkpoint is a dict with 'model' key containing the actual state_dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.emfv1.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Weights loaded successfully for {temporal_kind}")
    else:
        print(f"[WARNING] Checkpoint not found at {ckpt_path}, exporting with random weights.")
        
    model.eval()
    
    samples_per_patch = model.waveform_to_mel.samples_per_patch
    dummy_input = torch.randn(1, samples_per_patch, device=device)
    
    # Export to ONNX
    print(f"[INFO] Exporting {temporal_kind} to {out_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["audio"],
        output_names=["probabilities"],
        dynamic_axes={
            "audio": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"[ONNX] Exported {temporal_kind} to {out_path}")
    
    # Validate ONNX model
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print(f"[ONNX] Model validation passed for {temporal_kind}.\n")

if __name__ == "__main__":
    out_dir = os.path.join(current_dir, "exported_models")
    os.makedirs(out_dir, exist_ok=True)
    
    for tk in ["cnn", "lstm", "resnet"]:
        out_path = os.path.join(out_dir, f"emfv1_{tk}_e2e.onnx")
        export_model(tk, out_path)
