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

class E2E_Unified(nn.Module):
    def __init__(self, temporal_kind="cnn", sr=22050, n_fft=2048, hop_length=512, n_mels=128, frames=96):
        super().__init__()
        self.waveform_to_mel = WaveformToMelExact(sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, frames=frames)
        self.emfv1 = EMFv1(num_classes=10, temporal_kind=temporal_kind, use_denoising=True)
        
    def forward(self, x: torch.Tensor):
        mel = self.waveform_to_mel(x)
        
        h = self.emfv1.stem(mel)
        h_freq = self.emfv1.freq_branch(h)
        h_time = self.emfv1.time_branch(h)
        
        # mean over channels for visualization
        freq_map = h_freq.mean(dim=1, keepdim=True)
        time_map = h_time.mean(dim=1, keepdim=True)
        
        h_concat = torch.cat([h_freq, h_time], dim=1)
        
        if self.emfv1.temporal_kind in {"mlp", "lstm", "rnn", "transformer"}:
            seq = self.emfv1.temporal_refiner(h_concat)
        elif self.emfv1.temporal_kind in {"cnn", "resnet"}:
            fmap = self.emfv1.temporal_refiner(h_concat)
            seq = fmap.mean(dim=2).transpose(1, 2)
        else:
            fmap = self.emfv1.mix(h_concat)
            seq = fmap.mean(dim=2).transpose(1, 2)
            seq = self.emfv1.temporal_refiner(seq)
            
        pooled, attn = self.emfv1.pool(seq)
        clean_emb = self.emfv1.to_emb(pooled)
        clean_logits = self.emfv1.classify(clean_emb)
        clean_probs = torch.softmax(clean_logits, dim=1)
        
        # Add noise
        t_val = torch.tensor([[0.55]], device=x.device)
        t_val = t_val.expand(clean_emb.size(0), -1)
        eps = torch.randn_like(clean_emb)
        noisy_emb = t_val * clean_emb + (1.0 - t_val) * eps
        noisy_logits = self.emfv1.classify(noisy_emb)
        noisy_probs = torch.softmax(noisy_logits, dim=1)
        
        # Denoise
        denoised_emb = self.emfv1.denoise(noisy_emb, t_val)
        denoised_logits = self.emfv1.classify(denoised_emb)
        denoised_probs = torch.softmax(denoised_logits, dim=1)
        
        return clean_probs, mel, freq_map, time_map, clean_emb, noisy_emb, denoised_emb, clean_probs, noisy_probs, denoised_probs

def export_model(temporal_kind, out_path):
    device = torch.device("cpu")
    model = E2E_Unified(temporal_kind=temporal_kind).to(device)
    
    ckpt_path = os.path.join(ablation_code_dir, "emf_train_runs_seed2025_backbones", f"{temporal_kind}_60ep_s2025", "best_emf_v1.pt")
    if os.path.exists(ckpt_path):
        print(f"[INFO] Loading trained weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
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
    
    names = [
        "probabilities", "mel", "freq_map", "time_map", 
        "clean_emb", "noisy_emb", "denoised_emb", 
        "clean_probs", "noisy_probs", "denoised_probs"
    ]
    dynamic = {name: {0: "batch_size"} for name in ["audio"] + names}
    
    print(f"[INFO] Exporting {temporal_kind} to {out_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["audio"],
        output_names=names,
        dynamic_axes=dynamic,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"[ONNX] Exported {temporal_kind} to {out_path}")
    
    onnx_model = onnx.load(out_path, load_external_data=True)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, out_path)
    
    data_file = out_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
        print(f"[ONNX] Cleaned up external data file: {data_file}")
    
    print(f"[ONNX] Model validation passed for {temporal_kind}.\n")

if __name__ == "__main__":
    out_dir = os.path.join(current_dir, "exported_models")
    os.makedirs(out_dir, exist_ok=True)
    
    for tk in ["cnn", "lstm", "resnet"]:
        out_path = os.path.join(out_dir, f"emfv1_{tk}_e2e.onnx")
        export_model(tk, out_path)
