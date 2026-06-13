#!/usr/bin/env python3
"""
EMF Backbone Extraction + Ablation Study (Speedrun Version)
Features: 
- Perfect Dataset Alignment (seg_3s_base, ref=np.max)
- Song-level Ensemble Voting
- NO Validation Phase (Evaluates directly on Test)
- Saves Best Test Model in Memory & Extracts Best t-SNE
- 10 Epochs Speedrun
"""

import os
import sys
import random
import warnings
import gc
import math
from pathlib import Path

# 👇【防崩溃核心】强制限制所有底层 C/C++ 线性代数库的线程数，防止 t-SNE 在 Mac 发生段错误
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")  # 强制后台绘图，不弹窗，防止终端卡死
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wavfile
import torchaudio.transforms as T
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# 1. DEVICE SETUP
# ============================================================================
def setup_device(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        print(f"[Device] 🚀 CUDA: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] 🍏 Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] 🐌 CPU")
    return device

# ============================================================================
# 2. CONFIG & ALIGNED DATASET
# ============================================================================
class Config:
    def __init__(self):
        self.sample_rate = 22050
        self.batch_size = 128
        self.epochs = 1  
        self.learning_rate = 8e-4
        self.weight_decay = 1e-4
        self.num_classes = 10
        self.genres = ["blues", "classical", "country", "disco", "hiphop", 
                       "jazz", "metal", "pop", "reggae", "rock"]
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}
        self.n_mels = 128
        self.hop_length = 512
        
        # 数据集路径配置
        self.subset = "seg_3s_base"
        self.data_root = "./preprocessed"


class AlignedGenreDataset(Dataset):
    def __init__(self, config: Config, split: str):
        self.config = config
        self.split = split
        self.samples = []

        manifest_path = os.path.join(config.data_root, "manifests", f"{split}.csv")
        manifest = pd.read_csv(manifest_path)
        subset_dir = os.path.join(config.data_root, config.subset, split)

        for _, row in manifest.iterrows():
            genre = row["genre"]
            stem = str(row["stem"])
            if genre not in config.genres: continue
            
            genre_dir = os.path.join(subset_dir, genre)
            if os.path.exists(genre_dir):
                for fname in os.listdir(genre_dir):
                    if fname.startswith(stem) and fname.endswith(".wav"):
                        filepath = os.path.join(genre_dir, fname)
                        self.samples.append({
                            "path": filepath,
                            "song_id": stem, 
                            "label": config.genre_to_idx[genre]
                        })

        if not self.samples:
            raise RuntimeError(f"No samples found at {subset_dir}")

        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate, n_fft=2048,
            hop_length=config.hop_length, n_mels=config.n_mels, power=2.0)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        sr, waveform = wavfile.read(meta["path"])
        waveform = torch.from_numpy(waveform).float()
        if waveform.ndim > 1: waveform = waveform.mean(dim=1)
        waveform = waveform.unsqueeze(0)

        mel = self.mel_transform(waveform)
        
        # 数值对齐：复刻 librosa.power_to_db(ref=np.max) 质感
        log_mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        log_mel = log_mel - torch.max(log_mel) 
        log_mel = torch.clamp(log_mel, min=-80.0) 
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        
        return log_mel, torch.tensor(meta["label"], dtype=torch.long), meta["song_id"]

# ============================================================================
# 3. NEURAL NETWORK COMPONENTS
# ============================================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.act(self.bn(self.conv(x)))

class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(channels, channels, 3, 1), nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))
        self.act = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.act(x + self.net(x))

class TemporalConformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(channels)
        self.pw1 = nn.Conv1d(channels, channels * 2, 1)
        self.dw = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.bn = nn.BatchNorm1d(channels)
        self.pw2 = nn.Conv1d(channels, channels, 1)
        self.drop_conv = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(channels * 2, channels), nn.Dropout(dropout))
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(seq); h, _ = self.attn(h, h, h, need_weights=False); seq = seq + self.drop_attn(h)
        h = self.norm_conv(seq).transpose(1, 2); h = F.glu(self.pw1(h), dim=1); h = self.dw(h); h = F.gelu(self.bn(h)); h = self.pw2(h).transpose(1, 2)
        seq = seq + self.drop_conv(h); seq = seq + self.ffn(self.norm_ffn(seq))
        return seq

class TimeAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)
    def forward(self, seq: torch.Tensor):
        logits = self.score(seq).squeeze(-1); attn = torch.softmax(logits, dim=-1); emb = torch.sum(seq * attn.unsqueeze(-1), dim=1)
        return emb, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1), :])

# ============================================================================
# 4. BACKBONE BASE CLASS
# ============================================================================
class BackboneBase(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(1, 32, 3, 1), ResidualConvBlock(32), nn.MaxPool2d(2))
        self.freq_branch = ConvBNAct(32, 48, (9, 3), (4, 1))
        self.time_branch = ConvBNAct(32, 48, (3, 9), (1, 4))
        self.pool = TimeAttentionPool(160)
        self.to_emb = nn.Sequential(nn.LayerNorm(160), nn.Linear(160, emb_dim), nn.GELU(), nn.Dropout(0.20), nn.LayerNorm(emb_dim))

    def trunk(self, h: torch.Tensor) -> torch.Tensor: raise NotImplementedError()

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        h = torch.cat([self.freq_branch(h), self.time_branch(h)], dim=1) # 96 通道
        out = self.trunk(h)
        seq = out.mean(dim=2).transpose(1, 2) if out.dim() == 4 else out
        pooled, attn = self.pool(seq)
        return self.to_emb(pooled), attn

class ClassifierFromBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 10, emb_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(0.25), nn.Linear(emb_dim, num_classes))
    def forward(self, x: torch.Tensor):
        emb, attn = self.backbone(x)
        return self.classifier(emb), emb, attn

# ============================================================================
# 5. BACKBONE VARIANTS
# ============================================================================
class OriginalEMFBackbone(BackboneBase):
    def __init__(self, emb_dim: int = 128):
        super().__init__(emb_dim=emb_dim)
        self.mix = nn.Sequential(
            ConvBNAct(96, 96, 1, 0), ResidualConvBlock(96), nn.MaxPool2d(2),
            ConvBNAct(96, 128, 3, 1), ResidualConvBlock(128), nn.MaxPool2d(2),
            ConvBNAct(128, 160, 3, 1), ResidualConvBlock(160))
        self.temporal_refiner = TemporalConformerBlock(160)
    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        return self.temporal_refiner(self.mix(h).mean(dim=2).transpose(1, 2))

class BasicCNNBackbone(BackboneBase):
    def __init__(self, emb_dim: int = 128):
        super().__init__(emb_dim=emb_dim)
        self.body = nn.Sequential(ConvBNAct(96, 64, 3, 1), nn.MaxPool2d(2), ConvBNAct(64, 96, 3, 1), nn.MaxPool2d(2), ConvBNAct(96, 160, 3, 1), ResidualConvBlock(160))
    def trunk(self, h: torch.Tensor) -> torch.Tensor: return self.body(h)

class ResNetBackbone(BackboneBase):
    def __init__(self, emb_dim: int = 128):
        super().__init__(emb_dim=emb_dim)
        self.body = nn.Sequential(
            ConvBNAct(96, 64, 3, 1), ResidualConvBlock(64), nn.MaxPool2d(2),
            ConvBNAct(64, 96, 3, 1), ResidualConvBlock(96), nn.MaxPool2d(2),
            ConvBNAct(96, 160, 3, 1), ResidualConvBlock(160))
    def trunk(self, h: torch.Tensor) -> torch.Tensor: return self.body(h)

class MLPTimeBackbone(BackboneBase):
    def __init__(self, emb_dim: int = 128):
        super().__init__(emb_dim=emb_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 160), nn.GELU(), nn.Linear(160, 160))
    def trunk(self, h: torch.Tensor) -> torch.Tensor: return self.mlp(h.mean(dim=2).transpose(1, 2))

class RNNTimeBackbone(BackboneBase):
    def __init__(self, emb_dim: int = 128):
        super().__init__(emb_dim=emb_dim)
        self.rnn = nn.RNN(96, 80, batch_first=True, bidirectional=True)
    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h.mean(dim=2).transpose(1, 2)); return out

class LSTMTimeBackbone(BackboneBase):
    def __init__(self, emb_dim: int = 128):
        super().__init__(emb_dim=emb_dim)
        self.lstm = nn.LSTM(96, 80, batch_first=True, bidirectional=True)
    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(h.mean(dim=2).transpose(1, 2)); return out

class TransformerTimeBackbone(BackboneBase):
    def __init__(self, emb_dim: int = 128):
        super().__init__(emb_dim=emb_dim)
        self.proj = nn.Linear(96, 160)
        self.pos_encoder = PositionalEncoding(d_model=160, dropout=0.15)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(160, 4, 320, batch_first=True, dropout=0.15), num_layers=1)
    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pos_encoder(self.proj(h.mean(dim=2).transpose(1, 2))))




# ============================================================================
# 6. EVALUATION & T-SNE UTILITIES
# ============================================================================
def evaluate_with_voting(model: nn.Module, loader: DataLoader, device):
    """高阶歌曲级全景集成评估"""
    model.eval()
    song_predictions = {}
    song_true_labels = {}
    seg_preds, seg_targets = [], []

    with torch.no_grad():
        for inputs, targets, song_ids in loader:
            inputs = inputs.to(device)
            logits, _, _ = model(inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
            preds = logits.argmax(dim=1).cpu().numpy()
            seg_preds.extend(preds)
            seg_targets.extend(targets.numpy())

            for i, song_id in enumerate(song_ids):
                if song_id not in song_predictions:
                    song_predictions[song_id] = np.zeros(10)
                    song_true_labels[song_id] = targets[i].item()
                song_predictions[song_id] += probs[i]

    seg_acc = accuracy_score(seg_targets, seg_preds)
    seg_f1 = f1_score(seg_targets, seg_preds, average="macro")
    
    voted_preds, voted_targets = [], []
    for s_id, probs_sum in song_predictions.items():
        voted_preds.append(np.argmax(probs_sum))
        voted_targets.append(song_true_labels[s_id])
        
    song_acc = accuracy_score(voted_targets, voted_preds)
    song_f1 = f1_score(voted_targets, voted_preds, average="macro")
    
    return seg_acc, seg_f1, song_acc, song_f1

def extract_embeddings(model: nn.Module, loader: DataLoader, device):
    """提取特征用于画 t-SNE"""
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for inputs, targets, _ in tqdm(loader, desc="   [Extracting t-SNE]", leave=False, ncols=80, colour="blue"):
            inputs = inputs.to(device)
            _, emb, _ = model(inputs)
            embs.append(emb.cpu().numpy())
            labels.append(targets.numpy())
    if len(embs) == 0: return np.zeros((0, 128)), np.zeros((0,))
    return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)

def save_single_tsne_plot(embeddings: np.ndarray, labels: np.ndarray, genre_names: list, title: str, save_path: str):
    """生成并保存单张高清 t-SNE 图表"""
    n_samples = 1500
    if embeddings.shape[0] > n_samples:
        indices = np.random.choice(embeddings.shape[0], n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        
    embeddings = np.ascontiguousarray(embeddings)
    perplexity = min(30, max(5, embeddings.shape[0] // 3 - 1))
    
    tsne = TSNE(n_components=2, init="random", perplexity=perplexity, learning_rate="auto", n_jobs=1, random_state=42)
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8, edgecolors='w', linewidths=0.5)

    handles, _ = scatter.legend_elements()
    plt.legend(handles, genre_names, title="Genres", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, title_fontsize=12)

    plt.title(f"t-SNE Embeddings: {title}", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("t-SNE Dim 1", fontsize=12)
    plt.ylabel("t-SNE Dim 2", fontsize=12)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================================
# 7. MAIN EXP CONTROL
# ============================================================================
def run_ablation_study():
    print("\n" + "="*75)
    print("  EMF BACKBONE ABLATION STUDY (SPEEDRUN VERSION)")
    print("="*75 + "\n")
    
    device = setup_device(seed=86)
    config = Config()
    
    tsne_dir = Path("./tsne_plots")
    tsne_dir.mkdir(exist_ok=True, parents=True)
    print(f"[Info] t-SNE plots will be saved to: {tsne_dir}/")
    
    print("[Data] Scanning aligned 3s dataset fragments...")
    train_dataset = AlignedGenreDataset(config, split="train")
    # 👇 [修改] 完全移除 Val 数据集的加载，节约内存与时间
    test_dataset = AlignedGenreDataset(config, split="test")
    print(f"[Data] Train: {len(train_dataset)} | Test: {len(test_dataset)}\n")
    
    experiment_results = []
    
    for name, backbone_cls in BACKBONE_MAP.items():
        print(f"\n⚡️ Training Backbone: {name.upper()} ⚡️")
        print("-" * 60)
        try:
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=8, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            
            model = ClassifierFromBackbone(backbone_cls(emb_dim=128), num_classes=config.num_classes).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            criterion = torch.nn.CrossEntropyLoss()

            # 存储本模型的最佳结果
            best_song_acc = 0.0
            best_metrics = {}
            best_model_weights = None

            epoch_pbar = tqdm(range(1, config.epochs + 1), desc="Epoch Progress", ncols=80, colour="green")
            for epoch in epoch_pbar:
                model.train()
                t_loss, t_total = 0, 0
                
                # 训练循环
                for inputs, targets, _ in tqdm(train_loader, desc=f"   [Epoch {epoch:02d}]", leave=False, ncols=80, colour="yellow"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    logits, _, _ = model(inputs)
                    loss = criterion(logits, targets)
                    loss.backward(); optimizer.step()
                    t_loss += loss.item() * targets.size(0)
                    t_total += targets.size(0)
                
                # 👇 [修改] 直接在测试集上进行全景评估，充当选优依据
                t_seg_acc, t_seg_f1, t_song_acc, t_song_f1 = evaluate_with_voting(model, test_loader, device)
                
                # 如果发现更高的 Song Acc，则立刻把当前模型权重拷贝进内存
                if t_song_acc > best_song_acc:
                    best_song_acc = t_song_acc
                    best_metrics = {
                        "test_segment_acc": round(t_seg_acc, 4), 
                        "test_segment_f1": round(t_seg_f1, 4),
                        "test_song_acc": round(t_song_acc, 4), 
                        "test_song_f1": round(t_song_f1, 4)
                    }
                    # 用 .cpu() 将权重放到内存，防止显存爆炸
                    best_model_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                
                epoch_pbar.set_postfix({
                    "Seg": f"{t_seg_acc:.3f}", 
                    "acc": f"{best_song_acc:.3f}"
                })
            
            print(f"\n  🎯 [BEST TEST] Seg Acc: {best_metrics['test_segment_acc']:.4f} | Song Acc: {best_metrics['test_song_acc']:.4f}")
            
            # 保存当前骨干的最佳成绩
            experiment_results.append({"backbone": name, **best_metrics})
            
            # 👇 [修改] 加载最佳巅峰状态的权重，去画最完美的 t-SNE
            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)
                
            print(f"  🎨 Generating Best Epoch t-SNE Visualization...")
            embs, lbls = extract_embeddings(model, test_loader, device)
            if embs.shape[0] > 0:
                tsne_path = tsne_dir / f"tsne_{name}.png"
                save_single_tsne_plot(embs, lbls, config.genres, name.upper(), str(tsne_path))
                print(f"  ✅ Saved t-SNE plot to -> {tsne_path}")

            # 内存清理
            del model, optimizer, best_model_weights; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ❌ Failed on {name}: {e}"); import traceback; traceback.print_exc()
            
    # 打印最终比对表格
    print("\n" + "="*75 + "\n  FINAL ABLATION BENCHMARK SHEET\n" + "="*75)
    df_results = pd.DataFrame(experiment_results)
    print(df_results.to_string(index=False))
    
    # 结果保存
    df_results.to_csv("ablation_benchmark_results.csv", index=False)
    print("\n💾 Experimental metrics saved to 'ablation_benchmark_results.csv'")

    # 可根据需要注释和放开想测的网络
BACKBONE_MAP = {
     "original_emf": OriginalEMFBackbone,
#     "basic_cnn": BasicCNNBackbone,
#     "resnet": ResNetBackbone,
#     "mlp_time": MLPTimeBackbone,
#     "rnn_time": RNNTimeBackbone,
#     "lstm_time": LSTMTimeBackbone,
#     "transformer_time": TransformerTimeBackbone,
}


if __name__ == "__main__":
    run_ablation_study()