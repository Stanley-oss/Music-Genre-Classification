import os
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from model_e2e import AudioResNetE2E, GENRES
from dataset_waveform import GTZANWaveformDataset, set_seed


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(args):
    train_dataset = GTZANWaveformDataset(
        data_dir=args.data_dir, split_txt_path=args.train_split,
        sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
        n_mels=args.n_mels, frames=args.frames, is_train=True,
    )
    val_dataset = GTZANWaveformDataset(
        data_dir=args.data_dir, split_txt_path=args.val_split,
        sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
        n_mels=args.n_mels, frames=args.frames, is_train=False,
    )
    test_dataset = None
    if args.test_split and os.path.exists(args.test_split):
        test_dataset = GTZANWaveformDataset(
            data_dir=args.data_dir, split_txt_path=args.test_split,
            sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
            n_mels=args.n_mels, frames=args.frames, is_train=False,
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False,
    )
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False,
        )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size
        
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_preds = [], []
    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
    return running_loss / total, correct / total, all_labels, all_preds


def export_onnx(model, samples_per_patch, out_path, device):
    model.eval()
    
    # 包装一层 Softmax，让前端直接拿到概率，无需再算
    class E2EWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return torch.softmax(self.m(x), dim=1)
    
    wrapper = E2EWrapper(model).to(device).eval()
    dummy_input = torch.randn(1, samples_per_patch, device=device)
    
    torch.onnx.export(
        wrapper,
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
    print(f"[ONNX] Exported to {out_path}")
    
    import onnx
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("[ONNX] Model validation passed.")


def train_model(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = build_loaders(args)
    print(f"[INFO] Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    model = AudioResNetE2E(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        frames=args.frames,
    ).to(device)
    
    samples_per_patch = model.waveform_to_mel.samples_per_patch
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.gamma, patience=args.patience
        )
    else:
        scheduler = None
        
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None
    start_epoch, best_acc = 1, 0.0
    
    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
    
    best_model_path = os.path.join(args.output_dir, "best_resnet18_gtzan.pth")
    last_ckpt_path = os.path.join(args.output_dir, "last_checkpoint.pth")
    onnx_path = os.path.join(args.output_dir, "gtzan_e2e.onnx")
    
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.amp
        )
        val_loss, val_acc, val_labels, val_preds = evaluate(
            model, val_loader, criterion, device
        )
        if scheduler:
            if args.scheduler == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        torch.save({
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "args": vars(args),
        }, last_ckpt_path)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            torch.save({
                "epoch": epoch,
                "best_acc": best_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            }, os.path.join(args.output_dir, "best_checkpoint.pth"))
            print(f"[INFO] New best val acc: {best_acc*100:.2f}%")
    
    print(f"[DONE] Best val acc: {best_acc*100:.2f}%")
    
    # 导出 ONNX（加载最佳权重）
    print("[INFO] Exporting ONNX...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    export_onnx(model, samples_per_patch, onnx_path, device)
    
    if test_loader:
        print("\n[INFO] Evaluating on test set...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc, test_labels, test_preds = evaluate(
            model, test_loader, criterion, device
        )
        print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc*100:.2f}%")
        print(classification_report(test_labels, test_preds, target_names=GENRES, digits=4))
        print(confusion_matrix(test_labels, test_preds))


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end GTZAN training + ONNX export")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--val_split", type=str, required=True)
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_e2e")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)