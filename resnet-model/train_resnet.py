import os
import time
import argparse
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset import GTZANSturmDataset


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


class AudioResNet(nn.Module):
    """
    基于 ResNet-18 的音频分类模型。

    输入:
        x:[B, 1, 128, 96]

    输出:
        logits: [B, num_classes]

    注意:
        GTZAN 是单标签多分类任务，因此 forward 不加 sigmoid / softmax。
        训练时直接使用 nn.CrossEntropyLoss。
    """

    def __init__(self, num_classes=10, pretrained=False, dropout=0.0):
        super().__init__()

        # torchvision 新版本推荐使用 weights 参数
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.resnet = models.resnet18(weights=weights)
            except Exception:
                self.resnet = models.resnet18(pretrained=True)
        else:
            try:
                self.resnet = models.resnet18(weights=None)
            except Exception:
                self.resnet = models.resnet18(pretrained=False)

        # 原始 ResNet 输入是 RGB 三通道，这里改成单通道 Mel 频谱图
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,  # 修复 1：补齐缺失的逗号
            stride=2,
            padding=3,
            bias=False,
        )

        num_ftrs = self.resnet.fc.in_features

        if dropout > 0:
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(num_ftrs, num_classes)
            )
        else:
            self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        logits = self.resnet(x)
        return logits


def set_seed(seed=42):
    """
    固定随机种子，增强可复现性。
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 可复现设置
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device():
    """
    自动选择运行设备。
    优先级:
        CUDA > MPS > CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # 修复 2：补齐缺失的括号和引号

    return torch.device("cpu")


def build_loaders(args):
    """
    构建 train / val / test DataLoader。
    """

    train_dataset = GTZANSturmDataset(
        data_dir=args.data_dir,
        split_txt_path=args.train_split,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        frames=args.frames,
        is_train=True,
    )

    val_dataset = GTZANSturmDataset(
        data_dir=args.data_dir,
        split_txt_path=args.val_split,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        frames=args.frames,
        is_train=False,
    )

    test_dataset = None
    if args.test_split is not None and os.path.exists(args.test_split):
        test_dataset = GTZANSturmDataset(
            data_dir=args.data_dir,
            split_txt_path=args.test_split,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            frames=args.frames,
            is_train=False,  # 修复 3：补齐变量名 is_train
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler=None, use_amp=False
):
    """
    单个 epoch 的训练过程。
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    验证或测试过程。
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(
            device, non_blocking=True
        )  # 修复 4：补齐等号左侧被截断的 labels 变量

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += batch_size

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_labels, all_preds


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, args):
    """
    保存完整 checkpoint，方便恢复训练。
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)  # 修复 5：分离被合并的两行代码，补齐右括号


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    加载 checkpoint。
    """

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)

    return start_epoch, best_acc


def train_model(args):
    """
    主训练函数。
    """

    set_seed(args.seed)

    device = get_device()
    print(f"[INFO] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(args)

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches:   {len(val_loader)}")

    if test_loader is not None:
        print(f"[INFO] Test batches:  {len(test_loader)}")

    model = AudioResNet(
        num_classes=args.num_classes, pretrained=args.pretrained, dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

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

    start_epoch = 1
    best_acc = 0.0

    if args.resume is not None:  # 修复 6：补充丢失的 is 关键字
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=device
        )

    best_model_path = os.path.join(args.output_dir, "best_resnet18_gtzan.pth")
    last_ckpt_path = os.path.join(args.output_dir, "last_checkpoint.pth")

    print("[INFO] Start training...")
    print(f"[INFO] Best model will be saved to: {best_model_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=args.amp,
        )

        val_loss, val_acc, val_labels, val_preds = evaluate(
            model=model, loader=val_loader, criterion=criterion, device=device
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()  # 修复 7：scheduler() 语法错误，应为 scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # 保存 last checkpoint
        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_acc=best_acc,
            args=args,
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc

            torch.save(model.state_dict(), best_model_path)

            save_checkpoint(
                path=os.path.join(args.output_dir, "best_checkpoint.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,  # 修复 8：补齐参数列表缺失的逗号
                epoch=epoch,
                best_acc=best_acc,
                args=args,
            )

            print(f"[INFO] New best val acc: {best_acc * 100:.2f}%, model saved.")

    print(f"[DONE] Training finished. Best val acc: {best_acc * 100:.2f}%")

    # 如果有 test_split，则加载最佳模型进行最终测试
    if test_loader is not None:
        print("\n[INFO] Evaluating best model on test set...")

        model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_loss, test_acc, test_labels, test_preds = evaluate(
            model=model, loader=test_loader, criterion=criterion, device=device
        )

        print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc * 100:.2f}%")

        print("\n[TEST] Classification Report:")
        print(
            classification_report(
                test_labels, test_preds, target_names=GENRES, digits=4
            )
        )

        print("[TEST] Confusion Matrix:")
        print(confusion_matrix(test_labels, test_preds))


def parse_args():
    parser = argparse.ArgumentParser(  # 修复 9：修复 parser 行缩进错误
        description="Train ResNet-18 on GTZAN Mel Spectrograms"
    )

    # 数据路径
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="GTZAN audio root directory, e.g. ./gtzan_dataset/genres_original",
    )

    parser.add_argument(
        "--train_split", type=str, required=True, help="Path to train split txt"
    )

    parser.add_argument(
        "--val_split", type=str, required=True, help="Path to validation split txt"
    )

    parser.add_argument(
        "--test_split", type=str, default=None, help="Optional path to test split txt"
    )

    # 音频特征参数，需要和 Dataset 保持一致
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=96)

    # 训练
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step", "plateau"],
    )

    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained ResNet-18 backbone. Usually False for audio from scratch.",
    )

    parser.add_argument(
        "--amp", action="store_true", help="Use mixed precision training on CUDA"
    )

    parser.add_argument(
        "--pin_memory", action="store_true", help="Use pin_memory in DataLoader"
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="./checkpoints_resnet18")

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
