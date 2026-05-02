### 一、 核心管线流程 (The Pipeline)

整个系统由四个标准的工业级模块串联而成：

1. **防泄露数据划分 (Data Splitting):** 传统的随机划分会导致模型学习“特定歌手的音色”而非“流派的音乐性”。我们通过读取 CSV 元数据，以“歌手/乐队 (Artist)”为最小隔离单元，确保同一个艺术家的歌绝对不会跨界出现在训练集和测试集中。
2. **声学特征提取 (DSP & Dataloading):** 放弃原始的一维波形，将音频读取并重采样为固定的 $22050\text{ Hz}$，通过短时傅里叶变换 (STFT) 和梅尔滤波器组，提取出符合人类听觉心理学的 **对数梅尔频谱图 (Log-Mel Spectrogram)**。为了适应网络，将其规范化为 `[1, 128, 96]` (通道, 频带, 帧数) 的二维张量（切片时长约 2.23 秒）。
3. **残差网络训练 (Model Training):** 采用经典的 ResNet-18 骨干网络。为适配音频任务，将输入层的 `3x3x3` RGB 卷积核魔改为单通道卷积核。全程使用 AMP (混合精度) 加速训练，配合 Cosine 学习率退火，在防泄露数据集上跑出真实的高泛化能力权重。
4. **流式推断评估 (Streaming Inference):** 面对一首完整的 30 秒未知歌曲，不进行随机裁剪，而是严格按照滑动窗口（Sliding Window）将其切割成多个 2.23 秒的连续切片。模型对所有切片进行独立预测，最后通过全局平均池化（Average Pooling）得出最稳健的最终流派判定。

---

### 二、 核心文件用途与调用参数字典

#### 1. `generate_gtzan_artist_split.py` (防泄露分割器)
*   **用途：** 根据 `GTZAN_SONGTITLE_ARTIST.csv`，将 1000 首歌曲划分为安全的 `train.txt`、`val.txt` 和 `test.txt`，保存在 `splits_sturm_safe` 目录中。
*   **调用参数：**
    *   `--metadata_csv`: (必填) 元数据 CSV 的路径。
    *   `--out_dir`: (必填) 生成的三份 `.txt` 和报告的输出目录。
    *   `--audio_root`: (可选) 音频原文件的根目录，用于校验文件是否真存在。
    *   `--drop_missing`: 配合上个参数，若音频丢失则在列表中剔除。
    *   `--group_by`: 隔离策略，默认 `artist`，可选 `artist_title` 或 `title`。
    *   `--train_ratio` / `--val_ratio` / `--test_ratio`: 划分比例，默认 `0.7 / 0.15 / 0.15`。
    *   `--seed`: 随机种子，默认 `42`。
    *   `--sturm_train` / `_val` / `_test`: (可选) 强行传入学术界原版 Sturm 分割列表作校验。

#### 2. `dataset.py` (数据加载与预处理引擎)
*   **用途：** 包含 `GTZANSturmDataset` 类。负责音频截取、坏文件跳过（防崩溃容错）、提取 Mel 频谱、Z-score 归一化。它也可作为独立脚本执行，用来验证数据链路是否畅通、GPU 是否能拿到正确的 Tensor。
*   **调用参数 (直接运行测试时)：**
    *   `--data_dir`: (必填) 音频文件夹根目录 (`genres_original`)。
    *   `--split_txt`: (必填) 用来测试的分割列表路径 (如 `train.txt`)。
    *   `--mode`: 运行模式，`train`, `val`, 或 `test`。
    *   `--batch_size` / `--num_workers`: 批次大小 (默认 16) 与数据加载线程数 (默认 4)。
    *   `--sr` / `--n_fft` / `--hop_length` / `--n_mels` / `--frames`: DSP 特征参数。默认依次为 `22050, 2048, 512, 128, 96`。
    *   `--seed`: 随机种子。

#### 3. `train_resnet.py` (模型训练主轴)
*   **用途：** 实例化 `AudioResNet`，绑定 DataLoader，执行前向/反向传播，将性能最好的模型保存至 `checkpoints_resnet18/best_resnet18_gtzan.pth`，并支持断点续训。
*   **调用参数：**
    *   **路径类**：`--data_dir`, `--train_split`, `--val_split` (必填)；`--test_split` (可选)；`--output_dir` (保存目录)；`--resume` (断点 `.pth` 路径)。
    *   **DSP类**：`--sr`, `--n_fft`, `--hop_length`, `--n_mels`, `--frames` (必须与 dataset 保持一致)。
    *   **训练超参**：`--num_classes` (默认 10)，`--epochs` (默认 50)，`--batch_size` (默认 64)，`--lr` (默认 1e-3)，`--weight_decay` (默认 1e-4)。
    *   **调度与正则化**：`--scheduler` (`cosine`, `step`, `plateau`)，`--dropout` (默认 0.2)，`--label_smoothing` (默认 0.0)。
    *   **工程优化**：`--amp` (开启混合精度加速)，`--pin_memory` (开启显存锁页加速)，`--pretrained` (是否使用 ImageNet 预训练权重)。

#### 4. `infer_resnet.py` (推理与评估脚本)
*   **用途：** 针对任意一段音乐，使用滑动窗口切片，调用训练好的 `.pth` 模型进行预测，打印 Top-K 的流派置信度。
*   **调用参数：**
    *   `--audio_path`: (必填) 测试音乐 `.wav` 的路径。
    *   `--model_path`: (默认 `checkpoints.../best...pth`) 模型权重路径。
    *   **DSP类**：`--sr` 等 5 个音频特征参数 (必须与训练时绝对对齐)。
    *   **切片控制**：`--overlap` (切片重叠率，默认 0.0)，`--no_include_last` (丢弃不足 2 秒的尾部音频)。
    *   **网络结构映射**：`--dropout` (非常关键！必须填训练时的值，默认需传 `0.2`，否则 state_dict 字典会因 `Sequential` 结构对不上号而报错)。
    *   **环境与展示**：`--device` (`cuda` / `cpu` / `mps`)，`--batch_size` (推理批次)，`--top_k` (展示前几名概率)。

---

### 三、 跑通全管线的一键示例命令

假设你刚拉取完代码，处于根目录下，请依次执行以下 4 条命令（完全适配拥有 GPU 环境的机器）：

#### Step 1: 制造防泄露划分名单
```bash
python generate_gtzan_artist_split.py \
    --metadata_csv ./gtzan_dataset/GTZAN_SONGTITLE_ARTIST.csv \
    --out_dir ./splits_sturm_safe \
    --audio_root ./gtzan_dataset/genres_original \
    --group_by artist
```

#### Step 2: (可选) 测试数据管道连通性
```bash
python dataset.py \
    --data_dir ./gtzan_dataset/genres_original \
    --split_txt ./splits_sturm_safe/train.txt \
    --batch_size 16
```

#### Step 3: 启动混合精度全速训练
```bash
python train_resnet.py \
    --data_dir ./gtzan_dataset/genres_original \
    --train_split ./splits_sturm_safe/train.txt \
    --val_split ./splits_sturm_safe/val.txt \
    --test_split ./splits_sturm_safe/test.txt \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.2 \
    --amp
```

#### Step 4: 测试任意一首音乐的流派
```bash
python infer_resnet.py \
    --audio_path ./gtzan_dataset/genres_original/metal/metal.00000.wav \
    --model_path ./checkpoints_resnet18/best_resnet18_gtzan.pth \
    --device cuda \
    --dropout 0.2
```
