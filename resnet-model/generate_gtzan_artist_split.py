import os
import re
import csv
import argparse
import random
from collections import defaultdict, Counter


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


def normalize_text(text):
    """
    用于规范化 artistName / songTitle。
    避免大小写、空格、标点差异导致同一艺术家被当成不同 group。
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("&", "and")

    # 保留字母、数字和空格，去掉大部分标点
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def genre_from_ref(ref):
    """
    GTZAN 文件名格式:
        blues.00000.wav
        rock.00015.wav
    """
    return ref.split(".")[0]


def read_metadata_csv(csv_path, group_by="artist"):
    """
    读取 GTZAN metadata CSV。

    需要至少包含:
        ref, genre, artistName

    可选:
        songTitle
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    items = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {"ref", "genre", "artistName"}
        missing_cols = required_cols - set(reader.fieldnames)

        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}. "
                f"Current columns are: {reader.fieldnames}"
            )

        for row in reader:
            ref = row["ref"].strip()
            # 修复 1：调用正确的 genre_from_ref
            genre = row["genre"].strip() if row.get("genre") else genre_from_ref(ref)
            # 修复 2：补齐缺失的括号和引号
            artist = row.get("artistName", "").strip()
            title = row.get("songTitle", "").strip()

            if not ref:
                continue

            if genre not in GENRES:
                raise ValueError(f"Unknown genre '{genre}' for ref '{ref}'")

            norm_artist = normalize_text(artist)
            norm_title = normalize_text(title)

            if group_by == "artist":
                if norm_artist:
                    group_id = f"artist::{norm_artist}"
                else:
                    # 如果 artist 缺失，不要把所有缺失 artist 的歌放到一个 group
                    group_id = f"unknown_artist::{ref}"

            elif group_by == "artist_title":
                if norm_artist or norm_title:
                    group_id = f"artist_title::{norm_artist}::{norm_title}"
                else:
                    group_id = f"unknown_artist_title::{ref}"

            elif group_by == "title":
                if norm_title:
                    group_id = f"title::{norm_title}"
                else:
                    group_id = f"unknown_title::{ref}"

            else:
                raise ValueError(f"Unsupported group_by: {group_by}")

            items.append(
                {
                    "ref": ref,
                    "genre": genre,
                    "artistName": artist,
                    "songTitle": title,
                    "group_id": group_id,
                }
            )

    return items


def check_audio_exists(items, audio_root, drop_missing=False):
    """
    检查音频文件是否存在。

    GTZAN 默认结构:
        audio_root/blues/blues.00000.wav
        audio_root/rock/rock.00001.wav
    """

    if audio_root is None:
        return items

    kept = []
    missing = []

    for item in items:
        ref = item["ref"]
        genre = item["genre"]
        path = os.path.join(audio_root, genre, ref)

        if os.path.exists(path):
            kept.append(item)
        else:
            missing.append(path)

    if missing:
        print("\n[WARNING] Missing audio files:")
        for p in missing[:20]:
            print(f"  {p}")

        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

        if drop_missing:
            print(f"[INFO] Drop missing files: {len(missing)}")
            return kept
        else:
            raise FileNotFoundError(
                f"{len(missing)} audio files are missing. "
                f"Use --drop_missing to ignore them."
            )

    return kept


def build_groups(items):
    """
    根据 group_id 构建 group。

    一个 group 代表一个 artist，或者 artist-title。
    同一个 group 不能跨 train / val / test。
    """

    groups = {}

    for item in items:
        gid = item["group_id"]

        if gid not in groups:
            groups[gid] = {
                "group_id": gid,
                "refs": [],
                "genres": [],
                "items": [],
                "genre_counter": Counter(),
            }

        groups[gid]["refs"].append(item["ref"])
        groups[gid]["genres"].append(item["genre"])
        groups[gid]["items"].append(item)
        groups[gid]["genre_counter"][item["genre"]] += 1

    return list(groups.values())


def read_split_txt(path):
    """
    读取 split txt，每行一个文件名:
        blues.00000.wav
    """
    if path is None:
        return set()

    # 修复 3：补齐缺失的引号并调整错误缩进
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")

    refs = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ref = line.strip()
            if not ref:
                continue
            ref = ref.split()[0]
            refs.add(ref)

    return refs


def load_sturm_assignment(train_path, val_path, test_path):
    """
    读取 Sturm split，返回:
        ref_to_split[ref] = "train" / "val" / "test"
    """

    split_files = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }

    ref_to_split = {}

    for split_name, path in split_files.items():
        refs = read_split_txt(path)

        for ref in refs:
            if ref in ref_to_split:
                raise ValueError(
                    f"File '{ref}' appears in multiple Sturm splits: "
                    f"{ref_to_split[ref]} and {split_name}"
                )
            ref_to_split[ref] = split_name

    return ref_to_split


def assign_groups_by_sturm_majority(groups, ref_to_split):
    """
    以 Sturm split 为初始依据，但保证 group跨集合。

    对每个 artist group:
        统计它的歌曲在 Sturm train / val / test 中分别有多少；
        放入数量最多的 split。

    如果某个 group 中的文件完全不在 Sturm split 中，则暂时不分配。
    """

    assignments = {}
    unassigned_groups = []

    leakage_before = []

    for group in groups:
        split_counter = Counter()

        for ref in group["refs"]:
            if ref in ref_to_split:
                split_counter[ref_to_split[ref]] += 1

        if len(split_counter) == 0:
            unassigned_groups.append(group)
            continue

        if len(split_counter) > 1:
            leakage_before.append(
                (group["group_id"], dict(split_counter), group["refs"])
            )

        # 出现最多的 split
        # 如果数量相同，优先级 train > val > test，可按需要改
        priority = {"train": 0, "val": 1, "test": 2}

        best_split = sorted(
            split_counter.items(), key=lambda x: (-x[1], priority[x[0]])
        )[0][0]

        # 修复 4：补齐缺失的右括号和引号
        assignments[group["group_id"]] = best_split

    return assignments, unassigned_groups, leakage_before


def compute_global_counts(groups):
    total = 0
    genre_counter = Counter()

    for group in groups:
        total += len(group["refs"])
        genre_counter.update(group["genre_counter"])

    return total, genre_counter


def initialize_split_counters(groups, assignments):
    split_total = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    split_genre_counter = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    group_dict = {g["group_id"]: g for g in groups}

    for gid, split_name in assignments.items():
        group = group_dict[gid]
        n = len(group["refs"])

        split_total[split_name] += n
        split_genre_counter[split_name].update(group["genre_counter"])

    return split_total, split_genre_counter


def greedy_assign_groups(groups, ratios, existing_assignments=None, seed=42):
    """
    group-aware stratified greedy split。

    目标:
        1. 同一个 group 不跨集合；
        2. train / val / test 比例尽量接近设定比例；
        3. 每个 genre 在三个集合中的分布尽量接近整体比例。
    """

    if existing_assignments is None:
        existing_assignments = {}

    rng = random.Random(seed)

    total_n, global_genre_counter = compute_global_counts(groups)

    split_names = ["train", "val", "test"]

    target_total = {split: total_n * ratios[split] for split in split_names}

    target_genre = {
        split: {genre: global_genre_counter[genre] * ratios[split] for genre in GENRES}
        for split in split_names
    }

    assignments = dict(existing_assignments)

    split_total, split_genre_counter = initialize_split_counters(groups, assignments)

    assigned_gids = set(assignments.keys())

    remaining_groups = [g for g in groups if g["group_id"] not in assigned_gids]

    # 大 group 先分配减少后期比例失控
    rng.shuffle(remaining_groups)
    remaining_groups.sort(key=lambda g: len(g["refs"]), reverse=True)

    def score_if_assign(group, split):
        """
        计算把 group 放入某个 split 后的偏差。
        改为按“目前容量相对占比（Fill Ratio）”计算，确保优先填满最空闲的集合。
        """
        new_total = split_total[split] + len(group["refs"])
        # 计算放入后，该集合总数占其目标容量的比例
        total_ratio = new_total / (target_total[split] + 1e-6)

        genre_ratio = 0.0
        for genre in GENRES:
            new_g_count = (
                split_genre_counter[split][genre] + group["genre_counter"][genre]
            )
            genre_ratio += new_g_count / (target_genre[split][genre] + 1e-6)

        # 比例越小，说明该集合越“缺”数据，越应该分配给它
        return total_ratio + genre_ratio

    # 之前报错是因为下面这段循环调用和 return 被你不小心覆盖掉了
    for group in remaining_groups:
        candidate_scores = []

        for split in split_names:
            s = score_if_assign(group, split)
            candidate_scores.append((s, split))

        # 找偏离得分最小的集合放入
        candidate_scores.sort(key=lambda x: x[0])
        best_split = candidate_scores[0][1]

        assignments[group["group_id"]] = best_split
        split_total[best_split] += len(group["refs"])
        split_genre_counter[best_split].update(group["genre_counter"])

    return assignments


def build_ref_splits(groups, assignments):
    """
    根据 group assignment 生成 ref-level split。
    """

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for group in groups:
        gid = group["group_id"]
        split = assignments[gid]

        for ref in group["refs"]:
            splits[split].append(ref)

    for split in splits:
        splits[split] = sorted(splits[split])

    return splits


def write_split_files(splits, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for split_name, refs in splits.items():
        path = os.path.join(out_dir, f"{split_name}.txt")

        with open(path, "w", encoding="utf-8") as f:
            for ref in refs:
                f.write(ref + "\n")

        # 修复 6：补齐 f-string 中大括号缺失
        print(f"[INFO] Wrote {split_name}: {path} n = {len(refs)}")


def check_no_ref_overlap(splits):
    ref_to_split = {}

    for split_name, refs in splits.items():
        for ref in refs:
            if ref in ref_to_split:
                raise RuntimeError(
                    f"Ref leakage: {ref} appears in both "
                    f"{ref_to_split[ref]} and {split_name}"
                )
            ref_to_split[ref] = split_name

    print("[OK] No file-level overlap among train / val / test.")


def check_no_group_leakage(groups, assignments):
    """
    检查 group 是否跨 split。
    由于我们的 assignments 是 group-level，理论上不会出现泄漏。
    """

    group_to_split = {}

    for group in groups:
        gid = group["group_id"]
        split = assignments[gid]

        if gid in group_to_split and group_to_split[gid] != split:
            raise RuntimeError(
                f"Group leakage: {gid} appears in both "
                f"{group_to_split[gid]} and {split}"
            )

        group_to_split[gid] = split

    print("[OK] No group-level leakage among train / val / test.")


# 修复 7：补齐函数开头的 def
def print_distribution(items, splits):
    """
    打印每个 split 的 genre 分布。
    """

    ref_to_genre = {item["ref"]: item["genre"] for item in items}

    print("\n========== Split distribution ==========")

    for split_name in ["train", "val", "test"]:
        refs = splits[split_name]
        counter = Counter(ref_to_genre[ref] for ref in refs)

        print(f"\n[{split_name}]")
        print(f"Total: {len(refs)}")

        for genre in GENRES:
            print(f"  {genre:10s}: {counter[genre]}")

    print("========================================\n")


def write_report(out_dir, items, groups, assignments, leakage_before=None):
    """
    保存一个报告文件，方便论文实验记录。
    """

    report_path = os.path.join(out_dir, "split_report.txt")

    group_to_items = {group["group_id"]: group["items"] for group in groups}

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("GTZAN artist-aware split\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Number of tracks: {len(items)}\n")
        f.write(f"Number of groups: {len(groups)}\n\n")

        split_group_counter = Counter(assignments.values())

        f.write("Number of groups per split:\n")
        for split in ["train", "val", "test"]:
            f.write(f"  {split}: {split_group_counter[split]}\n")

        f.write("\n")

        if leakage_before:
            f.write("Groups crossing original Sturm splits before correction:\n")
            f.write("-" * 80 + "\n")

            for gid, split_counter, refs in leakage_before:
                f.write(f"group_id: {gid}\n")
                f.write(f"original_sturm_counts: {split_counter}\n")
                f.write(f"refs: {', '.join(refs)}\n")
                f.write("\n")
        else:
            f.write(
                "No group leakage detected in original Sturm split, or no Sturm split provided.\n\n"
            )

        f.write("\nDetailed group assignments:\n")
        f.write("-" * 80 + "\n")

        # 修复 8：补齐 in
        for gid in sorted(assignments.keys()):
            split = assignments[gid]
            group_items = group_to_items[gid]

            artists = sorted(set(item["artistName"] for item in group_items))
            refs = sorted(item["ref"] for item in group_items)

            f.write(f"group_id: {gid}\n")
            f.write(f"split: {split}\n")
            f.write(f"artists: {artists}\n")
            f.write(f"refs: {', '.join(refs)}\n")
            f.write("\n")

    print(f"[INFO] Wrote report: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GTZAN Sturm-like artist-aware split files."
    )

    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="CSV file with columns: ref, genre, songTitle, artistName",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for train.txt, val.txt, test.txt",
    )

    parser.add_argument(
        "--audio_root",
        type=str,
        default=None,
        help="Optional GTZAN root, e.g. /path/to/genres_original",
    )

    parser.add_argument(
        "--drop_missing",
        action="store_true",
        help="Drop rows whose audio files do not exist under audio_root",
    )

    parser.add_argument(
        "--group_by",
        type=str,
        default="artist",
        choices=["artist", "artist_title", "title"],
        help=(
            "Group key for leakage prevention. "
            "artist is recommended for GTZAN performer leakage."
        ),
    )

    parser.add_argument("--train_ratio", type=float, default=0.7)

    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--sturm_train",
        type=str,
        default=None,
        help="Optional original Sturm train split txt",
    )

    parser.add_argument(
        "--sturm_val", type=str, default=None, help="Optional Sturm val split txt"
    )

    parser.add_argument(
        "--sturm_test",
        type=str,
        default=None,
        help="Optional original Sturm test split txt",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }

    print("[INFO] Reading metadata CSV...")
    items = read_metadata_csv(args.metadata_csv, group_by=args.group_by)

    print(f"[INFO] Loaded metadata rows: {len(items)}")

    if args.audio_root is not None:
        print("[INFO] Checking audio files...")
        items = check_audio_exists(
            items, audio_root=args.audio_root, drop_missing=args.drop_missing
        )

    groups = build_groups(items)

    # 修复 9：补齐变量名中的 args.
    print(f"[INFO] Number of groups by {args.group_by}: {len(groups)}")

    use_sturm = (
        args.sturm_train is not None
        or args.sturm_val is not None
        or args.sturm_test is not None
    )

    leakage_before = []

    if use_sturm:
        if not (args.sturm_train and args.sturm_val and args.sturm_test):
            raise ValueError(
                "If using Sturm split, you must provide all of "
                "--sturm_train, --sturm_val, --sturm_test"
            )

        print("[INFO] Loading original Sturm split...")
        ref_to_sturm_split = load_sturm_assignment(
            args.sturm_train, args.sturm_val, args.sturm_test
        )

        print(f"[INFO] Number of refs in Sturm split: {len(ref_to_sturm_split)}")

        print("[INFO] Assigning groups by Sturm majority...")
        initial_assignments, unassigned_groups, leakage_before = (
            assign_groups_by_sturm_majority(groups, ref_to_sturm_split)
        )

        print(f"[INFO] Initially assigned groups by Sturm: {len(initial_assignments)}")
        # 修复 10：补齐 { 左侧大括号
        print(f"[INFO] Unassigned groups not found in Sturm: {len(unassigned_groups)}")

        if leakage_before:
            print(
                f"[WARNING] Detected {len(leakage_before)} groups crossing original Sturm splits. "
                f"They will be collapsed into a single split by majority rule."
            )

        print("[INFO] Greedily assigning remaining groups...")
        assignments = greedy_assign_groups(
            groups,
            ratios=ratios,
            existing_assignments=initial_assignments,
            seed=args.seed,
        )

    else:
        print("[INFO] No Sturm split provided.")
        print("[INFO] Generating artist-aware stratified split from metadata CSV...")
        assignments = greedy_assign_groups(
            groups, ratios=ratios, existing_assignments=None, seed=args.seed
        )

    splits = build_ref_splits(groups, assignments)

    check_no_ref_overlap(splits)
    check_no_group_leakage(groups, assignments)

    write_split_files(splits, args.out_dir)
    print_distribution(items, splits)
    write_report(
        args.out_dir, items, groups, assignments, leakage_before=leakage_before
    )

    print("[DONE] Split generation finished.")


# 修复 11：补齐缺失的引号
if __name__ == "__main__":
    main()
