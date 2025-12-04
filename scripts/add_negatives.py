
import argparse
import shutil
from pathlib import Path
import random

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CATEGORIES = {"yagya", "sun", "light", "reflection"}

def ensure_structure(root: Path):
    for split in ["train", "val", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

def collect_images(source: Path):
    images = []
    if source.is_file() and source.suffix.lower() in SUPPORTED_EXTS:
        images.append(source)
    elif source.is_dir():
        for p in source.rglob("*"):
            if p.suffix.lower() in SUPPORTED_EXTS:
                images.append(p)
    return images

def split_list(items, train_ratio, val_ratio, test_ratio):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train+n_val]
    test = items[n_train+n_val:]
    return train, val, test

def copy_with_empty_label(files, split_dir: Path, category: str):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    copied = 0

    for src in files:
        dst_img = img_dir / src.name
        try:
            shutil.copy2(src, dst_img)
            dst_lbl = lbl_dir / f"{dst_img.stem}.txt"
            dst_lbl.touch(exist_ok=True)
            copied += 1
        except Exception as e:
            print(f"Failed to copy {src} -> {dst_img}: {e}")
    return copied

def main():
    parser = argparse.ArgumentParser(description="Add negative images (empty labels) to Dataset/negatives")
    parser.add_argument("--source", type=str, required=True, help="Folder or file with images to add")
    parser.add_argument("--category", type=str, required=True, choices=sorted(CATEGORIES), help="Negative category")
    parser.add_argument("--dest", type=str, default="Dataset/negatives", help="Destination dataset root")
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"),
                        help="Split ratios (sum must be 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = parser.parse_args()

    random.seed(args.seed)

    dest_root = Path(args.dest)
    ensure_structure(dest_root)

    src_path = Path(args.source)
    images = collect_images(src_path)
    if not images:
        print(f"No images found in: {src_path}")
        return 1

    train_ratio, val_ratio, test_ratio = args.split
    train_files, val_files, test_files = split_list(images, train_ratio, val_ratio, test_ratio)

    n_train = copy_with_empty_label(train_files, dest_root / "train", args.category)
    n_val = copy_with_empty_label(val_files, dest_root / "val", args.category)
    n_test = copy_with_empty_label(test_files, dest_root / "test", args.category)

    print(f"Added negatives for category '{args.category}':")
    print(f"  train: {n_train}")
    print(f"  val:   {n_val}")
    print(f"  test:  {n_test}")
    print("\nDataset YAML for negatives: Dataset/negatives.yaml")
    print("You can include negatives in training by mixing datasets or using --data Dataset/negatives.yaml for diagnostics.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())


