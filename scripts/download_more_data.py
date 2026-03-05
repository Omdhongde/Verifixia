"""
Download more images from Hemg/deepfake-and-real-images (HuggingFace).
Saves into DATA/Real/ and DATA/Fake/, skipping files that already exist.

Usage (from repo root):
    python scripts/download_more_data.py --real 500 --fake 500
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=int, default=500,
                        help="Target number of Real images")
    parser.add_argument("--fake", type=int, default=500,
                        help="Target number of Fake images")
    parser.add_argument("--data_dir", default="DATA",
                        help="Destination directory (default: DATA)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    real_dir = root / args.data_dir / "Real"
    fake_dir = root / args.data_dir / "Fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Count what already exists
    existing_real = len([f for f in real_dir.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    existing_fake = len([f for f in fake_dir.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    need_real = max(0, args.real - existing_real)
    need_fake = max(0, args.fake - existing_fake)

    print(f"\n{'='*55}")
    print(f"  Verifixia – Dataset Downloader")
    print(f"{'='*55}")
    print(f"  Existing  : {existing_real} Real  |  {existing_fake} Fake")
    print(f"  Target    : {args.real} Real  |  {args.fake} Fake")
    print(f"  To fetch  : {need_real} Real  |  {need_fake} Fake")
    print(f"{'='*55}\n")

    if need_real == 0 and need_fake == 0:
        print("✅ Already have enough images. Nothing to download.")
        return

    print("Connecting to Hemg/deepfake-and-real-images (streaming) …")
    ds = load_dataset(
        "Hemg/deepfake-and-real-images",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    counts   = {"Real": existing_real, "Fake": existing_fake}
    saved    = {"Real": 0, "Fake": 0}
    scanned  = 0

    for sample in ds:
        # label: 0 = Fake, 1 = Real  (from the dataset card)
        raw_label = sample.get("label", -1)
        label_str = {0: "Fake", 1: "Real"}.get(raw_label)
        if label_str is None:
            continue

        # Skip if we already have enough of this class
        if label_str == "Real" and saved["Real"] >= need_real:
            if saved["Fake"] >= need_fake:
                break
            continue
        if label_str == "Fake" and saved["Fake"] >= need_fake:
            if saved["Real"] >= need_real:
                break
            continue

        img = sample.get("image")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                continue

        if img.mode != "RGB":
            img = img.convert("RGB")

        dest_dir = real_dir if label_str == "Real" else fake_dir
        filename = f"{label_str}_{counts[label_str]}.jpg"
        save_path = dest_dir / filename

        try:
            img.save(save_path, quality=95)
            counts[label_str] += 1
            saved[label_str]  += 1
        except Exception as e:
            print(f"  ⚠  Could not save {filename}: {e}")
            continue

        scanned += 1
        total_saved = saved["Real"] + saved["Fake"]
        total_need  = need_real + need_fake
        bar_len = 40
        filled  = int(bar_len * total_saved / max(total_need, 1))
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {total_saved}/{total_need}  "
            f"Real={saved['Real']}/{need_real}  "
            f"Fake={saved['Fake']}/{need_fake}  "
            f"scanned={scanned + existing_real + existing_fake}",
            end="",
            flush=True,
        )

        if saved["Real"] >= need_real and saved["Fake"] >= need_fake:
            break

    print()
    print(f"\n{'='*55}")
    print(f"  Download complete.")
    print(f"  Real : {counts['Real']} total  (+{saved['Real']} new)")
    print(f"  Fake : {counts['Fake']} total  (+{saved['Fake']} new)")
    print(f"{'='*55}")
    print("\nNext step – retrain the model:")
    print("  python scripts/train_sklearn.py --n_aug 6\n")


if __name__ == "__main__":
    main()
