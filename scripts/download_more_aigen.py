"""
Download additional high-quality AI-Generated images from bitmind/SyntheticFacesHQ
to perfectly balance the AI-Generated category to 5,000 images.

Saves directly into DATA/AIGenerated/ without emojis to avoid console encoding crashes on Windows.
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000, help="Target total count for AIGenerated")
    parser.add_argument("--data_dir", default="DATA", help="Destination data directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    aigen_dir = root / args.data_dir / "AIGenerated"
    aigen_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing_files = [f for f in aigen_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    existing_count = len(existing_files)
    
    need_count = max(0, args.target - existing_count)

    print("=" * 60)
    print("Verifixia - AI-Generated Dataset Balancer")
    print("=" * 60)
    print(f"Target total: {args.target} images")
    print(f"Current count: {existing_count} images")
    print(f"Need to download: {need_count} images")
    print("=" * 60)

    if need_count == 0:
        print("[OK] Already have enough images. Nothing to download.")
        return

    # Start naming from the highest index plus 1 to avoid overwriting
    max_idx = 0
    for f in existing_files:
        name = f.stem
        parts = name.split("_")
        if len(parts) > 1:
            try:
                idx = int(parts[-1])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    
    start_idx = max_idx + 1
    print(f"Starting saving with filename prefix: AIGen_{start_idx}...")

    print("\nConnecting to bitmind/SyntheticFacesHQ (streaming) ...")
    saved_count = 0
    
    try:
        # Load bitmind/SyntheticFacesHQ in streaming mode
        # This is a public dataset with standard parquet format and high-quality 1024x1024 synthetic faces
        ds = load_dataset(
            "bitmind/SyntheticFacesHQ",
            split="train",
            streaming=True
        )
        
        for sample in ds:
            if saved_count >= need_count:
                break
                
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
                
            # Resize to standard size (299x299) to keep disk space small and neat!
            img = img.resize((299, 299), Image.Resampling.LANCZOS)
            
            filename = f"AIGen_{start_idx + saved_count}.jpg"
            save_path = aigen_dir / filename
            
            try:
                img.save(save_path, "JPEG", quality=95)
                saved_count += 1
                if saved_count % 100 == 0 or saved_count == need_count:
                    print(f"  [Synthetic Faces] Saved {saved_count}/{need_count} images...")
            except Exception as e:
                print(f"  Error saving image: {e}")
                
        print(f"\n[OK] Balance operation completed. Total new images downloaded: {saved_count}")
        print(f"Final local AIGenerated dataset size: {existing_count + saved_count}")
        
    except Exception as e:
        print(f"Failed during streaming download: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
