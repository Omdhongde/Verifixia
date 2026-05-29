"""
Balance the AI-Generated dataset folder to exactly 5,000 images
using local high-fidelity data augmentation (oversampling).
This runs instantly without any network dependencies.
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance

def main():
    root = Path(__file__).resolve().parent.parent
    aigen_dir = root / "DATA" / "AIGenerated"
    
    if not aigen_dir.exists():
        print(f"[Error] Directory not found: {aigen_dir}")
        return

    # Gather all existing local AI-Generated images
    existing_files = [f for f in aigen_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    existing_count = len(existing_files)
    target = 5000
    need_count = max(0, target - existing_count)

    print("=" * 60)
    print("Verifixia - High-Fidelity Dataset Augmentation Balancer")
    print("=" * 60)
    print(f"Target count  : {target} images")
    print(f"Current count : {existing_count} images")
    print(f"Need to create: {need_count} images (Oversampling & Augmenting)")
    print("=" * 60)

    if need_count == 0:
        print("[OK] Already balanced. Nothing to do.")
        return

    # Randomly select samples with replacement to fill the gap
    selected_files = random.choices(existing_files, k=need_count)
    
    saved_count = 0
    print("Augmenting and saving images locally...")
    
    for idx, img_path in enumerate(selected_files):
        try:
            with Image.open(img_path) as img:
                # 1. Random Horizontal Flip (highly effective, keeps realism)
                if random.random() > 0.5:
                    img = ImageOps.mirror(img)
                
                # 2. Slight rotation (-8 to +8 degrees)
                angle = random.uniform(-8.0, 8.0)
                img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
                
                # 3. Slight brightness/contrast tweak (+/- 8%)
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.92, 1.08))
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(random.uniform(0.92, 1.08))
                
                # Save as a unique augmented instance
                filename = f"AIGen_Aug_{idx + 1}.jpg"
                img.save(aigen_dir / filename, "JPEG", quality=95)
                saved_count += 1
                
                if saved_count % 300 == 0 or saved_count == need_count:
                    print(f"  Processed {saved_count}/{need_count} augmented instances...")
                    
        except Exception as e:
            print(f"  Error augmenting {img_path.name}: {e}")

    # Double check final count
    final_files = [f for f in aigen_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print("\n" + "=" * 60)
    print(f"[OK] Completed local dataset balancing!")
    print(f"New total local AI-Generated images: {len(final_files)} (perfectly balanced with Real & Deepfake)")
    print("=" * 60)

if __name__ == "__main__":
    main()
