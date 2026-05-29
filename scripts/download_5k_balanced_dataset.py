import os
import sys
import argparse
from pathlib import Path
from PIL import Image

def get_existing_count(directory):
    path = Path(directory)
    if not path.exists():
        return 0
    return len([f for f in path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

def download_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000, help="Target count per class")
    parser.add_argument("--data_dir", default="DATA", help="Destination data directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_path = root / args.data_dir
    
    real_dir = data_path / "Real"
    deepfake_dir = data_path / "Deepfake"
    aigen_dir = data_path / "AIGenerated"

    real_dir.mkdir(parents=True, exist_ok=True)
    deepfake_dir.mkdir(parents=True, exist_ok=True)
    aigen_dir.mkdir(parents=True, exist_ok=True)

    existing_real = get_existing_count(real_dir)
    existing_deepfake = get_existing_count(deepfake_dir)
    existing_aigen = get_existing_count(aigen_dir)

    print("=" * 60)
    print("Verifixia - 5k Balanced Dataset Downloader")
    print("=" * 60)
    print(f"Target count: {args.target} images per class")
    print(f"Current counts: Real: {existing_real} | Deepfake: {existing_deepfake} | AIGenerated: {existing_aigen}")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("[FAIL] 'datasets' library is not installed in this environment.")
        return

    # 1. Download AI-Generated images
    need_aigen = max(0, args.target - existing_aigen)
    if need_aigen > 0:
        print(f"\nDownloading {need_aigen} AI-Generated images...")
        try:
            print("Connecting to Parveshiiii/AI-vs-Real...")
            # Streaming mode is fast and doesn't download the entire dataset archive upfront
            ds_ai = load_dataset("Parveshiiii/AI-vs-Real", split="train", streaming=True)
            saved_aigen = 0
            for sample in ds_ai:
                # Label 0 in Parveshiiii/AI-vs-Real is AI-Generated
                if sample.get("binary_label") == 0:
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
                    
                    filename = f"AIGen_{existing_aigen + saved_aigen}.jpg"
                    img.save(aigen_dir / filename, quality=95)
                    saved_aigen += 1
                    
                    if saved_aigen % 250 == 0 or saved_aigen == need_aigen:
                        print(f"  [AI-Generated] Saved {saved_aigen}/{need_aigen} images...")
                    
                    if saved_aigen >= need_aigen:
                        break
            
            existing_aigen += saved_aigen
            print(f"Finished AI-Generated phase. Total: {existing_aigen}")
        except Exception as e:
            print(f"  Error downloading AI-Generated images: {e}")

    # 2. Download Real images
    need_real = max(0, args.target - existing_real)
    if need_real > 0:
        print(f"\nDownloading {need_real} Real human face images...")
        try:
            print("Connecting to Parveshiiii/AI-vs-Real for Real faces...")
            ds_ai = load_dataset("Parveshiiii/AI-vs-Real", split="train", streaming=True)
            saved_real = 0
            for sample in ds_ai:
                # Label 1 in Parveshiiii/AI-vs-Real is Real
                if sample.get("binary_label") == 1:
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
                    
                    filename = f"Real_{existing_real + saved_real}.jpg"
                    img.save(real_dir / filename, quality=95)
                    saved_real += 1
                    
                    if saved_real % 250 == 0 or saved_real == need_real:
                        print(f"  [Real - Phase 1] Saved {saved_real}/{need_real} images...")
                    
                    if saved_real >= need_real:
                        break
            existing_real += saved_real
            need_real = max(0, args.target - existing_real)
        except Exception as e:
            print(f"  Warning: error downloading Real images from Parveshiiii/AI-vs-Real: {e}")

        if need_real > 0:
            print(f"Connecting to Hemg/deepfake-and-real-images for remaining {need_real} Real faces...")
            try:
                ds_hr = load_dataset("Hemg/deepfake-and-real-images", split="train", streaming=True)
                saved_real = 0
                for sample in ds_hr:
                    # Label 1 is Real
                    if sample.get("label") == 1:
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
                        
                        filename = f"Real_{existing_real + saved_real}.jpg"
                        img.save(real_dir / filename, quality=95)
                        saved_real += 1
                        
                        if saved_real % 250 == 0 or saved_real == need_real:
                            print(f"  [Real - Phase 2] Saved {saved_real}/{need_real} images...")
                        
                        if saved_real >= need_real:
                            break
                existing_real += saved_real
            except Exception as e:
                print(f"  Error downloading Real images from Hemg: {e}")
        print(f"Finished Real phase. Total: {existing_real}")

    # 3. Download Deepfake images
    need_deepfake = max(0, args.target - existing_deepfake)
    if need_deepfake > 0:
        print(f"\nDownloading {need_deepfake} Deepfake face images...")
        try:
            print("Connecting to Hemg/deepfake-and-real-images...")
            ds_hd = load_dataset("Hemg/deepfake-and-real-images", split="train", streaming=True)
            saved_deepfake = 0
            for sample in ds_hd:
                # Label 0 is Fake
                if sample.get("label") == 0:
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
                    
                    filename = f"Deepfake_{existing_deepfake + saved_deepfake}.jpg"
                    img.save(deepfake_dir / filename, quality=95)
                    saved_deepfake += 1
                    
                    if saved_deepfake % 250 == 0 or saved_deepfake == need_deepfake:
                        print(f"  [Deepfake] Saved {saved_deepfake}/{need_deepfake} images...")
                    
                    if saved_deepfake >= need_deepfake:
                        break
            existing_deepfake += saved_deepfake
            print(f"Finished Deepfake phase. Total: {existing_deepfake}")
        except Exception as e:
            print(f"  Error downloading Deepfake images from Hemg: {e}")

    print("\n" + "=" * 60)
    print("Balanced dataset preparation finished!")
    print(f"Final local counts:")
    print(f"  - Real: {get_existing_count(real_dir)} images")
    print(f"  - Deepfake: {get_existing_count(deepfake_dir)} images")
    print(f"  - AI-Generated: {get_existing_count(aigen_dir)} images")
    print("=" * 60)

if __name__ == "__main__":
    download_dataset()
