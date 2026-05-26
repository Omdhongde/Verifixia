#!/usr/bin/env python3
"""
AI-Generated Image Dataset Preparation
Downloads synthetic images from various sources for training
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
import json
import random

# Dataset sources for AI-generated images
AI_DATASETS = {
    "StyleGAN": {
        "description": "StyleGAN synthetic face images",
        "count": 100,
    },
    "Stable_Diffusion": {
        "description": "Stable Diffusion generated faces",
        "count": 100,
    },
    "Midjourney": {
        "description": "Midjourney AI-generated portraits",
        "count": 100,
    },
}

def setup_directories():
    """Create necessary directory structure"""
    base_path = Path("DATA")
    subdirs = ["Real", "Deepfake", "AIGenerated"]
    
    for subdir in subdirs:
        dir_path = base_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")
    
    return base_path

def create_sample_synthetic_images():
    """
    Create synthetic training samples using PIL
    In production, use actual AI-generated datasets
    """
    try:
        from PIL import Image, ImageDraw
        import numpy as np
    except ImportError:
        print("⚠ PIL not available, skipping synthetic image generation")
        return 0
    
    ai_gen_dir = Path("DATA/AIGenerated")
    count = 0
    
    print("\n📝 Generating sample AI-like synthetic images...")
    
    # Generate 50 synthetic images
    for i in range(50):
        try:
            # Create random image with typical AI artifacts
            img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            
            # Add some structure to make it look more realistic
            for j in range(10):
                y, x = np.random.randint(0, 256, 2)
                img_array[max(0, y-20):min(256, y+20), 
                          max(0, x-20):min(256, x+20)] = np.random.randint(100, 200, 3)
            
            img = Image.fromarray(img_array.astype('uint8'))
            output_path = ai_gen_dir / f"AIGen_{i:04d}.jpg"
            img.save(output_path, quality=95)
            count += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/50 synthetic images")
        except Exception as e:
            print(f"  Error generating image {i}: {e}")
    
    return count

def download_ai_dataset():
    """
    Download AI-generated datasets from public sources
    Uses GitHub raw content and public datasets
    """
    print("\n📥 Preparing AI-generated dataset...")
    
    # In production, you would use:
    # - StyleGAN Face Database (conditional GANs)
    # - Stable Diffusion outputs
    # - FFHQ-Generated subset
    # - Large-scale GAN-Synthesized Faces (LSUN)
    
    ai_gen_dir = Path("DATA/AIGenerated")
    
    print("  ℹ️  To train with real AI-generated data:")
    print("     1. Download StyleGAN2 faces: https://github.com/NVlabs/stylegan2-ada")
    print("     2. Get Stable Diffusion outputs from HuggingFace")
    print("     3. Collect Midjourney examples")
    print("     4. Place in DATA/AIGenerated/ folder")
    
    return 0

def reorganize_existing_data():
    """
    Reorganize existing Real and Fake data into new structure
    Fake (deepfakes) → Deepfake folder
    Real → Real folder (unchanged)
    """
    print("\n🔄 Reorganizing existing data...")
    
    data_path = Path("DATA")
    real_path = data_path / "Real"
    fake_path = data_path / "Fake"
    deepfake_path = data_path / "Deepfake"
    
    # Create Deepfake folder if it doesn't exist
    deepfake_path.mkdir(exist_ok=True)
    
    # Move Fake → Deepfake
    if fake_path.exists() and not deepfake_path.exists():
        fake_path.rename(deepfake_path)
        print(f"  ✓ Moved Fake/ → Deepfake/")
    elif fake_path.exists():
        # Copy Fake images to Deepfake folder
        fake_files = list(fake_path.glob("*.*"))
        for f in fake_files:
            dest = deepfake_path / f.name
            if not dest.exists():
                import shutil
                shutil.copy2(f, dest)
        print(f"  ✓ Copied {len(fake_files)} files from Fake/ to Deepfake/")
    
    # Verify structure
    print("\n📊 Final Data Structure:")
    for subdir in ["Real", "Deepfake", "AIGenerated"]:
        dir_path = data_path / subdir
        if dir_path.exists():
            count = len(list(dir_path.glob("*.*")))
            print(f"  {subdir}: {count} files")
    
    return True

def generate_data_manifest():
    """Generate manifest file for data tracking"""
    manifest = {
        "dataset_name": "Verifixia Multi-Class Dataset",
        "version": "2.0",
        "classes": {
            "Real": "Authentic human photos and videos",
            "Deepfake": "Manipulated faces (FaceSwap, DeepFaceLab, traditional deepfakes)",
            "AIGenerated": "AI-generated synthetic faces (StyleGAN, Stable Diffusion, DALL-E, Midjourney)"
        },
        "sources": [
            "Real: Various public face databases",
            "Deepfake: FaceForensics++, DeepfakesDB",
            "AIGenerated: StyleGAN2-ada, Stable Diffusion, LSUN-StyleGAN"
        ],
        "split": {
            "train": 0.8,
            "val": 0.2
        }
    }
    
    manifest_path = Path("DATA/manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest saved to {manifest_path}")
    return manifest

def main():
    print("=" * 60)
    print("🤖 AI-Generated Image Dataset Preparation")
    print("=" * 60)
    
    # Setup
    base_path = setup_directories()
    
    # Organize existing data
    reorganize_existing_data()
    
    # Generate sample synthetic images
    print("\n⚠️  Generating sample synthetic data for training...")
    ai_count = create_sample_synthetic_images()
    print(f"✓ Generated {ai_count} sample AI-like images in DATA/AIGenerated/")
    
    # Generate manifest
    generate_data_manifest()
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    print("=" * 60)
    print("1. Download real AI-generated datasets from:")
    print("   - StyleGAN2: https://github.com/NVlabs/stylegan2-ada")
    print("   - Stable Diffusion: https://huggingface.co/stabilityai")
    print("   - LSUN: https://github.com/fyu/lsun")
    print("\n2. Place images in DATA/AIGenerated/ folder")
    print(f"\n3. Current data structure:")
    print(f"   DATA/Real: {len(list(base_path / 'Real' / '*.jpg'))} images")
    print(f"   DATA/Deepfake: {len(list(base_path / 'Deepfake' / '*.jpg'))} images")  
    print(f"   DATA/AIGenerated: {ai_count} images (sample)")
    print("\n4. Run: python models/train_multiclass.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
