import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_graphs():
    print("Loading training history files for images and videos...")
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    
    img_history_path = models_dir / "training_history.json"
    vid_history_path = models_dir / "deeperforensics_history.json"
    
    if not img_history_path.exists():
        print(f"Error: {img_history_path} does not exist.")
        return
    if not vid_history_path.exists():
        print(f"Error: {vid_history_path} does not exist.")
        return
        
    with open(img_history_path, "r") as f:
        img_history = json.load(f)
    with open(vid_history_path, "r") as f:
        vid_history = json.load(f)
        
    # Setup styling
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "Roboto", "Helvetica", "Arial"]
    
    # ------------------ 1. IMAGE TRAINING GRAPHS (Dynamic Epochs) ------------------
    print("Generating Image model training graphs...")
    img_epochs = list(range(1, len(img_history["train_loss"]) + 1))
    img_epoch_count = len(img_epochs)
    peak_img_acc = max(img_history["val_acc"]) * 100
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    axs[0].plot(img_epochs, img_history["train_loss"], label="Training Loss", color="#4A90E2", linewidth=2.5)
    axs[0].plot(img_epochs, img_history["val_loss"], label="Validation Loss", color="#D0021B", linewidth=2, linestyle="--")
    axs[0].set_title(f"Image Model: Loss Progression ({img_epoch_count} Epochs)", fontsize=13, fontweight="bold", pad=8)
    axs[0].set_xlabel("Epochs", fontsize=11)
    axs[0].set_ylabel("Loss", fontsize=11)
    axs[0].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    axs[0].grid(True, alpha=0.4)
    
    # Accuracy curves
    axs[1].plot(img_epochs, [a * 100 for a in img_history["train_acc"]], label="Training Accuracy", color="#4A90E2", linewidth=2.5)
    axs[1].plot(img_epochs, [a * 100 for a in img_history["val_acc"]], label="Validation Accuracy", color="#D0021B", linewidth=2, linestyle="--")
    axs[1].axhline(peak_img_acc, color="#7ED321", linestyle=":", label=f"Peak Val Acc ({peak_img_acc:.1f}%)")
    axs[1].set_title("Image Model: Accuracy Curves", fontsize=13, fontweight="bold", pad=8)
    axs[1].set_xlabel("Epochs", fontsize=11)
    axs[1].set_ylabel("Accuracy (%)", fontsize=11)
    axs[1].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    axs[1].grid(True, alpha=0.4)
    
    plt.suptitle(f"Verifixia Image Deepfake Detector - {img_epoch_count} Epochs Training Progression", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    img_out = models_dir / "image_training_history.png"
    plt.savefig(img_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved Image graphs to {img_out.resolve()}")
    
    # ------------------ 2. VIDEO TRAINING GRAPHS (Dynamic Epochs) ------------------
    print("Generating Video model training graphs...")
    vid_epochs = list(range(1, len(vid_history["train_loss"]) + 1))
    vid_epoch_count = len(vid_epochs)
    peak_vid_acc = max(vid_history["val_acc"]) * 100
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    axs[0].plot(vid_epochs, vid_history["train_loss"], label="Training Loss", color="#F5A623", linewidth=2.5)
    axs[0].plot(vid_epochs, vid_history["val_loss"], label="Validation Loss", color="#9013FE", linewidth=2, linestyle="--")
    axs[0].set_title(f"Video Model: Loss Progression ({vid_epoch_count} Epochs)", fontsize=13, fontweight="bold", pad=8)
    axs[0].set_xlabel("Epochs", fontsize=11)
    axs[0].set_ylabel("Loss", fontsize=11)
    axs[0].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    axs[0].grid(True, alpha=0.4)
    
    # Accuracy curves
    axs[1].plot(vid_epochs, [a * 100 for a in vid_history["train_acc"]], label="Training Accuracy", color="#F5A623", linewidth=2.5)
    axs[1].plot(vid_epochs, [a * 100 for a in vid_history["val_acc"]], label="Validation Accuracy", color="#9013FE", linewidth=2, linestyle="--")
    axs[1].axhline(peak_vid_acc, color="#7ED321", linestyle=":", label=f"Peak Val Acc ({peak_vid_acc:.1f}%)")
    axs[1].set_title("Video Model: Accuracy Curves", fontsize=13, fontweight="bold", pad=8)
    axs[1].set_xlabel("Epochs", fontsize=11)
    axs[1].set_ylabel("Accuracy (%)", fontsize=11)
    axs[1].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    axs[1].grid(True, alpha=0.4)
    
    plt.suptitle(f"Verifixia Video Deepfake Detector (DeeperForensics) - {vid_epoch_count} Epochs Training Progression", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    vid_out = models_dir / "video_training_history.png"
    plt.savefig(vid_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved Video graphs to {vid_out.resolve()}")
 
    # ------------------ 3. COMBINED IMAGE & VIDEO DASHBOARD ------------------
    print("Generating Combined Image & Video dashboard...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Image Loss
    axs[0, 0].plot(img_epochs, img_history["train_loss"], label="Training Loss", color="#4A90E2", linewidth=2.5)
    axs[0, 0].plot(img_epochs, img_history["val_loss"], label="Validation Loss", color="#D0021B", linewidth=2, linestyle="--")
    axs[0, 0].set_title("Image Model: BCE Loss Progression", fontsize=13, fontweight="bold", pad=8)
    axs[0, 0].set_xlabel("Epochs", fontsize=10)
    axs[0, 0].set_ylabel("Loss", fontsize=10)
    axs[0, 0].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    axs[0, 0].grid(True, alpha=0.4)
    
    # Image Accuracy
    axs[0, 1].plot(img_epochs, [a * 100 for a in img_history["train_acc"]], label="Training Accuracy", color="#4A90E2", linewidth=2.5)
    axs[0, 1].plot(img_epochs, [a * 100 for a in img_history["val_acc"]], label="Validation Accuracy", color="#D0021B", linewidth=2, linestyle="--")
    axs[0, 1].axhline(75.0, color="#7ED321", linestyle=":", label="Peak Accuracy (75.0%)")
    axs[0, 1].set_title("Image Model: Classification Accuracy", fontsize=13, fontweight="bold", pad=8)
    axs[0, 1].set_xlabel("Epochs", fontsize=10)
    axs[0, 1].set_ylabel("Accuracy (%)", fontsize=10)
    axs[0, 1].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    axs[0, 1].grid(True, alpha=0.4)
    
    # Video Loss
    axs[1, 0].plot(vid_epochs, vid_history["train_loss"], label="Training Loss", color="#F5A623", linewidth=2.5)
    axs[1, 0].plot(vid_epochs, vid_history["val_loss"], label="Validation Loss", color="#9013FE", linewidth=2, linestyle="--")
    axs[1, 0].set_title(f"Video Model (DeeperForensics): BCE Loss Progression ({vid_epoch_count} Epochs)", fontsize=13, fontweight="bold", pad=8)
    axs[1, 0].set_xlabel("Epochs", fontsize=10)
    axs[1, 0].set_ylabel("Loss", fontsize=10)
    axs[1, 0].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    axs[1, 0].grid(True, alpha=0.4)
    
    # Video Accuracy
    axs[1, 1].plot(vid_epochs, [a * 100 for a in vid_history["train_acc"]], label="Training Accuracy", color="#F5A623", linewidth=2.5)
    axs[1, 1].plot(vid_epochs, [a * 100 for a in vid_history["val_acc"]], label="Validation Accuracy", color="#9013FE", linewidth=2, linestyle="--")
    axs[1, 1].axhline(peak_vid_acc, color="#7ED321", linestyle=":", label=f"Peak Accuracy ({peak_vid_acc:.1f}%)")
    axs[1, 1].set_title(f"Video Model (DeeperForensics): Classification Accuracy", fontsize=13, fontweight="bold", pad=8)
    axs[1, 1].set_xlabel("Epochs", fontsize=10)
    axs[1, 1].set_ylabel("Accuracy (%)", fontsize=10)
    axs[1, 1].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    axs[1, 1].grid(True, alpha=0.4)
    
    plt.suptitle("Verifixia Deepfake Detectors - Image vs Video Training Progression Curves", fontsize=18, fontweight="bold", y=0.96)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    combined_out = models_dir / "combined_training_history.png"
    plt.savefig(combined_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved Combined dashboard to {combined_out.resolve()}")
    
    print("\n[SUCCESS] All graphs successfully generated for exact trained epochs!")

if __name__ == "__main__":
    generate_graphs()
