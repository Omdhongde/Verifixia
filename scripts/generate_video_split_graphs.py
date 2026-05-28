import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_split_graphs():
    print("=========================================================")
    print("  Verifixia - Video Model Split Epochs Graph Generator")
    print("=========================================================")
    
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    history_path = models_dir / "deeperforensics_history.json"
    
    if not history_path.exists():
        print(f"Error: {history_path} does not exist.")
        return
        
    with open(history_path, "r") as f:
        history = json.load(f)
        
    train_loss = history["train_loss"]
    train_acc = history["train_acc"]
    val_loss = history["val_loss"]
    val_acc = history["val_acc"]
    
    total_epochs = len(train_loss)
    print(f"Loaded training history with {total_epochs} total epochs.")
    
    # Define splits (1-30, 30-50, 50-80) - inclusive of boundaries to show continuity
    ranges = [
        {"name": "Epochs 1 to 30", "start": 1, "end": 30, "filename": "video_epochs_1_30.png", "color_tr": "#F5A623", "color_val": "#9013FE"},
        {"name": "Epochs 30 to 50", "start": 30, "end": 50, "filename": "video_epochs_30_50.png", "color_tr": "#4A90E2", "color_val": "#D0021B"},
        {"name": "Epochs 50 to 80", "start": 50, "end": 80, "filename": "video_epochs_50_80.png", "color_tr": "#7ED321", "color_val": "#4A4A4A"}
    ]
    
    # Configure styles for elite presentation
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "Roboto", "Helvetica", "Arial"]
    
    for r in ranges:
        start_idx = r["start"] - 1
        end_idx = min(r["end"], total_epochs)
        
        # Check if we have enough epochs
        if start_idx >= total_epochs:
            print(f"[Warning] Skipping {r['name']} - training has not reached epoch {r['start']} yet.")
            continue
            
        epochs_range = list(range(r["start"], end_idx + 1))
        tr_loss_slice = train_loss[start_idx:end_idx]
        val_loss_slice = val_loss[start_idx:end_idx]
        tr_acc_slice = [a * 100 for a in train_acc[start_idx:end_idx]]
        val_acc_slice = [a * 100 for a in val_acc[start_idx:end_idx]]
        
        peak_acc = max(val_acc_slice)
        
        print(f"Generating graphs for {r['name']} (slicing indices {start_idx} to {end_idx})...")
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
        
        # Left Subplot: Loss Curves
        axs[0].plot(epochs_range, tr_loss_slice, label="Training Loss", color=r["color_tr"], linewidth=2.5)
        axs[0].plot(epochs_range, val_loss_slice, label="Validation Loss", color=r["color_val"], linewidth=2, linestyle="--")
        axs[0].set_title(f"Video Model: Loss Progression ({r['name']})", fontsize=13, fontweight="bold", pad=10)
        axs[0].set_xlabel("Epochs", fontsize=11)
        axs[0].set_ylabel("Loss (BCE)", fontsize=11)
        axs[0].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
        axs[0].grid(True, alpha=0.4)
        
        # Right Subplot: Accuracy Curves
        axs[1].plot(epochs_range, tr_acc_slice, label="Training Accuracy", color=r["color_tr"], linewidth=2.5)
        axs[1].plot(epochs_range, val_acc_slice, label="Validation Accuracy", color=r["color_val"], linewidth=2, linestyle="--")
        axs[1].axhline(peak_acc, color="#2CA02C", linestyle=":", label=f"Peak Val Acc ({peak_acc:.1f}%)")
        axs[1].set_title(f"Video Model: Accuracy curves ({r['name']})", fontsize=13, fontweight="bold", pad=10)
        axs[1].set_xlabel("Epochs", fontsize=11)
        axs[1].set_ylabel("Accuracy (%)", fontsize=11)
        axs[1].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
        axs[1].grid(True, alpha=0.4)
        
        plt.suptitle(f"Verifixia Video Deepfake Detector - {r['name']} Training Analysis", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        
        output_path = models_dir / r["filename"]
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Saved split graph to {output_path.resolve()}")
        
    print("\n[SUCCESS] Split training graphs generated successfully!")
    print("=========================================================")

if __name__ == "__main__":
    generate_split_graphs()
