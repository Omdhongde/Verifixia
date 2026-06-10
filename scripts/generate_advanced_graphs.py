import os
import json
import matplotlib
matplotlib.use('Agg') # Non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_graphs():
    print("Loading advanced training history...")
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    
    history_path = models_dir / "advanced_cnn_lstm_history.json"
    info_path = models_dir / "advanced_cnn_lstm_info.json"
    
    if not history_path.exists():
        print(f"[FAIL] History file not found: {history_path.resolve()}")
        return
        
    with open(history_path, "r") as f:
        history = json.load(f)
        
    # Setup styling
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "Roboto", "Helvetica", "Arial"]
    
    epochs = list(range(1, len(history["train_loss"]) + 1))
    epoch_count = len(epochs)
    
    if epoch_count > 0:
        peak_val_acc = max(history["val_acc"]) * 100
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Loss curves
        axs[0].plot(epochs, history["train_loss"], label="Training Loss (BCE)", color="#1F77B4", linewidth=2.5)
        axs[0].plot(epochs, history["val_loss"], label="Validation Loss (BCE)", color="#FF7F0E", linewidth=2, linestyle="--")
        axs[0].set_title(f"Advanced Model: Loss Progression ({epoch_count} Epochs)", fontsize=13, fontweight="bold", pad=8)
        axs[0].set_xlabel("Epochs", fontsize=11)
        axs[0].set_ylabel("Loss", fontsize=11)
        axs[0].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
        axs[0].grid(True, alpha=0.4)
        
        # 2. Accuracy curves
        axs[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Training Accuracy", color="#1F77B4", linewidth=2.5)
        axs[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation Accuracy", color="#FF7F0E", linewidth=2, linestyle="--")
        axs[1].axhline(peak_val_acc, color="#2CA02C", linestyle=":", label=f"Peak Val Acc ({peak_val_acc:.1f}%)")
        
        # Load test info if exists
        if info_path.exists():
            with open(info_path, "r") as f:
                info = json.load(f)
            test_acc = info.get("test_accuracy", 0.0) * 100
            axs[1].axhline(test_acc, color="#D62728", linestyle="-.", label=f"Test Accuracy ({test_acc:.1f}%)")
            
        axs[1].set_title("Advanced Model: Accuracy Curves", fontsize=13, fontweight="bold", pad=8)
        axs[1].set_xlabel("Epochs", fontsize=11)
        axs[1].set_ylabel("Accuracy (%)", fontsize=11)
        axs[1].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
        axs[1].grid(True, alpha=0.4)
        
        plt.suptitle(f"Verifixia Advanced 3-Tier Multi-Modal Detector - {epoch_count} Epochs Training Curves", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        
        out_path = models_dir / "advanced_training_history.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SUCCESS] Saved Advanced model graphs to {out_path.resolve()}")

if __name__ == "__main__":
    generate_graphs()
