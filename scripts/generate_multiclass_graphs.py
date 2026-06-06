import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_multiclass_graphs():
    print("Loading training history file for the multi-class model (80 epochs)...")
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    
    multi_history_path = models_dir / "multiclass_training_history.json"
    
    if not multi_history_path.exists():
        print(f"Error: {multi_history_path} does not exist.")
        return
        
    with open(multi_history_path, "r") as f:
        multi_history = json.load(f)
        
    # Setup styling
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "Roboto", "Helvetica", "Arial"]
    
    print("Generating Multi-Class model training graphs...")
    multi_epochs = list(range(1, len(multi_history["train_loss"]) + 1))
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cross-Entropy Loss curves
    axs[0].plot(multi_epochs, multi_history["train_loss"], label="Training Loss", color="#BD10E0", linewidth=2.5)
    axs[0].plot(multi_epochs, multi_history["val_loss"], label="Validation Loss", color="#50E3C2", linewidth=2, linestyle="--")
    axs[0].set_title("Multi-Class Model: Cross-Entropy Loss Progression", fontsize=13, fontweight="bold", pad=8)
    axs[0].set_xlabel("Epochs", fontsize=11)
    axs[0].set_ylabel("Loss", fontsize=11)
    axs[0].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    axs[0].grid(True, alpha=0.4)
    
    # Accuracy curves
    axs[1].plot(multi_epochs, [a * 100 for a in multi_history["train_acc"]], label="Training Accuracy", color="#BD10E0", linewidth=2.5)
    axs[1].plot(multi_epochs, [a * 100 for a in multi_history["val_acc"]], label="Validation Accuracy", color="#50E3C2", linewidth=2, linestyle="--")
    peak_val_acc = max(multi_history["val_acc"]) * 100
    axs[1].axhline(peak_val_acc, color="#7ED321", linestyle=":", label=f"Peak Val Acc ({peak_val_acc:.2f}%)")
    axs[1].set_title("Multi-Class Model: Accuracy Curves", fontsize=13, fontweight="bold", pad=8)
    axs[1].set_xlabel("Epochs", fontsize=11)
    axs[1].set_ylabel("Accuracy (%)", fontsize=11)
    axs[1].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    axs[1].grid(True, alpha=0.4)
    
    plt.suptitle(f"Verifixia Multi-Class Deepfake Detector (3 Classes) - {len(multi_epochs)} Epochs Training Progression", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    multi_out = models_dir / "multiclass_training_history.png"
    plt.savefig(multi_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved Multi-Class graphs to {multi_out.resolve()}")
    
    print("\n[SUCCESS] Dedicated Multi-Class graphs successfully generated for the 80 epochs training!")

if __name__ == "__main__":
    generate_multiclass_graphs()
