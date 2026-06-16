import os
import json
import matplotlib.pyplot as plt

def plot_and_save(train_loss, train_acc, val_loss, val_acc, start_epoch, end_epoch, output_path, title):
    actual_len = len(train_loss)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Determine what data we can actually plot
    plot_start = min(start_epoch, actual_len + 1)
    plot_end = min(end_epoch, actual_len)
    
    if plot_start <= plot_end:
        epochs = list(range(plot_start, plot_end + 1))
        t_loss = train_loss[plot_start-1:plot_end]
        v_loss = val_loss[plot_start-1:plot_end]
        t_acc = train_acc[plot_start-1:plot_end]
        v_acc = val_acc[plot_start-1:plot_end]
    else:
        epochs, t_loss, v_loss, t_acc, v_acc = [], [], [], [], []

    # Loss
    if epochs:
        axs[0].plot(epochs, t_loss, label="Training Loss", color="#4A90E2", linewidth=2)
        axs[0].plot(epochs, v_loss, label="Validation Loss", color="#D0021B", linestyle="--", linewidth=2)
    axs[0].set_title(f"{title} - Loss Progression")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlim([start_epoch, end_epoch])
    if epochs:
        axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.7)

    # Accuracy
    if epochs:
        axs[1].plot(epochs, [a * 100 for a in t_acc], label="Training Accuracy", color="#4A90E2", linewidth=2)
        axs[1].plot(epochs, [a * 100 for a in v_acc], label="Validation Accuracy", color="#D0021B", linestyle="--", linewidth=2)
    axs[1].set_title(f"{title} - Accuracy Curves")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_xlim([start_epoch, end_epoch])
    # Removing hardcoded y-axis limits allows matplotlib to auto-scale, making small improvements visible!
    if epochs:
        axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.7)

    if not epochs:
        fig.text(0.5, 0.5, f"No data for epochs {start_epoch}-{end_epoch}\n(Model converged early at {actual_len} epochs)", 
                 ha='center', va='center', fontsize=16, color='gray', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    base_dir = r"d:\Final project folo\Verifixia"
    out_dir = os.path.join(base_dir, "training_split_graphs")
    images_out = os.path.join(out_dir, "images")
    videos_out = os.path.join(out_dir, "videos")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(videos_out, exist_ok=True)

    print("Generating Image Graphs...")
    images_hist_path = os.path.join(base_dir, "models", "multiclass_training_history.json")
    with open(images_hist_path, "r") as f:
        img_hist = json.load(f)
    
    plot_and_save(img_hist['train_loss'], img_hist['train_acc'], img_hist['val_loss'], img_hist['val_acc'], 1, 30, os.path.join(images_out, "epochs_1_30.png"), "Image Model (Epochs 1-30)")
    plot_and_save(img_hist['train_loss'], img_hist['train_acc'], img_hist['val_loss'], img_hist['val_acc'], 31, 50, os.path.join(images_out, "epochs_31_50.png"), "Image Model (Epochs 31-50)")
    plot_and_save(img_hist['train_loss'], img_hist['train_acc'], img_hist['val_loss'], img_hist['val_acc'], 1, 50, os.path.join(images_out, "epochs_1_50.png"), "Image Model (Epochs 1-50)")

    print("Generating Video Graphs...")
    videos_hist_path = os.path.join(base_dir, "models", "deeperforensics_history.json")
    with open(videos_hist_path, "r") as f:
        vid_hist = json.load(f)

    plot_and_save(vid_hist['train_loss'], vid_hist['train_acc'], vid_hist['val_loss'], vid_hist['val_acc'], 1, 30, os.path.join(videos_out, "epochs_1_30.png"), "Video Model (Epochs 1-30)")
    
    print(f"All graphs have been saved to: {out_dir}")

if __name__ == "__main__":
    main()
