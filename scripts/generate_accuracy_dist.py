import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracy_distribution():
    # Data from the walkthrough
    classes = ['True Real', 'True Deepfake', 'True AI-Generated']
    
    # Predictions
    pred_real = [76.5, 3.5, 7.5]
    pred_deepfake = [14.0, 95.5, 1.0]
    pred_ai = [9.5, 1.0, 91.5]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, pred_real, width, label='Predicted Real', color='#4A90E2')
    rects2 = ax.bar(x, pred_deepfake, width, label='Predicted Deepfake', color='#D0021B')
    rects3 = ax.bar(x + width, pred_ai, width, label='Predicted AI', color='#F5A623')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Test Set Accuracy Distribution by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Auto-label bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    
    out_dir = r"d:\Final project folo\Verifixia\training_split_graphs\images"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "accuracy_distribution.png")
    
    plt.savefig(out_path, dpi=150)
    print(f"Saved accuracy distribution to {out_path}")

if __name__ == "__main__":
    plot_accuracy_distribution()
