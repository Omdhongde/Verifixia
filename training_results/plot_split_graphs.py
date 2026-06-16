import json
import matplotlib.pyplot as plt
import os

history_file = r'd:\Final project folo\Verifixia\models\multiclass_training_history.json'
out_1_30 = 'training_graphs_1_30.png'
out_31_50 = 'training_graphs_31_50.png'

with open(history_file, 'r') as f:
    history = json.load(f)

# Function to plot
def plot_range(start_idx, end_idx, title_suffix, output_file):
    epochs = range(start_idx + 1, end_idx + 1)
    
    t_acc = history['train_acc'][start_idx:end_idx]
    v_acc = history['val_acc'][start_idx:end_idx]
    t_loss = history['train_loss'][start_idx:end_idx]
    v_loss = history['val_loss'][start_idx:end_idx]
    
    plt.figure(figsize=(14, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, t_acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, v_acc, 'g-', label='Validation Accuracy', linewidth=2)
    plt.title(f'Training and Validation Accuracy {title_suffix}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, t_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, v_loss, 'g-', label='Validation Loss', linewidth=2)
    plt.title(f'Training and Validation Loss {title_suffix}', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {output_file}")

# Plot 1 to 30 (indices 0 to 30)
plot_range(0, 30, '(Epochs 1-30)', out_1_30)

# Plot 31 to 50 (indices 30 to 50)
plot_range(30, 50, '(Epochs 31-50)', out_31_50)
