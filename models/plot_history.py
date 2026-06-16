import json
import matplotlib.pyplot as plt
import os

history_file = 'multiclass_training_history.json'
output_file = 'training_graphs.png'

with open(history_file, 'r') as f:
    history = json.load(f)

epochs = range(1, len(history['train_acc']) + 1)

plt.figure(figsize=(14, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
plt.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy', linewidth=2)
plt.title('Training and Validation Accuracy (30 Epochs)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs, history['val_loss'], 'g-', label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss (30 Epochs)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Graph saved to {output_file}")
