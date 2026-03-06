import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection"""
    def __init__(self, root_dir, split='train', transform=None, use_huggingface=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_huggingface = use_huggingface
        self.samples = []

        if not use_huggingface:
            # Local dataset structure
            real_dir = os.path.join(root_dir, 'REAL_Images')
            fake_dir = os.path.join(root_dir, 'DeepFake_Images')
            
            all_samples = []
            
            # Load Real Images (Label 0)
            if os.path.exists(real_dir):
                for img_file in os.listdir(real_dir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        all_samples.append((os.path.join(real_dir, img_file), 0))
            else:
                print(f"Warning: Real images directory not found at {real_dir}")

            # Load Deepfake Images (Label 1)
            if os.path.exists(fake_dir):
                for img_file in os.listdir(fake_dir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        all_samples.append((os.path.join(fake_dir, img_file), 1))
            else:
                print(f"Warning: Deepfake images directory not found at {fake_dir}")

            # Shuffle and Split (80% Train, 20% Val)
            # Use fixed seed for reproducibility across train/val instantiation
            import random
            random.seed(42) 
            random.shuffle(all_samples)
            
            split_idx = int(0.8 * len(all_samples))
            
            if split == 'train':
                self.samples = all_samples[:split_idx]
            else:
                self.samples = all_samples[split_idx:]
            
            print(f"Loaded {len(self.samples)} images for {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_huggingface:
            # Placeholder for Hugging Face dataset
            img_path, label = self._get_huggingface_sample(idx)
        else:
            img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback to avoid crashing training
            image = Image.new('RGB', (299, 299), color='black')
            if self.transform:
                image = self.transform(image)
            return image, label

    def _get_huggingface_sample(self, idx):
        # Placeholder - in real implementation, load from actual dataset
        # For now, return dummy data
        return f"dummy_path_{idx}.jpg", np.random.randint(0, 2)

class DeepfakeDetector(nn.Module):
    """Xception-based deepfake detection model"""
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Entry flow
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        # Middle flow (simplified)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)

        # Exit flow
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Entry flow
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Middle flow
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        # Exit flow
        x = self.relu(self.bn5(self.conv5(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)

def train_model(config):
    """Train the deepfake detection model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    if config['use_huggingface_dataset']:
        print("Using Hugging Face dataset...")
        # Placeholder for Hugging Face dataset loading
        train_dataset = DeepfakeDataset(None, transform=transform, use_huggingface=True)
        val_dataset = DeepfakeDataset(None, transform=transform, use_huggingface=True)
    else:
        train_dataset = DeepfakeDataset(config['train_data_path'], split='train', transform=transform, use_huggingface=False)
        val_dataset = DeepfakeDataset(config['val_data_path'], split='val', transform=transform, use_huggingface=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Initialize model
    model = DeepfakeDetector()
    model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    best_accuracy = 0.0
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(train_labels, train_preds)
        train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_loss = val_loss / len(val_loader)

        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
        )
        print(
            "Val Loss: "
            f"{val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
            f"F1: {val_f1:.4f}"
        )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")

        scheduler.step()

    print(f"\nTraining completed. Best validation accuracy: {best_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Sanitize and validate the config file path to prevent path traversal
    config_path = os.path.realpath(args.config)
    allowed_base = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    if not config_path.startswith(allowed_base):
        raise ValueError(f"Config path '{config_path}' is outside the allowed directory.")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)

    # Train the model
    train_model(config)

if __name__ == '__main__':
    main()
