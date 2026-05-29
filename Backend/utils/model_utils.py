# Model utilities and helper functions

import os
import time
from typing import Dict, Tuple, Optional, Any

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Stub so the rest of the module doesn't fail at import time
    nn = None
    torch = None


if _TORCH_AVAILABLE:
    class SqueezeExcitationBlock(nn.Module):
        """Channel Attention mechanism"""
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return x * self.se(x)

    class ResidualBlock(nn.Module):
        """Residual block with SE attention"""
        def __init__(self, in_channels, out_channels, stride=1, reduction=16):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.se = SqueezeExcitationBlock(out_channels, reduction)
            self.relu = nn.ReLU(inplace=True)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            residual = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out += residual
            out = self.relu(out)
            return out

    class DeepfakeDetector(nn.Module):
        """Improved Deepfake Detection Model with Residual Blocks & SE Attention"""
        def __init__(self, pretrained=False):
            super().__init__()
            
            # Initial convolution layer
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # Residual layers with increasing channels
            self.layer1 = self._make_layer(64, 64, 2, stride=1)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            
            # Multi-scale feature aggregation
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            
            # Classification head with bottleneck
            self.fc1 = nn.Linear(512 * 2, 1024)
            self.bn_fc1 = nn.BatchNorm1d(1024)
            self.dropout1 = nn.Dropout(0.6)
            
            self.fc2 = nn.Linear(1024, 512)
            self.bn_fc2 = nn.BatchNorm1d(512)
            self.dropout2 = nn.Dropout(0.5)
            
            self.fc3 = nn.Linear(512, 256)
            self.bn_fc3 = nn.BatchNorm1d(256)
            self.dropout3 = nn.Dropout(0.4)
            
            self.fc_out = nn.Linear(256, 1)
            self.sigmoid = nn.Sigmoid()
            
            # Initialize weights
            self._init_weights()
        
        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            # Initial layers
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            
            # Residual blocks
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            # Multi-scale pooling (Avg + Max)
            avg_feat = self.avg_pool(x)
            max_feat = self.max_pool(x)
            x = torch.cat([avg_feat, max_feat], dim=1)
            x = x.view(x.size(0), -1)
            
            # Classification head
            x = self.relu(self.bn_fc1(self.fc1(x)))
            x = self.dropout1(x)
            
            x = self.relu(self.bn_fc2(self.fc2(x)))
            x = self.dropout2(x)
            
            x = self.relu(self.bn_fc3(self.fc3(x)))
            x = self.dropout3(x)
            
            x = self.fc_out(x)
            return self.sigmoid(x)

    class MultiClassDetector(nn.Module):
        """Multi-Class Detector: Real vs Deepfake vs AI-Generated (3-class classification)"""
        def __init__(self, num_classes=3):
            super().__init__()
            
            # Initial convolution layer
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # Residual layers with increasing channels
            self.layer1 = self._make_layer(64, 64, 2, stride=1)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            
            # Multi-scale feature aggregation
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            
            # Classification head with bottleneck
            self.fc1 = nn.Linear(512 * 2, 1024)
            self.bn_fc1 = nn.BatchNorm1d(1024)
            self.dropout1 = nn.Dropout(0.5)
            
            self.fc2 = nn.Linear(1024, 512)
            self.bn_fc2 = nn.BatchNorm1d(512)
            self.dropout2 = nn.Dropout(0.4)
            
            self.fc3 = nn.Linear(512, 256)
            self.bn_fc3 = nn.BatchNorm1d(256)
            self.dropout3 = nn.Dropout(0.3)
            
            # Multi-class output
            self.fc_out = nn.Linear(256, num_classes)
            
            # Initialize weights
            self._init_weights()
        
        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            # Initial layers
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            
            # Residual blocks
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            # Multi-scale pooling (Avg + Max)
            avg_feat = self.avg_pool(x)
            max_feat = self.max_pool(x)
            x = torch.cat([avg_feat, max_feat], dim=1)
            x = x.view(x.size(0), -1)
            
            # Classification head
            x = self.relu(self.bn_fc1(self.fc1(x)))
            x = self.dropout1(x)
            
            x = self.relu(self.bn_fc2(self.fc2(x)))
            x = self.dropout2(x)
            
            x = self.relu(self.bn_fc3(self.fc3(x)))
            x = self.dropout3(x)
            
            x = self.fc_out(x)
            return x
else:
    class DeepfakeDetector:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable"""
        pass


class ModelUtils:
    """Utility class for model operations"""

    @staticmethod
    def load_model(model_path: str = None, device: Optional[Any] = None) -> Tuple[Any, Any, str]:
        """Load a trained model with error handling. Supports both binary and multi-class models."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Resolve paths relative to Backend directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        multiclass_path = os.path.join(backend_dir, '..', 'models', 'multiclass_detector.pth')
        binary_path = model_path or os.path.join(backend_dir, '..', 'models', 'xception_deepfake.pth')
        
        # Normalize paths
        multiclass_path = os.path.normpath(multiclass_path)
        binary_path = os.path.normpath(binary_path)
        
        model = None
        model_type = None
        
        # Try multi-class model
        if os.path.exists(multiclass_path):
            try:
                model = MultiClassDetector(num_classes=3)
                state_dict = torch.load(multiclass_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                print(f"Multi-class model loaded successfully from {multiclass_path}")
                model_type = "multiclass"
                return model, device, model_type
            except Exception as e:
                print(f"Warning: Could not load multi-class model: {e}")
                model = None
        
        # Fallback to binary model
        try:
            model = DeepfakeDetector()
            state_dict = torch.load(binary_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"Binary model loaded successfully from {binary_path}")
            model_type = "binary"
            return model, device, model_type
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @staticmethod
    def preprocess_image(image_path: str, image_size: int = 299) -> Tuple[Any, float]:
        """Preprocess image for model input and return preprocessing time"""
        start_time = time.time()
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        
        preprocessing_time = time.time() - start_time
        return tensor, preprocessing_time

    @staticmethod
    def predict_image(model: Any, image_tensor: Any, device: Any, model_type: str = "binary") -> Dict[str, Any]:
        """Make prediction with detailed information. Supports both binary and multi-class models."""
        start_time = time.time()
        
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            output = model(image_tensor)
            
        inference_time = time.time() - start_time
        
        if model_type == "multiclass":
            # Multi-class prediction
            logits = output
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities, dim=0).item()
            confidence = probabilities[predicted_class].item()
            
            class_names = ["Real", "Deepfake", "AIGenerated"]
            prediction = class_names[predicted_class]
            
            return {
                "prediction": prediction,
                "confidence": confidence * 100,
                "class_probabilities": {
                    "Real": float(probabilities[0].item()) * 100,
                    "Deepfake": float(probabilities[1].item()) * 100,
                    "AIGenerated": float(probabilities[2].item()) * 100
                },
                "inference_time_ms": round(inference_time * 1000, 2),
                "model_type": "multiclass"
            }
        else:
            # Binary prediction
            # Since model was trained with 0=Fake, 1=Real, the sigmoid output
            # represents probability of being Real. Therefore, the probability
            # of being Fake is 1.0 - output.item().
            confidence_raw = 1.0 - output.item()
            
            # Determine prediction
            prediction = "Fake" if confidence_raw > 0.5 else "Real"
            
            # Calculate confidence percentage (0-100%)
            if prediction == "Fake":
                confidence_percent = confidence_raw * 100
            else:
                confidence_percent = (1 - confidence_raw) * 100
            
            # Determine threat level
            if confidence_raw > 0.7:
                threat_level = "high"
            elif confidence_raw > 0.4:
                threat_level = "medium"
            else:
                threat_level = "low"
            
            return {
                "prediction": prediction,
                "confidence": confidence_percent,
                "confidence_raw": confidence_raw,
                "threat_level": threat_level,
                "inference_time_ms": round(inference_time * 1000, 2),
                "model_type": "binary"
            }

    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """Get comprehensive information about the model"""
        info = {
            "model_name": "Verifixia AI Xception",
            "version": "2.4.1",
            "architecture": "Xception-based CNN",
            "input_size": "299x299",
            "framework": "PyTorch",
        }
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            info.update({
                "exists": True,
                "size_mb": round(file_size, 2),
                "path": model_path,
                "status": "loaded"
            })
        else:
            info.update({
                "exists": False,
                "path": model_path,
                "status": "not_found"
            })
        
        return info

    @staticmethod
    def get_model_metadata(model: Any, device: Any) -> Dict[str, Any]:
        """Get detailed model metadata including parameter count"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(device),
            "layers": {
                "convolutional": 5,
                "batch_norm": 5,
                "fully_connected": 1,
                "dropout": 1
            },
            "architecture_details": {
                "entry_flow": "Conv2d(3→32→64)",
                "middle_flow": "Conv2d(64→128→256)",
                "exit_flow": "Conv2d(256→512)",
                "classifier": "FC(512→1) + Sigmoid"
            }
        }

    @staticmethod
    def interpret_confidence(confidence_raw: float) -> Dict[str, str]:
        """Interpret confidence score and provide detailed analysis"""
        if confidence_raw > 0.9:
            return {
                "level": "Very High",
                "description": "Strong indicators of deepfake manipulation detected",
                "recommendation": "Content should be flagged and reviewed"
            }
        elif confidence_raw > 0.7:
            return {
                "level": "High",
                "description": "Multiple deepfake artifacts identified",
                "recommendation": "Content likely manipulated, further analysis recommended"
            }
        elif confidence_raw > 0.5:
            return {
                "level": "Moderate",
                "description": "Some suspicious patterns detected",
                "recommendation": "Content may be manipulated, manual review suggested"
            }
        elif confidence_raw > 0.3:
            return {
                "level": "Low",
                "description": "Minimal deepfake indicators found",
                "recommendation": "Content appears mostly authentic"
            }
        else:
            return {
                "level": "Very Low",
                "description": "No significant manipulation detected",
                "recommendation": "Content appears authentic"
            }
