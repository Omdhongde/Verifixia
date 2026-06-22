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

    class CNNLSTMDetector(nn.Module):
        """CNN-LSTM Detector: Extracts frame features using a 2D CNN, 
        then uses an LSTM to analyze temporal relations across sequence of frames."""
        def __init__(self, latent_dim=512, lstm_hidden_dim=256, lstm_layers=1):
            super().__init__()
            # Use the feature extraction part of DeepfakeDetector
            base_model = DeepfakeDetector()
            self.cnn = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
            )
            self.avgpool = base_model.avg_pool
            self.maxpool = base_model.max_pool
            
            self.lstm = nn.LSTM(
                input_size=latent_dim * 2, # average pooling (512) + max pooling (512) = 1024
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=False
            )
            
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            # Input shape: (Batch, SequenceLength, Channels, Height, Width)
            batch_size, seq_len, c, h, w = x.size()
            
            # Reshape input to (Batch * SequenceLength, Channels, Height, Width) to pass through CNN
            x = x.view(batch_size * seq_len, c, h, w)
            
            # CNN feature extraction
            features = self.cnn(x)
            avg_feat = self.avgpool(features)
            max_feat = self.maxpool(features)
            features = torch.cat([avg_feat, max_feat], dim=1)
            features = features.view(batch_size * seq_len, -1)
            
            # Reshape back to sequence form: (Batch, SequenceLength, FeatureDim)
            features = features.view(batch_size, seq_len, -1)
            
            # LSTM processing
            lstm_out, (hn, cn) = self.lstm(features)
            
            # Take final sequence step hidden state
            out = lstm_out[:, -1, :]
            
            # Classifier prediction
            return self.fc(out)

    import torchvision.models as models

    class FrequencyAttentionGate(nn.Module):
        def __init__(self, in_channels=24):
            super().__init__()
            self.mask = nn.Parameter(torch.randn(1, 1, 56, 56))
            self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            
        def forward(self, x):
            mask = torch.sigmoid(self.mask)
            if x.shape[2:] != mask.shape[2:]:
                mask = nn.functional.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)
            masked_x = x * mask
            fused = torch.cat([x, masked_x], dim=1)
            out = self.conv1x1(fused)
            return out

    class BiologicalPlausibilityModule(nn.Module):
        def __init__(self, in_channels=112):
            super().__init__()
            self.proj = nn.Conv2d(in_channels, 64, kernel_size=1) if in_channels != 64 else nn.Identity()
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            
            self.regions = {
                'forehead': [0.05, 0.30, 0.25, 0.70],
                'left_cheek': [0.45, 0.15, 0.70, 0.45],
                'right_cheek': [0.45, 0.55, 0.70, 0.85],
                'nose_bridge': [0.30, 0.40, 0.55, 0.60],
                'chin': [0.75, 0.35, 0.95, 0.65]
            }
            
            self.correlation_mlp = nn.Sequential(
                nn.Linear(25, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            self.bio_embed = nn.Sequential(
                nn.Linear(5 * 3136, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 32)
            )
            
        def forward(self, x):
            x_proj = self.proj(x)
            
            region_vecs = []
            for name, box in self.regions.items():
                h, w = x_proj.shape[2], x_proj.shape[3]
                y_start = int(box[0] * h)
                y_end = int(box[2] * h)
                x_start = int(box[1] * w)
                x_end = int(box[3] * w)
                y_start = max(0, min(y_start, h - 1))
                y_end = max(y_start + 1, min(y_end, h))
                x_start = max(0, min(x_start, w - 1))
                x_end = max(x_start + 1, min(x_end, w))
                
                crop = x_proj[:, :, y_start:y_end, x_start:x_end]
                pooled = self.pool(crop)
                flat = pooled.view(pooled.size(0), -1)
                region_vecs.append(flat)
                
            V = torch.stack(region_vecs, dim=1)
            
            V_mean = V.mean(dim=2, keepdim=True)
            V_std = V.std(dim=2, keepdim=True) + 1e-6
            V_norm = (V - V_mean) / V_std
            corr_matrix = torch.bmm(V_norm, V_norm.transpose(1, 2)) / 3136.0
            
            corr_flat = corr_matrix.view(corr_matrix.size(0), -1)
            consistency_score = self.correlation_mlp(corr_flat)
            
            flat_concat = V.view(V.size(0), -1)
            bio_emb = self.bio_embed(flat_concat)
            
            return consistency_score, bio_emb

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)
            
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg()

    class CrossAttentionFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_A = nn.Linear(512, 256)
            self.proj_B = nn.Linear(320, 256)
            self.proj_C = nn.Linear(32, 256)
            self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
            self.ln = nn.LayerNorm(256)
            
        def forward(self, A, B, C):
            proj_a = self.proj_A(A).unsqueeze(1)
            proj_b = self.proj_B(B).unsqueeze(1)
            proj_c = self.proj_C(C).unsqueeze(1)
            
            keys_values = torch.cat([proj_b, proj_c], dim=1)
            attn_out, _ = self.cross_attn(query=proj_a, key=keys_values, value=keys_values)
            fused = self.ln(proj_a + attn_out).squeeze(1)
            return fused

    class AdvancedCNNLSTMDetector(nn.Module):
        def __init__(self, use_pretrained=False):
            super().__init__()
            if use_pretrained:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            else:
                weights = None
            
            efficientnet = models.efficientnet_b0(weights=weights)
            features = efficientnet.features
            
            self.stage1 = nn.Sequential(
                features[0],
                features[1],
                features[2],
            )
            self.fag = FrequencyAttentionGate(in_channels=24)
            
            self.stage2 = nn.Sequential(
                features[3],
                features[4],
                features[5],
            )
            self.bpm = BiologicalPlausibilityModule(in_channels=112)
            
            self.stage3 = nn.Sequential(
                features[6],
                features[7],
            )
            self.head_conv = features[8]
            self.compress_head = nn.Conv2d(1280, 640, kernel_size=1)
            
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            
            self.lstm1 = nn.LSTM(input_size=672, hidden_size=512, batch_first=True, bidirectional=False)
            self.dropout1 = nn.Dropout(0.3)
            self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=True)
            self.dropout2 = nn.Dropout(0.3)
            
            self.temporal_self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
            self.layer_norm_self_attn = nn.LayerNorm(512)
            
            self.theta = nn.Parameter(torch.tensor(0.5))
            self.fusion = CrossAttentionFusion()
            
            self.verdict_layer = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2)
            )
            
        def forward(self, x):
            batch_size, seq_len, c, h, w = x.size()
            x_flat = x.view(batch_size * seq_len, c, h, w)
            
            feat1 = self.stage1(x_flat)
            feat1 = self.fag(feat1)
            
            feat2 = self.stage2(feat1)
            consistency_score, bio_emb = self.bpm(feat2)
            
            feat3 = self.stage3(feat2)
            
            anomaly_feat = GradReverse.apply(feat3)
            anomaly_feat = self.avgpool(anomaly_feat).view(batch_size * seq_len, -1)
            
            head_feat = self.head_conv(feat3)
            head_feat = self.compress_head(head_feat)
            spatial_vec = self.avgpool(head_feat).view(batch_size * seq_len, -1)
            
            spatial_bio_vec = torch.cat([spatial_vec, bio_emb], dim=1)
            
            seq_features = spatial_bio_vec.view(batch_size, seq_len, -1)
            
            lstm_out1, _ = self.lstm1(seq_features)
            lstm_out1 = self.dropout1(lstm_out1)
            
            lstm_out2, _ = self.lstm2(lstm_out1)
            lstm_out2 = self.dropout2(lstm_out2)
            
            attn_out, _ = self.temporal_self_attn(lstm_out2, lstm_out2, lstm_out2)
            attn_out = self.layer_norm_self_attn(lstm_out2 + attn_out)
            
            deltas = torch.norm(attn_out[:, 1:] - attn_out[:, :-1], p=2, dim=-1)
            spike_flags = deltas > self.theta
            
            query = attn_out[:, -1, :]
            
            avg_anomaly = anomaly_feat.view(batch_size, seq_len, -1).mean(dim=1)
            avg_bio_emb = bio_emb.view(batch_size, seq_len, -1).mean(dim=1)
            
            fused = self.fusion(query, avg_anomaly, avg_bio_emb)
            logits = self.verdict_layer(fused)
            
            return logits, consistency_score, spike_flags, deltas
else:
    class DeepfakeDetector:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable"""
        pass
    class MultiClassDetector:
        pass
    class CNNLSTMDetector:
        pass
    class FrequencyAttentionGate:
        pass
    class BiologicalPlausibilityModule:
        pass
    class CrossAttentionFusion:
        pass
    class AdvancedCNNLSTMDetector:
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
        """Preprocess image for model input and return preprocessing time.
        Automatically detects and crops the face, as the models were trained on faces.
        Falls back to full image if no face is detected.
        """
        start_time = time.time()
        
        # Default to loading full image via PIL
        image = None
        
        try:
            import cv2
            img_bgr = cv2.imread(image_path)
            if img_bgr is not None:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(cascade_path)
                
                if not face_cascade.empty():
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                    )
                    if len(faces) > 0:
                        # Get the largest face
                        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                        x, y, w, h = faces[0]
                        
                        # Add slight padding
                        pad_h = int(h * 0.15)
                        pad_w = int(w * 0.15)
                        y1 = max(0, y - pad_h)
                        y2 = min(img_bgr.shape[0], y + h + pad_h)
                        x1 = max(0, x - pad_w)
                        x2 = min(img_bgr.shape[1], x + w + pad_w)
                        
                        face_crop = img_bgr[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(img_rgb)
        except Exception as e:
            print(f"Face crop failed, falling back to full image: {e}")
            
        # Fallback to full image if face cropping failed or wasn't found
        if image is None:
            image = Image.open(image_path).convert('RGB')
            
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
            
            # Apply Temperature Scaling to soften overconfident predictions
            # T > 1.0 reduces confidence (e.g. 99.9% -> ~80-90%)
            temperature = 2.5
            scaled_logits = logits / temperature
            
            probabilities = torch.softmax(scaled_logits, dim=1)[0]
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
            # WARNING: Check which script trained the binary model weights (xception_deepfake.pth).
            # - train_deeperforensics.py streams HF dataset and uses: 0 = Fake, 1 = Real.
            #   For this model, the sigmoid output represents the probability of being Real,
            #   so the probability of being Fake is 1.0 - output.item().
            # - train.py uses: 0 = Real, 1 = Fake (EfficientNet pattern).
            # The logic below assumes the model was trained with the train_deeperforensics.py mapping (0 = Fake, 1 = Real).
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
