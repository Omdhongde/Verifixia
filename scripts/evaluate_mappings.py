import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import MultiClassDetector
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../Backend'))
from utils.model_utils import MultiClassDetector, ModelUtils

def get_avg_probs(folder_path, model, count=50):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files = files[:count]
    
    all_probs = []
    
    for f in files:
        try:
            image = Image.open(f).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                all_probs.append(probs)
        except Exception as e:
            pass
            
    if not all_probs:
        return None
        
    return np.mean(all_probs, axis=0)

def main():
    backend_dir = os.path.join(os.path.dirname(__file__), '../Backend')
    multiclass_path = os.path.normpath(os.path.join(backend_dir, "..", "models", "multiclass_detector.pth"))
    
    if not os.path.exists(multiclass_path):
        print(f"Model path {multiclass_path} does not exist!")
        return
        
    model = MultiClassDetector(num_classes=3)
    state_dict = torch.load(multiclass_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Evaluating multiclass model on different categories...")
    
    data_dir = os.path.join(os.path.dirname(__file__), '../DATA')
    categories = ['Real', 'Deepfake', 'AIGenerated']
    
    for cat in categories:
        cat_path = os.path.join(data_dir, cat)
        if os.path.exists(cat_path):
            avg_probs = get_avg_probs(cat_path, model)
            if avg_probs is not None:
                print(f"\nAverage probabilities for images in folder '{cat}':")
                print(f"  Class 0 (assumed Real)       : {avg_probs[0]*100:.2f}%")
                print(f"  Class 1 (assumed Deepfake)   : {avg_probs[1]*100:.2f}%")
                print(f"  Class 2 (assumed AIGenerated): {avg_probs[2]*100:.2f}%")
            else:
                print(f"No valid images found in {cat_path}")
        else:
            print(f"Category directory {cat_path} does not exist")

if __name__ == '__main__':
    main()
