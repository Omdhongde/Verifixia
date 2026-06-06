import os
import torch
from PIL import Image
from torchvision import transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import MultiClassDetector
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../Backend'))
from utils.model_utils import MultiClassDetector

def evaluate_file(f_path, model):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(f_path):
        print(f"File {f_path} does not exist!")
        return
        
    try:
        image = Image.open(f_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predicted_class = torch.argmax(logits, dim=1).item()
            class_names = ["Real", "Deepfake", "AIGenerated"]
            print(f"File: {os.path.basename(f_path)}")
            print(f"  Prediction: {class_names[predicted_class]}")
            print(f"  Class 0 (Real)       : {probs[0]*100:.2f}%")
            print(f"  Class 1 (Deepfake)   : {probs[1]*100:.2f}%")
            print(f"  Class 2 (AIGenerated): {probs[2]*100:.2f}%")
    except Exception as e:
        print(f"Error evaluating {f_path}: {e}")

def main():
    backend_dir = os.path.join(os.path.dirname(__file__), '../Backend')
    multiclass_path = os.path.normpath(os.path.join(backend_dir, "..", "models", "multiclass_detector.pth"))
    
    model = MultiClassDetector(num_classes=3)
    state_dict = torch.load(multiclass_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    data_dir = os.path.join(os.path.dirname(__file__), '../DATA')
    
    files_to_test = [
        os.path.join(data_dir, 'Real/Real_308.jpg'),
        os.path.join(data_dir, 'Real/Real_433.jpg'),
        os.path.join(data_dir, 'Real/Real_418.jpg'),
        os.path.join(data_dir, 'Real/Real_317.jpg'),
        os.path.join(data_dir, 'Real/Real_5.jpg'),
        os.path.join(data_dir, 'Real/Real_495.jpg'),
    ]
    
    for f in files_to_test:
        evaluate_file(f, model)

if __name__ == '__main__':
    main()
