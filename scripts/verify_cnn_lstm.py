import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../Backend'))
from utils.model_utils import CNNLSTMDetector

def test():
    print("Testing CNNLSTMDetector shape flow...")
    # Batch size = 2, Sequence Length = 25, 3 channels, 299x299
    dummy_input = torch.randn(2, 25, 3, 299, 299)
    
    # Instantiate model
    model = CNNLSTMDetector()
    
    # Run forward pass
    output = model(dummy_input)
    
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, 1), f"Output shape was {output.shape}, expected (2, 1)"
    print("[OK] Shape flow is correct!")

if __name__ == '__main__':
    test()
