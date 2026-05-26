#!/usr/bin/env python3
"""Test multiclass detection API"""
import requests
import json
from pathlib import Path

test_paths = [
    ('DATA/Real', 50),
    ('DATA/Deepfake', 200),
]

for test_dir, idx in test_paths:
    img_files = list(Path(test_dir).glob('*.jpg'))
    if idx < len(img_files):
        img_path = img_files[idx]
        print(f'\nTesting: {img_path.parent.name}/{img_path.name}')
        with open(img_path, 'rb') as f:
            files = {'file': f}
            r = requests.post('http://localhost:3001/api/upload', files=files)
            if r.status_code == 200:
                result = r.json()
                # Show key fields
                print(f"  Prediction: {result.get('prediction')}")
                print(f"  Confidence: {result.get('confidence')}%")
                print(f"  Model Used: {result.get('model_used')}")
                if result.get('class_probabilities'):
                    print(f"  Class Probabilities:")
                    for cls, prob in result.get('class_probabilities').items():
                        print(f"    {cls}: {prob:.2f}%")
                else:
                    print('  ⚠️ No class_probabilities in response!')
            else:
                print(f'Error: {r.status_code}')
