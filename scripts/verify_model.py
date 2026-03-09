"""Quick script to verify that the backend can load models and make a sample prediction.

Usage:
    python verify_model.py /path/to/sample/image.jpg

This is useful when deploying to ensure the binary weights are present and the
ML stack is functioning.  It does not start the server; it just imports the
same utilities used by `app.py`.
"""

import sys
import os

# make sure we can import the backend package path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Backend")))

from app import predict_deepfake, PYTORCH_AVAILABLE, SKLEARN_AVAILABLE


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_model.py /path/to/sample.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    print("PyTorch available:", PYTORCH_AVAILABLE)
    print("scikit-learn available:", SKLEARN_AVAILABLE)

    result = predict_deepfake(image_path)
    print("Prediction result:")
    print(result)


if __name__ == "__main__":
    main()
