import os
import cv2
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd())
DATA_DIR = PROJECT_ROOT / "DATA"
CASCADE_PATH = PROJECT_ROOT / "scripts" / "haarcascade_frontalface_default.xml"

def crop_faces_in_folder(folder_path, detector):
    if not folder_path.exists():
        print(f"[-] Directory not found: {folder_path}")
        return
        
    print(f"\n[*] Processing folder: {folder_path.name}")
    
    total_images = 0
    face_detected = 0
    fallback_center = 0
    errors = 0
    
    # Supported extensions
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Process files
    for filepath in folder_path.iterdir():
        if filepath.suffix.lower() not in valid_exts:
            continue
            
        total_images += 1
        try:
            # Load image using OpenCV
            img = cv2.imread(str(filepath))
            if img is None:
                raise ValueError("Could not read image file")
                
            h, w, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            cropped = None
            if len(faces) > 0:
                # If multiple faces, choose the largest one
                largest_face = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                fx, fy, fw, fh = largest_face
                
                # Expand crop with 20% padding
                pad_w = int(fw * 0.2)
                pad_h = int(fh * 0.2)
                
                x1 = max(0, fx - pad_w)
                y1 = max(0, fy - pad_h)
                x2 = min(w, fx + fw + pad_w)
                y2 = min(h, fy + fh + pad_h)
                
                cropped = img[y1:y2, x1:x2]
                face_detected += 1
            else:
                # Fallback: Center crop (70% of dimensions)
                cw = int(w * 0.7)
                ch = int(h * 0.7)
                cx = (w - cw) // 2
                cy = (h - ch) // 2
                cropped = img[cy:cy+ch, cx:cx+cw]
                fallback_center += 1
                
            if cropped is None or cropped.size == 0:
                cropped = img # Last resort fallback: original image
                
            # Resize cropped face to exactly 256x256 using bilinear interpolation
            resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # Save back (overwrite)
            cv2.imwrite(str(filepath), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            if total_images % 1000 == 0:
                print(f"  Processed {total_images} images (Faces: {face_detected}, Fallbacks: {fallback_center})...")
                
        except Exception as e:
            errors += 1
            # print(f"  [Error] {filepath.name}: {e}")
            
    print(f"[OK] Completed {folder_path.name}: Total: {total_images}, Faces: {face_detected}, Fallbacks: {fallback_center}, Errors: {errors}")

def main():
    print("=" * 70)
    print("      Uniform Face-Cropping Dataset Pipeline (Haar Cascades)")
    print("=" * 70)
    
    if not CASCADE_PATH.exists():
        print(f"[-] Haar Cascade file not found at: {CASCADE_PATH}")
        return
        
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        print("[-] Error loading Haar Cascade Classifier")
        return
        
    folders = ["Real", "Deepfake", "AIGenerated"]
    for folder in folders:
        crop_faces_in_folder(DATA_DIR / folder, detector)
        
    print("\n" + "=" * 70)
    print("[OK] Dataset face-cropping complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
