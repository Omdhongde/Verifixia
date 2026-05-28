"""
Verifixia - Video Frame Extraction & Face Cropping Pipeline
============================================================
Extracts individual frames from a local video file, runs Haar Cascade
face detection, crops detected faces, and resizes them to the 299x299
classifier input dimensions.

Saves cropped face frames to DATA/Video/extracted_faces/ for visual review
or model training.
"""

import os
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

def extract_faces_from_video(video_path: str, output_dir: str, max_frames: int = 50, frame_interval: int = 5):
    """
    Reads video frames, crops faces using Haar Cascades, and saves them.
    """
    if not _OPENCV_AVAILABLE:
        print("[FAIL] OpenCV ('cv2') or PIL is not installed. Please install opencv-python and pillow first.")
        return False

    vpath = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not vpath.exists():
        print(f"[FAIL] Video file not found: {video_path}")
        return False

    print("=" * 65)
    print("  Verifixia - Video Face Extraction Pipeline")
    print("=" * 65)
    print(f"  Source Video : {vpath.name}")
    print(f"  Output Dir   : {out_dir.resolve()}")
    print(f"  Frame Step   : Every {frame_interval} frames")
    print("=" * 65)

    # Load Haar Cascade face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[FAIL] Could not load Haar Cascade face detector!")
        return False

    # Open video
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"[FAIL] Could not open video file: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"[INFO] Video loaded successfully. Duration: {duration:.2f}s, Total Frames: {total_frames}, FPS: {fps:.1f}")

    frame_count = 0
    saved_count = 0
    extracted_faces_num = 0

    print("\nProcessing frames ...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process at selected intervals
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces (scaleFactor and minNeighbors configured for general reliability)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            for i, (x, y, w, h) in enumerate(faces):
                # Apply slight padding to capture full head shape
                pad_h = int(h * 0.15)
                pad_w = int(w * 0.15)
                
                y1 = max(0, y - pad_h)
                y2 = min(frame.shape[0], y + h + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(frame.shape[1], x + w + pad_w)

                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue

                # Resize to standard model dimensions (299x299)
                face_resized = cv2.resize(face_crop, (299, 299), interpolation=cv2.INTER_CUBIC)
                
                # Save crop
                face_filename = f"face_frame_{frame_count:04d}_crop_{i}.jpg"
                cv2.imwrite(str(out_dir / face_filename), face_resized)
                extracted_faces_num += 1

            saved_count += 1
            print(f"  Frame {frame_count:5d}/{total_frames:5d} processed. Faces detected: {len(faces)}", flush=True)

            if saved_count >= max_frames:
                print(f"[INFO] Reached max target frames threshold ({max_frames}). Stopping.")
                break

        frame_count += 1

    cap.release()
    print("=" * 65)
    print(f"SUCCESS: Face frame extraction completed!")
    print(f"  Total frames analyzed  : {saved_count}")
    print(f"  Total faces extracted  : {extracted_faces_num}")
    print(f"  Saved directory        : {out_dir.resolve()}")
    print("=" * 65)
    return True

if __name__ == "__main__":
    # Configure path relative to repo root
    root = Path(__file__).resolve().parent.parent
    sample_video = root / "DATA" / "Video" / "DeeperForensics-1.0-master" / "perturbation" / "data" / "input.mp4"
    output_faces = root / "DATA" / "Video" / "extracted_faces"
    
    success = extract_faces_from_video(
        video_path=str(sample_video),
        output_dir=str(output_faces),
        max_frames=30,      # limit processing to 30 frames for fast testing
        frame_interval=4    # extract every 4th frame
    )
    sys.exit(0 if success else 1)
