"""
CondSeg Video Inference Script
================================
Loads a trained CondSeg checkpoint and runs frame-by-frame inference
on a video file. Displays the 1024×1024 letterboxed input with overlays
drawn directly on top (exactly what the model sees).

Usage:
    python inference.py --checkpoint best.pt --video path/to/video.mov
    python inference.py --checkpoint best.pt --video path/to/video.mov --output result.mp4
    python inference.py --checkpoint best.pt --video path/to/video.mov --no-display
"""

import argparse
import sys
import os
import time
import math

import cv2
import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, ".")
from condseg import CondSeg


# ── ImageNet normalization constants ──
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="CondSeg Video Inference")
    p.add_argument("--checkpoint", type=str, default="best.pt",
                   help="Path to trained checkpoint (.pt)")
    p.add_argument("--video", type=str, required=True,
                   help="Path to input video file")
    p.add_argument("--output", type=str, default=None,
                   help="Path to save output video (default: input_condseg.mp4)")
    p.add_argument("--img_size", type=int, default=1024,
                   help="Model input resolution (must match training)")
    p.add_argument("--no-display", action="store_true",
                   help="Disable live window display")
    p.add_argument("--device", type=str, default=None,
                   help="Device: cuda or cpu (auto-detect if omitted)")
    return p.parse_args()


def load_model(checkpoint_path: str, img_size: int, device: torch.device) -> CondSeg:
    """Load CondSeg model from checkpoint."""
    model = CondSeg(img_size=img_size, epsilon=0.01, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    model = model.to(device).eval()
    print(f"  ✓ Model loaded from {checkpoint_path}")
    if "epoch" in ckpt:
        print(f"    Epoch: {ckpt['epoch']}, Best val loss: {ckpt.get('best_val_loss', 'N/A')}")
    return model


def letterbox_frame(frame_bgr: np.ndarray, img_size: int) -> np.ndarray:
    """
    Resize frame preserving aspect ratio, pad to square with edge replication.
    Fast: uses cv2.copyMakeBorder + seam blur (~1ms).
    Returns the 1024×1024 square image.
    """
    h, w = frame_bgr.shape[:2]
    scale = img_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top    = (img_size - new_h) // 2
    pad_bottom = img_size - new_h - pad_top
    pad_left   = (img_size - new_w) // 2
    pad_right  = img_size - new_w - pad_left

    square = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )

    # Light blur on padding seams
    blur_band = 8
    if pad_top > 0:
        y1, y2 = max(0, pad_top - blur_band), min(img_size, pad_top + blur_band)
        square[y1:y2] = cv2.GaussianBlur(square[y1:y2], (1, 15), 0)
    if pad_bottom > 0:
        y_loc = pad_top + new_h
        y1, y2 = max(0, y_loc - blur_band), min(img_size, y_loc + blur_band)
        square[y1:y2] = cv2.GaussianBlur(square[y1:y2], (1, 15), 0)

    return square


def preprocess_square(square_bgr: np.ndarray) -> torch.Tensor:
    """Normalize a 1024×1024 BGR image and convert to BCHW tensor."""
    rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def draw_overlay(square_bgr: np.ndarray, outputs: dict) -> np.ndarray:
    """
    Draw model outputs directly onto the 1024×1024 model input image.
    - Eye region: green overlay
    - Visible iris: blue overlay
    - Iris ellipse: yellow contour
    - Params text
    """
    canvas = square_bgr.copy().astype(np.float32)

    # Extract masks (already 1024×1024 — same space as the image)
    pred_eye  = outputs["predicted_eye_region_mask"][0, 0].cpu().numpy()
    pred_iris = outputs["predicted_visible_iris"][0, 0].cpu().numpy()
    iris_params = outputs["iris_params"][0].cpu().numpy()

    # Eye region: green
    eye_alpha = (pred_eye * 0.3)[:, :, np.newaxis]
    canvas = canvas * (1 - eye_alpha) + np.array([0, 200, 0], dtype=np.float32) * eye_alpha

    # Iris: blue
    iris_alpha = (pred_iris * 0.45)[:, :, np.newaxis]
    canvas = canvas * (1 - iris_alpha) + np.array([255, 100, 0], dtype=np.float32) * iris_alpha

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # Ellipse contour (params are already in 1024×1024 space)
    x0, y0, a, b, theta = iris_params
    center = (int(x0), int(y0))
    axes   = (max(1, int(a)), max(1, int(b)))
    angle  = math.degrees(theta)
    cv2.ellipse(canvas, center, axes, angle, 0, 360, (0, 255, 255), 2)

    # Text
    info = f"a={a:.0f} b={b:.0f} th={math.degrees(theta):.1f}"
    cv2.putText(canvas, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return canvas


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("CondSeg Video Inference")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Video      : {args.video}")
    print(f"  Img size   : {args.img_size}")
    print()

    # ── Load model ──
    model = load_model(args.checkpoint, args.img_size, device)

    # ── Open video ──
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video      : {W}×{H} @ {fps:.1f} FPS, {total} frames")

    # ── Output video writer (1024×1024, XVID — works everywhere) ──
    if args.output is None:
        base = os.path.splitext(args.video)[0]
        args.output = base + "_condseg.avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_writer = cv2.VideoWriter(args.output, fourcc, fps,
                                 (args.img_size, args.img_size))
    if not out_writer.isOpened():
        print("ERROR: Cannot open VideoWriter. Check codec support.")
        sys.exit(1)
    print(f"  Output     : {args.output}  ({args.img_size}×{args.img_size})")
    print()

    # ── Inference loop ──
    frame_idx = 0
    t_start = time.time()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Letterbox → 1024×1024 (exactly what model sees)
            square = letterbox_frame(frame, args.img_size)

            # Preprocess + forward
            tensor = preprocess_square(square).to(device)
            with autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(tensor)

            # Draw overlays on the 1024×1024 image
            vis = draw_overlay(square, outputs)

            # Write
            out_writer.write(vis)

            # Display
            if not args.no_display:
                cv2.imshow("CondSeg Inference", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n  Interrupted by user (q)")
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - t_start
                current_fps = frame_idx / elapsed
                print(f"  Frame {frame_idx}/{total}  "
                      f"({frame_idx/total*100:.1f}%)  "
                      f"{current_fps:.1f} FPS", end="\r")

    # ── Cleanup ──
    cap.release()
    out_writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\n\n  Done! {frame_idx} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")
    print(f"  Output saved to: {args.output}")


if __name__ == "__main__":
    main()
