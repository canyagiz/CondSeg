"""
CondSeg Video Inference Script
================================
Loads a trained CondSeg checkpoint and runs frame-by-frame inference
on a video file. Displays the 1024×1024 letterboxed input with overlays
drawn directly on top (exactly what the model sees).

Usage:
    python inference.py --checkpoint best.pt --video path/to/video.mov
    python inference.py --checkpoint best.pt --video path/to/video.mov --save_video
    python inference.py --checkpoint best.pt --video path/to/video.mov --no_display --half
    python inference.py --checkpoint best.pt --video path/to/video.mov --output_dir results/
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
    p.add_argument("--checkpoint", type=str, default="models/best_2.pt",
                   help="Path to trained checkpoint (.pt)")
    p.add_argument("--video", type=str, required=True,
                   help="Path to input video file")
    p.add_argument("--output", type=str, default=None,
                   help="Path to save output video (default: input_condseg.avi)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to save CSV signals (default: next to video)")
    p.add_argument("--img_size", type=int, default=1024,
                   help="Model input resolution (must match training)")
    p.add_argument("--save_video", action="store_true",
                   help="Save overlay video to disk")
    p.add_argument("--no-display", "--no_display", action="store_true",
                   help="Disable live window display")
    p.add_argument("--half", action="store_true",
                   help="Use FP16 (half precision) for faster inference")
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

    # ── Resolve output directory ──
    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.video))
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CondSeg Video Inference")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Video      : {args.video}")
    print(f"  Img size   : {args.img_size}")
    print(f"  Half (FP16): {args.half}")
    print(f"  Save video : {args.save_video}")
    print(f"  Output dir : {args.output_dir}")
    print()

    # ── Load model ──
    model = load_model(args.checkpoint, args.img_size, device)
    if args.half and device.type == "cuda":
        model = model.half()
        print("  ✓ Model converted to FP16 (half precision)")

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

    # ── Output video writer (optional) ──
    out_writer = None
    if args.save_video:
        if args.output is None:
            args.output = os.path.join(args.output_dir, video_basename + "_condseg.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_writer = cv2.VideoWriter(args.output, fourcc, fps,
                                     (args.img_size, args.img_size))
        if not out_writer.isOpened():
            print("ERROR: Cannot open VideoWriter. Check codec support.")
            sys.exit(1)
        print(f"  Video out  : {args.output}  ({args.img_size}×{args.img_size})")
    print()

    # ── Inference loop ──
    frame_idx = 0
    t_start = time.time()
    all_params = []  # collect (x0, y0, a, b, theta) per frame

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Letterbox → 1024×1024 (exactly what model sees)
            square = letterbox_frame(frame, args.img_size)

            # Preprocess + forward
            tensor = preprocess_square(square).to(device)
            if args.half:
                tensor = tensor.half()
            outputs = model(tensor)

            # Collect ellipse params
            iris_params = outputs["iris_params"][0].cpu().numpy()
            all_params.append(iris_params.copy())

            # Draw + display / write
            if not args.no_display or out_writer is not None:
                vis = draw_overlay(square, outputs)
                if out_writer is not None:
                    out_writer.write(vis)
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
    if out_writer is not None:
        out_writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # ── Save ellipse params as separate .npy signals ──
    if len(all_params) > 0:
        params_array = np.array(all_params)  # (N, 5)
        signal_names = ["x0", "y0", "a", "b", "theta_rad"]
        print()
        for i, name in enumerate(signal_names):
            npy_path = os.path.join(args.output_dir, f"{video_basename}_{name}.npy")
            np.save(npy_path, params_array[:, i])
            print(f"  ✓ {name:>9s} → {npy_path}  ({len(all_params)} samples)")


    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\n  Done! {frame_idx} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")
    if out_writer is not None:
        print(f"  Video saved to: {args.output}")


if __name__ == "__main__":
    main()
