#!/usr/bin/env python3
"""
Iris Stabilization Script
=========================
Reverses the synthetic gaze motion and head motion applied by simulate_movement.py,
producing a video where the iris appears stationary.

The forward pipeline applies transformations in this order:
  1. OPA radial warp (iris pulsation)
  2. Gaze affine (rotate + scale + translate) on iris region only
  3. Head motion (global translate on entire frame)

To invert, we reverse the order:
  Step 1: Undo head motion  → global translate by (-head_dx, -head_dy)
  Step 2: Undo gaze affine  → invertAffineTransform(M) on iris region only
  Step 3: (OPA warp remains — that's the signal we want to keep)

Iris mask and center are detected automatically using CondSeg model.

Inputs:
  - The warped video (with gaze + head motion)
  - gaze_trajectory.json  (dx, dy, torsion_deg, scale per frame)
  - head_motion.npy        (Nx2 array of [head_dx, head_dy] per frame)
  - CondSeg checkpoint     (for automatic iris mask + center detection)

Output:
  - Stabilized video where iris is back in its original position

Usage (fully automatic):
    python stabilize_iris.py \\
        --video opa_sim_outputs/.../dpx0p30_fhz1p20_head_gaze.mov \\
        --checkpoint models/best_2.pt

    # gaze_trajectory.json and head_motion.npy are auto-detected from
    # the same directory as the video.
    # iris mask and center are auto-detected from the first frame via CondSeg.
"""

import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# CondSeg model (lazy import — loaded only when needed)
# ─────────────────────────────────────────────────────────
_condseg_model = None  # cached after first load

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_CHECKPOINT = "models/CNN/best_3.pt"


# ─────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────
def _interpolate_invalid(arr: np.ndarray, valid: np.ndarray) -> None:
    """Linearly interpolate invalid frames in-place using neighboring valid frames."""
    if valid.all():
        return
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return
    invalid_idx = np.where(~valid)[0]
    arr[invalid_idx] = np.interp(invalid_idx, valid_idx, arr[valid_idx])


def load_gaze_trajectory(json_path: str) -> dict:
    """
    Load gaze trajectory — supports two formats:

    Synthetic (simulate_movement.py output):
        {"dx": [...], "dy": [...], "torsion_deg": [...], "scale": [...]}

    Real-life (per-frame tracking output):
        {"reference": {"cx": ..., "cy": ...}, "frames": [
            {"frame_idx": 0, "dx": 0.0, "dy": 0.0, "torsion_deg": 0.0,
             "scale": 1.0, "iris_r": ..., "valid": true}, ...]}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # ── Real-life format: has a 'frames' list ──
    if 'frames' in data:
        frames = data['frames']
        n = data.get('n_frames', len(frames))

        dx      = np.zeros(n, dtype=np.float32)
        dy      = np.zeros(n, dtype=np.float32)
        torsion = np.zeros(n, dtype=np.float32)
        scale   = np.ones(n,  dtype=np.float32)
        valid   = np.zeros(n, dtype=bool)

        for f in frames:
            idx = f.get('frame_idx', 0)
            if idx >= n:
                continue
            dx[idx]      = f.get('dx', 0.0)
            dy[idx]      = f.get('dy', 0.0)
            torsion[idx] = f.get('torsion_deg', 0.0)
            scale[idx]   = f.get('scale', 1.0)
            valid[idx]   = f.get('valid', True)

        # Linearly interpolate over invalid frames
        _interpolate_invalid(dx,      valid)
        _interpolate_invalid(dy,      valid)
        _interpolate_invalid(torsion, valid)
        _interpolate_invalid(scale,   valid)

        n_invalid = int((~valid).sum())
        if n_invalid:
            print(f"    {n_invalid} invalid frames interpolated")

        result = {
            'dx': dx, 'dy': dy, 'torsion_deg': torsion, 'scale': scale,
            'valid': valid,
        }
        # Expose reference center so main() can use it when cx/cy are not set
        ref = data.get('reference', {})
        if 'cx' in ref and 'cy' in ref:
            result['reference_cx'] = float(ref['cx'])
            result['reference_cy'] = float(ref['cy'])
        return result

    # ── Synthetic format: parallel top-level arrays ──
    return {
        'dx':         np.array(data['dx'],         dtype=np.float32),
        'dy':         np.array(data['dy'],         dtype=np.float32),
        'torsion_deg': np.array(data['torsion_deg'], dtype=np.float32),
        'scale':      np.array(data['scale'],      dtype=np.float32),
    }


def load_head_motion(npy_path: str) -> tuple:
    """Load head motion trajectory from NPY file saved by simulate_movement.py."""
    hm = np.load(npy_path)  # shape (N, 2) → columns are [dx, dy]
    return hm[:, 0].astype(np.float32), hm[:, 1].astype(np.float32)


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed on {video_path}: {result.stderr}")
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError(f"No video stream found in {video_path}")
    stream = info["streams"][0]

    fps_str = stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "n_frames": int(stream.get("nb_frames", 0)),
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "profile": stream.get("profile"),
    }


# ─────────────────────────────────────────────────────────
# CondSeg-based auto-detection (self-contained, no inference.py import)
# ─────────────────────────────────────────────────────────
def _letterbox(frame_bgr: np.ndarray, img_size: int = 1024):
    """
    Resize frame preserving aspect ratio, pad to square.
    Returns (square_bgr, scale, pad_left, pad_top).
    """
    h, w = frame_bgr.shape[:2]
    scale = img_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top    = (img_size - new_h) // 2
    pad_bottom = img_size - new_h - pad_top
    pad_left   = (img_size - new_w) // 2
    pad_right  = img_size - new_w - pad_left

    square = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )
    return square, scale, pad_left, pad_top


def _load_condseg(checkpoint: str, device: str = "cpu") -> "torch.nn.Module":
    """Load CondSeg model (cached after first call)."""
    global _condseg_model
    if _condseg_model is not None:
        return _condseg_model

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from condseg import CondSeg

    model = CondSeg(img_size=1024, epsilon=0.01, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    _condseg_model = model
    print(f"  ✓ CondSeg model loaded from {checkpoint}")
    return model


def detect_iris_condseg(
    video_path: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    frame_index: int = 0,
    img_size: int = 1024,
) -> dict:
    """
    Detect iris mask + center from a video frame using CondSeg.

    Self-contained: all preprocessing is done locally (no inference.py import).

    The model operates in 1024×1024 letterbox space.  Results are mapped
    back to the original video coordinate system so that the downstream
    stabilization (which works in original resolution) can use them directly.

    Args:
        video_path:   Path to input video
        checkpoint:   Path to CondSeg checkpoint (.pt)
        device:       "cpu" or "cuda"
        frame_index:  Which frame to use for detection (default: 0)
        img_size:     Model input size (default: 1024)

    Returns:
        dict with keys: iris_mask (H×W uint8), cx, cy, radius_px
        or None if detection fails
    """
    import torch

    # ── Extract frame ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  Error: Could not read frame {frame_index} from {video_path}")
        return None

    orig_h, orig_w = frame.shape[:2]
    print(f"\n[CondSeg] Detecting iris from frame {frame_index}...")
    print(f"  Frame size: {orig_w}×{orig_h}")

    # ── Letterbox → model space ──
    square, lb_scale, pad_left, pad_top = _letterbox(frame, img_size)

    # ── Preprocess (ImageNet normalize → BCHW tensor) ──
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # ── Inference ──
    model = _load_condseg(checkpoint, device)
    with torch.no_grad():
        outputs = model(tensor)

    # ── Extract results (in 1024×1024 letterbox space) ──
    iris_params = outputs["iris_params"][0].cpu().numpy()  # [x0, y0, a, b, θ]
    vis_iris = outputs["predicted_visible_iris"][0, 0].cpu().numpy()  # (1024, 1024)

    x0_lb, y0_lb, a_lb, b_lb, theta = iris_params

    # ── Map from letterbox space → original video coordinates ──
    cx = (x0_lb - pad_left) / lb_scale
    cy = (y0_lb - pad_top)  / lb_scale
    a_orig = a_lb / lb_scale
    b_orig = b_lb / lb_scale
    radius_px = (a_orig + b_orig) / 2.0

    # ── Map mask back to original resolution ──
    # 1. Crop out padding
    new_h = int(orig_h * lb_scale)
    new_w = int(orig_w * lb_scale)
    mask_cropped = vis_iris[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    # 2. Resize to original
    mask_orig = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # 3. Binarize (threshold at 0.5)
    iris_mask = (mask_orig > 0.5).astype(np.uint8) * 255

    print(f"  Detected iris:")
    print(f"    Center: ({cx:.1f}, {cy:.1f})")
    print(f"    Semi-axes: a={a_orig:.1f}, b={b_orig:.1f} → radius≈{radius_px:.1f} px")
    print(f"    Theta: {math.degrees(theta):.1f}°")
    print(f"    Mask pixels: {np.count_nonzero(iris_mask)}")

    return {
        'iris_mask': iris_mask,
        'cx': float(cx),
        'cy': float(cy),
        'radius_px': float(radius_px),
        'iris_params_orig': np.array([cx, cy, a_orig, b_orig, theta], dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────
# Core stabilization
# ─────────────────────────────────────────────────────────
def build_forward_gaze_matrix(dx, dy, torsion_deg, scale, cx, cy):
    """
    Build the same affine matrix that simulate_movement.py applies:
        M = getRotationMatrix2D((cx, cy), torsion_deg, scale)
        M[0,2] += dx
        M[1,2] += dy
    """
    M = cv2.getRotationMatrix2D(
        (float(cx), float(cy)), float(torsion_deg), float(scale)
    )
    M[0, 2] += float(dx)
    M[1, 2] += float(dy)
    return M


def stabilize_video(
    video_path: str,
    output_path: str,
    gaze_traj: dict = None,
    head_dx: np.ndarray = None,
    head_dy: np.ndarray = None,
    iris_mask: np.ndarray = None,
    cx: float = 0.0,
    cy: float = 0.0,
):
    """
    Read the warped video frame-by-frame and undo gaze + head motion.

    For each frame i:
      1. Undo head motion: warpAffine with (-head_dx[i], -head_dy[i])
      2. Undo gaze affine:
         a. Build forward M (same as in simulate_movement.py)
         b. Compute M_inv = cv2.invertAffineTransform(M)
         c. Warp the frame with M_inv
         d. Alpha-composite: original mask region selects corrected iris,
            outside mask keeps sclera/background untouched.
    """
    vinfo = get_video_info(video_path)
    w, h = vinfo["width"], vinfo["height"]
    fps = vinfo["fps"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Input : {video_path}")
    print(f"  Frames: {n_frames}, Resolution: {w}x{h}, FPS: {fps:.1f}")

    has_head = head_dx is not None and head_dy is not None
    has_gaze = gaze_traj is not None and iris_mask is not None

    if has_head:
        assert len(head_dx) >= n_frames, \
            f"head_motion has {len(head_dx)} entries but video has {n_frames} frames"
    if has_gaze:
        assert len(gaze_traj['dx']) >= n_frames, \
            f"gaze_trajectory has {len(gaze_traj['dx'])} entries but video has {n_frames} frames"

    # Prepare soft iris mask
    if has_gaze:
        mask_binary = (iris_mask > 0).astype(np.uint8) * 255
        mask_float = cv2.GaussianBlur(
            mask_binary.astype(np.float32) / 255.0, (9, 9), 2.0
        )

    # ── FFmpeg writer (match input codec) ──
    codec_name = vinfo.get("codec_name", "prores")
    pix_fmt = vinfo.get("pix_fmt", "yuv422p10le")
    profile = vinfo.get("profile", "")

    encoder_map = {
        "prores": "prores_ks", "h264": "libx264",
        "hevc": "libx265", "h265": "libx265",
    }
    encoder = encoder_map.get(codec_name, codec_name)

    if encoder == "prores_ks":
        profile_map = {
            "Proxy": "0", "LT": "1", "Standard": "2",
            "HQ": "3", "4444": "4", "4444 XQ": "5",
        }
        prores_profile = "3"
        for name, num in profile_map.items():
            if name in profile:
                prores_profile = num
                break
        encode_args = [
            "-c:v", "prores_ks", "-profile:v", prores_profile,
            "-vendor", "apl0", "-pix_fmt", pix_fmt,
        ]
    else:
        encode_args = ["-c:v", encoder, "-pix_fmt", pix_fmt]

    popen_kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(int(fps)), "-i", "-",
        *encode_args, output_path,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, **popen_kwargs)

    # ── Frame loop ──
    for i in tqdm(range(n_frames), desc="  Stabilizing"):
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Could not read frame {i}, stopping.")
            break

        # ─── STEP 1: Undo head motion (global inverse translate) ───
        if has_head:
            inv_head = np.float64([
                [1, 0, -float(head_dx[i])],
                [0, 1, -float(head_dy[i])],
            ])
            frame_f = frame.astype(np.float32)
            frame_f = cv2.warpAffine(
                frame_f, inv_head, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT101,
            )
            frame = np.clip(frame_f, 0, 255).astype(np.uint8)

        # ─── STEP 2: Undo gaze affine (iris-only compositing) ───
        if has_gaze:
            # Build forward M (same as simulate_movement.py)
            M_fwd = build_forward_gaze_matrix(
                gaze_traj['dx'][i], gaze_traj['dy'][i],
                gaze_traj['torsion_deg'][i], gaze_traj['scale'][i],
                cx, cy,
            )
            # Inverse affine
            M_inv = cv2.invertAffineTransform(M_fwd)

            # Warp frame back (moves displaced iris to original position)
            corrected = cv2.warpAffine(
                frame, M_inv, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT101,
            )

            # Composite: mask inside → corrected iris, outside → original frame
            alpha = mask_float[:, :, np.newaxis]
            blended = (
                frame.astype(np.float32) * (1.0 - alpha)
                + corrected.astype(np.float32) * alpha
            )
            frame = np.clip(blended, 0, 255).astype(np.uint8)

        # Write
        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
            ffmpeg_proc.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("FFmpeg terminated unexpectedly.")

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with code {ffmpeg_proc.returncode}")

    print(f"  Output: {output_path}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Stabilize iris by reversing synthetic gaze + head motion.\n"
                    "Iris mask and center are auto-detected via CondSeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required ──
    parser.add_argument("--video", type=str, required=True,
                        help="Input video with gaze/head motion")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <input>_stabilized.<ext>)")

    # ── Motion data (auto-detected from video dir if not given) ──
    parser.add_argument("--gaze_trajectory", type=str, default=None,
                        help="Path to gaze_trajectory.json")
    parser.add_argument("--head_motion", type=str, default=None,
                        help="Path to head_motion.npy")

    # ── Manual iris override (skip CondSeg) ──
    parser.add_argument("--iris_mask", type=str, default=None,
                        help="Manual iris mask PNG (skip CondSeg detection)")
    parser.add_argument("--pupil_mask", type=str, default=None,
                        help="Manual pupil mask PNG (combined with iris)")
    parser.add_argument("--cx", type=float, default=None,
                        help="Manual iris center X (skip CondSeg)")
    parser.add_argument("--cy", type=float, default=None,
                        help="Manual iris center Y (skip CondSeg)")

    # ── CondSeg auto-detection params ──
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to CondSeg checkpoint (default: models/best_2.pt)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu (auto-detect if omitted)")
    parser.add_argument("--detect_frame", type=int, default=0,
                        help="Frame index for CondSeg detection (default: 0 = first)")

    # ── Auto-fill from results JSON ──
    parser.add_argument("--results_json", type=str, default=None,
                        help="Path to auto_simulate_results.json (auto-fills cx, cy)")

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    print("=" * 60)
    print("Iris Stabilization")
    print("=" * 60)

    # ────────────────────────────────────────────────────────
    # Auto-detect gaze_trajectory.json and head_motion.npy
    # from the same directory as the input video
    # ────────────────────────────────────────────────────────
    video_dir = os.path.dirname(os.path.abspath(args.video))

    if args.gaze_trajectory is None:
        candidate = os.path.join(video_dir, "gaze_trajectory.json")
        if os.path.exists(candidate):
            args.gaze_trajectory = candidate
            print(f"  Auto-detected: {candidate}")

    if args.head_motion is None:
        candidate = os.path.join(video_dir, "head_motion.npy")
        if os.path.exists(candidate):
            args.head_motion = candidate
            print(f"  Auto-detected: {candidate}")

    # ────────────────────────────────────────────────────────
    # Auto-fill cx/cy from results_json if provided
    # ────────────────────────────────────────────────────────
    if args.results_json:
        with open(args.results_json, 'r') as f:
            res = json.load(f)
        sim = res.get("simulations", {}).get("left_eye", {})
        if args.cx is None and 'cx' in sim:
            args.cx = float(sim['cx'])
        if args.cy is None and 'cy' in sim:
            args.cy = float(sim['cy'])
        print(f"  From results_json: cx={args.cx}, cy={args.cy}")

    # ────────────────────────────────────────────────────────
    # Validate: at least one motion file must exist
    # ────────────────────────────────────────────────────────
    if args.gaze_trajectory is None and args.head_motion is None:
        print("Error: No gaze_trajectory.json or head_motion.npy found.")
        print("  Provide them explicitly or place them next to the video.")
        return 1

    # ────────────────────────────────────────────────────────
    # Get iris mask + center: CondSeg auto-detect or manual
    # ────────────────────────────────────────────────────────
    iris_mask_arr = None
    need_iris = args.gaze_trajectory is not None

    if need_iris:
        if args.iris_mask and args.cx is not None and args.cy is not None:
            # ── Manual mode ──
            print(f"\n  Using manual iris mask: {args.iris_mask}")
            iris_mask_arr = cv2.imread(args.iris_mask, cv2.IMREAD_GRAYSCALE)
            if iris_mask_arr is None:
                print(f"Error: Could not read iris mask: {args.iris_mask}")
                return 1
            if args.pupil_mask:
                pupil = cv2.imread(args.pupil_mask, cv2.IMREAD_GRAYSCALE)
                if pupil is not None:
                    iris_mask_arr = np.maximum(iris_mask_arr, pupil)
        else:
            # ── CondSeg auto-detect ──
            detection = detect_iris_condseg(
                video_path=args.video,
                checkpoint=args.checkpoint,
                device=args.device,
                frame_index=args.detect_frame,
            )

            if detection is None:
                print("Error: CondSeg could not detect iris. Try --iris_mask + --cx + --cy.")
                return 1

            iris_mask_arr = detection['iris_mask']
            args.cx = detection['cx']
            args.cy = detection['cy']
            print(f"  Using CondSeg-detected iris: center=({args.cx:.1f}, {args.cy:.1f}), "
                  f"radius≈{detection['radius_px']:.1f}px")

    # ────────────────────────────────────────────────────────
    # Load motion data
    # ────────────────────────────────────────────────────────
    gaze_traj = None
    if args.gaze_trajectory:
        print(f"\n  Loading gaze: {args.gaze_trajectory}")
        gaze_traj = load_gaze_trajectory(args.gaze_trajectory)
        n = len(gaze_traj['dx'])
        print(f"    {n} frames | dx [{gaze_traj['dx'].min():.2f}, {gaze_traj['dx'].max():.2f}] | "
              f"torsion [{gaze_traj['torsion_deg'].min():.2f}, {gaze_traj['torsion_deg'].max():.2f}]°")
        # Real-life format embeds the reference center — use it if cx/cy not set yet
        if args.cx is None and 'reference_cx' in gaze_traj:
            args.cx = gaze_traj['reference_cx']
            args.cy = gaze_traj['reference_cy']
            print(f"    Reference center from trajectory: ({args.cx:.1f}, {args.cy:.1f})")

    head_dx_arr, head_dy_arr = None, None
    if args.head_motion:
        print(f"  Loading head: {args.head_motion}")
        head_dx_arr, head_dy_arr = load_head_motion(args.head_motion)
        print(f"    {len(head_dx_arr)} frames | "
              f"dx [{head_dx_arr.min():.2f}, {head_dx_arr.max():.2f}] | "
              f"dy [{head_dy_arr.min():.2f}, {head_dy_arr.max():.2f}]")

    # ────────────────────────────────────────────────────────
    # Output path
    # ────────────────────────────────────────────────────────
    if args.output is None:
        base, ext = os.path.splitext(args.video)
        args.output = f"{base}_stabilized{ext}"

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────
    # Run stabilization
    # ────────────────────────────────────────────────────────
    print(f"\n  Undo head motion : {'YES' if head_dx_arr is not None else 'NO'}")
    print(f"  Undo gaze affine : {'YES' if gaze_traj is not None else 'NO'}")
    if args.cx is not None:
        print(f"  Iris center      : ({args.cx:.1f}, {args.cy:.1f})")

    stabilize_video(
        video_path=args.video,
        output_path=args.output,
        gaze_traj=gaze_traj,
        head_dx=head_dx_arr,
        head_dy=head_dy_arr,
        iris_mask=iris_mask_arr,
        cx=args.cx or 0.0,
        cy=args.cy or 0.0,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
