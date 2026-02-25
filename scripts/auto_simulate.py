#!/usr/bin/env python3
"""
Auto Simulate Movement Pipeline
===============================
Automatically extracts a frame from video, detects iris boundaries using SAM2,
and generates OPA simulation videos for both eyes.

This script uses existing TonoCAM modules instead of duplicating functions:
- SAM2Loader for segmentation
- EyeRegionSeparator for left/right eye separation
- ContourMeasurementPipeline for contour extraction

Robustness against eyelid occlusion:
- Uses lateral sectors (0° and 180°) as reference for radius
- These sectors are never occluded by eyelids
- Computes robust bounds even when eye is partially closed

Usage:
    python scripts/auto_simulate.py --video input.mov --delta_um 5.0
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

# Add TonoCAM to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules - no duplication!
from src.segmentation import SAM2Loader, EyeRegionSeparator
from scripts.simulate_movement import run_simulation

# Default model paths
DEFAULT_SAM_CHECKPOINT = "sam2-backup_SAM-FINETUNED-DILATION-3.pt"
DEFAULT_SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_YOLO_IRIS_PUPIL = "yolo-models/yolov8n-iris-pupil.pt"
DEFAULT_YOLO_EYE = "yolo-models/yolov8n-all-eye.pt"


def extract_frame(video_path: str, frame_index: int = None) -> tuple:
    """
    Extract a single frame from video using OpenCV.
    
    Args:
        video_path: Path to video file
        frame_index: Which frame to extract (None = middle frame)
        
    Returns:
        Tuple of (frame, frame_index, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_index is None:
        frame_index = total_frames // 2
    
    frame_index = max(0, min(frame_index, total_frames - 1))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    
    return frame, frame_index, fps, total_frames


def get_robust_iris_bounds(mask: np.ndarray, occlusion_threshold: float = 0.60) -> dict:
    """
    Extract iris boundary with robustness to eyelid occlusion.
    
    Uses lateral sectors (0° and 180°) as reference because they
    are never occluded by eyelids. This gives accurate radius
    even when the eye is partially closed.
    
    Also detects if eye is fully open by comparing lateral vs vertical sectors.
    
    Args:
        mask: Binary iris mask (H, W)
        occlusion_threshold: Ratio threshold for occlusion detection.
                            If vertical_radius / lateral_radius < threshold,
                            eye is considered partially closed (default: 0.90 = 10% tolerance)
        
    Returns:
        Dict with keys: L, R, U, D, center, radius_px, is_eye_fully_open
        Returns None if no valid contour found
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
    )
    
    if not contours:
        return None
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    if area < 100:
        return None
    
    # Compute center from moments (stable even with occlusion)
    M = cv2.moments(largest)
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        x, y, w, h = cv2.boundingRect(largest)
        cx, cy = x + w / 2, y + h / 2
    
    center = (cx, cy)
    
    # Convert contour to (N, 2) array
    contour = largest.reshape(-1, 2).astype(np.float32)
    
    # ================================================================
    # ROBUST RADIUS FROM LATERAL SECTORS
    # Lateral sectors (0° and 180°) are never occluded by eyelids
    # ================================================================
    
    # Compute polar coordinates relative to center
    dx = contour[:, 0] - cx
    dy = contour[:, 1] - cy
    angles = np.arctan2(dy, dx)  # Range: -pi to pi
    radii = np.sqrt(dx**2 + dy**2)
    
    # Define lateral sectors: ±45° around 0° and 180°
    lateral_mask = (
        (np.abs(angles) < np.pi/4) |  # 0° ± 45°
        (np.abs(np.abs(angles) - np.pi) < np.pi/4)  # 180° ± 45°
    )
    
    # Define vertical sectors: ±45° around 90° and 270° (top and bottom)
    vertical_mask = (
        (np.abs(angles - np.pi/2) < np.pi/4) |  # 90° ± 45° (bottom)
        (np.abs(angles + np.pi/2) < np.pi/4)    # -90° ± 45° (top)
    )
    
    lateral_radii = radii[lateral_mask]
    vertical_radii = radii[vertical_mask]
    
    if len(lateral_radii) < 5:
        robust_radius = np.median(radii)
        lateral_median = robust_radius
    else:
        lateral_median = np.median(lateral_radii)
        robust_radius = lateral_median
    
    # ================================================================
    # EYELID OCCLUSION DETECTION
    # Compare vertical sectors with lateral sectors
    # If vertical is significantly smaller, eye is partially closed
    # ================================================================
    
    is_eye_fully_open = True
    occlusion_ratio = 1.0
    
    if len(vertical_radii) >= 3 and len(lateral_radii) >= 3:
        vertical_median = np.median(vertical_radii)
        occlusion_ratio = vertical_median / (lateral_median + 1e-6)
        
        # Eye is NOT fully open if vertical sectors are smaller than threshold
        is_eye_fully_open = occlusion_ratio >= occlusion_threshold
    
    # Compute bounds using robust radius (circular assumption)
    bounds = {
        'L': int(cx - robust_radius),
        'R': int(cx + robust_radius),
        'U': int(cy - robust_radius),
        'D': int(cy + robust_radius),
        'center': center,
        'radius_px': robust_radius,
        'area': area,
        'n_lateral_points': len(lateral_radii),
        'n_vertical_points': len(vertical_radii),
        'n_total_points': len(contour),
        'is_eye_fully_open': is_eye_fully_open,
        'occlusion_ratio': float(occlusion_ratio)
    }
    
    return bounds


def segment_irises(
    frame: np.ndarray,
    sam_checkpoint: str,
    sam_config: str,
    yolo_iris_pupil_path: str,
    yolo_eye_path: str = None,
    device: str = None,
    center_crop_mode: bool = False,
    close_setup_mode: bool = False,
    with_resize: bool = False,
    manual_iris_bbox: str = None,
    manual_pupil_bbox: str = None
) -> dict:
    """
    Segment irises in frame using SAM2Loader from existing codebase.
    
    Args:
        frame: Input frame (BGR)
        sam_checkpoint: Path to SAM2 checkpoint
        sam_config: Path to SAM2 config
        yolo_iris_pupil_path: Path to YOLO iris/pupil model
        yolo_eye_path: Path to YOLO eye detection model (optional)
        device: 'cuda' or 'cpu'
        center_crop_mode: If True, crop from frame center instead of YOLO eye detection
        
    Returns:
        Dict with 'left' and 'right' keys, each containing iris bounds or None
    """
    import torch
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"  Loading SAM2 model on {device}...")
    
    # Use existing SAM2Loader from src.segmentation
    model = SAM2Loader(
        sam_checkpoint=sam_checkpoint,
        sam_config=sam_config,
        yolo_iris_pupil_path=yolo_iris_pupil_path,
        yolo_eye_path=yolo_eye_path,
        class_mapping={0: 'limbus', 1: 'iris', 2: 'pupil', 3: 'sclera'},
        device=device,
        yolo_conf_threshold=0.5,
        sam_score_threshold=0.8,
        use_bf16=True,
        center_crop_mode=center_crop_mode,
        close_setup_mode=close_setup_mode,
        with_resize=with_resize,
        manual_iris_bbox=[float(x) for x in manual_iris_bbox.split(',')] if manual_iris_bbox else None,
        manual_pupil_bbox=[float(x) for x in manual_pupil_bbox.split(',')] if manual_pupil_bbox else None,
    )
    
    print("  Running segmentation...")
    
    # Run segmentation
    _, instances = model.predict(frame)
    
    if not instances:
        print("  Warning: No instances detected")
        return {'left': None, 'right': None}
    
    # Separate eyes using EyeRegionSeparator
    h, w = frame.shape[:2]
    eye_separator = EyeRegionSeparator()
    
    result = {'left': None, 'right': None}
    
    # Check if instances have eye labels (SAM2 mode with crop)
    has_eye_labels = instances and 'eye_label' in instances[0]
    
    if has_eye_labels:
        for inst in instances:
            eye = inst.get('eye_label')
            if eye is None:
                eye_idx = inst.get('eye_index', 0)
                eye = 'left' if eye_idx == 0 else 'right'
            
            if inst.get('class_name') == 'iris' and eye in result:
                # Get robust bounds from mask
                bounds = get_robust_iris_bounds(inst['mask'])
                if bounds:
                    # Add crop offset if present
                    if 'crop_offset' in inst:
                        ox, oy = inst['crop_offset']
                        bounds['L'] += ox
                        bounds['R'] += ox
                        bounds['U'] += oy
                        bounds['D'] += oy
                        cx, cy = bounds['center']
                        bounds['center'] = (cx + ox, cy + oy)
                    result[eye] = bounds
    else:
        # Fallback: use eye separator
        left_instances, right_instances = eye_separator.separate_eyes(instances, (h, w))
        
        for eye, eye_instances in [('left', left_instances), ('right', right_instances)]:
            for inst in eye_instances:
                if inst.get('class_name') == 'iris':
                    bounds = get_robust_iris_bounds(inst['mask'])
                    if bounds:
                        result[eye] = bounds
                    break
    
    return result


def auto_simulate(
    video_path: str,
    frame_index: int = None,
    delta_um: float = 10.0,
    delta_px: float = None,
    use_um: bool = False,
    iris_diam_mm: float = 11.7,
    dur_s: float = 10.0,
    f_hz: float = 1.5,
    out_dir: str = "opa_sim_outputs",
    sam_checkpoint: str = DEFAULT_SAM_CHECKPOINT,
    sam_config: str = DEFAULT_SAM_CONFIG,
    yolo_iris_pupil_path: str = DEFAULT_YOLO_IRIS_PUPIL,
    yolo_eye_path: str = DEFAULT_YOLO_EYE,
    ref_video: str = None,
    device: str = None,
    eye: str = 'both',
    generate_zoom: bool = False,
    zoom_size: int = 200,
    center_crop_mode: bool = False,
    close_setup_mode: bool = False,
    with_resize: bool = False,
    # Head motion parameters
    head_motion: bool = False,
    head_motion_seed: int = 42,
    drift_amp: float = 1.5,
    drift_freq: float = 0.3,
    tremor_amp: float = 0.3,
    tremor_freq: float = 10.0,
    jitter_amp: float = 0.5,
    manual_iris_bbox: str = None,
    manual_pupil_bbox: str = None,
    output_fps: int = None
) -> dict:
    """
    Full pipeline: extract frame → SAM2 segmentation → robust bounds → simulate_movement.
    
    Args:
        video_path: Path to source video
        frame_index: Frame to extract (None = middle)
        delta_um: Peak radial displacement in microns
        iris_diam_mm: Assumed iris diameter for um->px scaling
        dur_s: Output video duration
        f_hz: Pulse frequency in Hz
        out_dir: Output directory
        sam_checkpoint: Path to SAM2 checkpoint
        sam_config: Path to SAM2 config
        yolo_iris_pupil_path: Path to YOLO iris/pupil model
        yolo_eye_path: Path to YOLO eye model
        device: 'cuda' or 'cpu'
        eye: 'left', 'right', or 'both'
        generate_zoom: Generate zoom videos
        zoom_size: Zoom video size
        
    Returns:
        Dict with simulation results for each eye
    """
    print("=" * 60)
    print("Auto Simulate Movement Pipeline")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print(f"  Delta: {delta_um} µm")
    print(f"  Frequency: {f_hz} Hz ({f_hz * 60:.0f} BPM)")
    
    # Step 1: Extract frame
    print(f"\n[1/3] Extracting frame from video...")
    frame, actual_frame_idx, fps, total_frames = extract_frame(video_path, frame_index)
    print(f"  Frame: {actual_frame_idx} / {total_frames}")
    print(f"  Size: {frame.shape[1]}x{frame.shape[0]}")
    print(f"  FPS: {fps}")
    
    # Step 2: Segment irises with SAM2
    print(f"\n[2/3] Segmenting irises with SAM2...")
    iris_bounds = segment_irises(
        frame,
        sam_checkpoint=sam_checkpoint,
        sam_config=sam_config,
        yolo_iris_pupil_path=yolo_iris_pupil_path,
        yolo_eye_path=yolo_eye_path,
        device=device,
        center_crop_mode=center_crop_mode,
        close_setup_mode=close_setup_mode,
        with_resize=with_resize,
        manual_iris_bbox=manual_iris_bbox,
        manual_pupil_bbox=manual_pupil_bbox
    )
    
    detected_eyes = [e for e in ['left', 'right'] if iris_bounds[e] is not None]
    print(f"\n  Detected eyes: {detected_eyes}")
    
    for eye_name in detected_eyes:
        bounds = iris_bounds[eye_name]
        status = "✓ FULLY OPEN" if bounds['is_eye_fully_open'] else "✗ PARTIALLY CLOSED"
        print(f"\n  {eye_name.capitalize()} iris (robust bounds):")
        print(f"    L={bounds['L']}, R={bounds['R']}, U={bounds['U']}, D={bounds['D']}")
        print(f"    Center: ({bounds['center'][0]:.1f}, {bounds['center'][1]:.1f})")
        print(f"    Radius: {bounds['radius_px']:.1f} px")
        print(f"    Occlusion ratio: {bounds['occlusion_ratio']:.2f} ({status})")
        print(f"    Lateral/Vertical points: {bounds['n_lateral_points']} / {bounds['n_vertical_points']}")
    
    # Step 3: Save frame and run simulations
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    frame_out_dir = os.path.join(out_dir, video_basename)
    os.makedirs(frame_out_dir, exist_ok=True)
    
    frame_path = os.path.join(frame_out_dir, f"frame_{actual_frame_idx}.png")
    cv2.imwrite(frame_path, frame)
    print(f"\n  Saved frame: {frame_path}")
    
    print(f"\n[3/3] Running simulations...")
    
    results = {
        'source_video': video_path,
        'frame_index': actual_frame_idx,
        'frame_path': frame_path,
        'fps': fps,
        'iris_bounds': {},
        'simulations': {}
    }
    
    # Convert bounds to JSON-serializable format
    for eye_name in detected_eyes:
        bounds = iris_bounds[eye_name]
        results['iris_bounds'][eye_name] = {
            'L': bounds['L'],
            'R': bounds['R'],
            'U': bounds['U'],
            'D': bounds['D'],
            'center': list(bounds['center']),
            'radius_px': float(bounds['radius_px']),
            'is_eye_fully_open': bool(bounds['is_eye_fully_open']),
            'occlusion_ratio': float(bounds['occlusion_ratio'])
        }
    
    eyes_to_process = []
    if eye == 'both':
        eyes_to_process = detected_eyes
    elif eye in detected_eyes:
        eyes_to_process = [eye]
    else:
        print(f"  Warning: Requested eye '{eye}' not detected")
    
    for eye_name in eyes_to_process:
        bounds = iris_bounds[eye_name]
        
        # Skip if eye is not fully open
        if not bounds['is_eye_fully_open']:
            print(f"\n  ⚠ Skipping {eye_name} eye: partially closed (occlusion ratio: {bounds['occlusion_ratio']:.2f})")
            results['simulations'][eye_name] = {
                'skipped': True,
                'reason': 'Eye partially closed',
                'occlusion_ratio': bounds['occlusion_ratio']
            }
            continue
        
        print(f"\n  Simulating {eye_name} eye...")
        
        try:
            # Use existing run_simulation from simulate_movement.py
            sim_result = run_simulation(
                img_path=frame_path,
                L=bounds['L'],
                R=bounds['R'],
                U=bounds['U'],
                D=bounds['D'],
                delta_um=delta_um,
                delta_px=delta_px,
                use_um=use_um,
                iris_diam_mm=iris_diam_mm,
                fps=int(output_fps) if output_fps is not None else int(fps),
                dur_s=dur_s,
                f_hz=f_hz,
                out_dir=os.path.join(frame_out_dir, eye_name),
                generate_zoom_video=generate_zoom,
                zoom_size=zoom_size,
                ref_video=ref_video,
                # Head motion
                head_motion=head_motion,
                head_motion_seed=head_motion_seed,
                drift_amp=drift_amp,
                drift_freq=drift_freq,
                tremor_amp=tremor_amp,
                tremor_freq=tremor_freq,
                jitter_amp=jitter_amp
            )
            
            results['simulations'][eye_name] = {
                'full_video': sim_result['full_video'],
                'overlay_png': sim_result['overlay_png'],
                'delta_px_at_peak': sim_result['delta_px_at_peak']
            }
            print(f"    ✓ Output: {sim_result['full_video']}")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            results['simulations'][eye_name] = {'error': str(e)}
    
    # Save results JSON
    results_path = os.path.join(frame_out_dir, "auto_simulate_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved: {results_path}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Auto-simulate OPA movement from video using SAM2 segmentation"
    )
    
    # Required
    parser.add_argument("--video", type=str, required=True,
                        help="Path to source video")
    
    # Frame selection
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame index to extract (default: middle frame)")
    
    # Simulation parameters
    parser.add_argument("--delta_um", type=float, default=10.0,
                        help="Peak radial displacement in microns (used if --use-um set)")
    parser.add_argument("--delta_px", type=float, default=None,
                        help="Peak radial displacement in pixels (default if --use-um NOT set)")
    parser.add_argument("--use_um", action="store_true",
                        help="Use microns for displacement instead of pixels")
    parser.add_argument("--iris_diam_mm", type=float, default=11.7,
                        help="Assumed iris diameter in mm (default: 11.7)")
    parser.add_argument("--dur_s", type=float, default=10.0,
                        help="Output video duration in seconds (default: 10.0)")
    parser.add_argument("--output-fps", type=int, default=None,
                        help="Output video FPS (default: same as input video)")
    parser.add_argument("--f_hz", type=float, default=1.5,
                        help="Pulse frequency in Hz (default: 1.5)")
    parser.add_argument("--bpm", type=float, default=None,
                        help="Heart rate in BPM (alternative to --f_hz)")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="opa_sim_outputs",
                        help="Output directory (default: opa_sim_outputs)")
    parser.add_argument("--eye", type=str, default="both",
                        choices=["left", "right", "both"],
                        help="Which eye(s) to process (default: both)")
    parser.add_argument("--zoom", action="store_true",
                        help="Generate zoom videos")
    parser.add_argument("--zoom_size", type=int, default=200,
                        help="Zoom video size (default: 200)")
    
    # Model paths
    parser.add_argument("--sam_checkpoint", type=str, default=DEFAULT_SAM_CHECKPOINT,
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--sam_config", type=str, default=DEFAULT_SAM_CONFIG,
                        help="Path to SAM2 config")
    parser.add_argument("--yolo_iris_pupil", type=str, default=DEFAULT_YOLO_IRIS_PUPIL,
                        help="Path to YOLO iris/pupil model")
    parser.add_argument("--yolo_eye", type=str, default=DEFAULT_YOLO_EYE,
                        help="Path to YOLO eye detection model")
    parser.add_argument("--ref-video", type=str, default=None,
                        help="Reference video to copy codec settings from (default: None = use ProRes 10-bit)")
    
    # Single-eye video mode
    parser.add_argument("--center-crop", action="store_true",
                        help="Skip eye detection, crop 1024x1024 from frame center (for single-eye videos)")
    
    # Close setup mode - full frame processing
    parser.add_argument("--close-setup", action="store_true",
                        help="Pass full 4K frame to iris/pupil model without any eye cropping")
    parser.add_argument("--with-resize", action="store_true",
                        help="When using --close-setup, resize frame to 1024x1024 before processing")

    # Manual bbox override
    parser.add_argument("--manual-iris", type=str, default=None,
                        help="Manual iris bbox 'x1,y1,x2,y2' (bypasses YOLO)")
    parser.add_argument("--manual-pupil", type=str, default=None,
                        help="Manual pupil bbox 'x1,y1,x2,y2' (bypasses YOLO)")


    # Head motion parameters
    parser.add_argument("--head_motion", action="store_true",
                        help="Enable physiological head motion noise")
    parser.add_argument("--head_motion_seed", type=int, default=42,
                        help="Seed for deterministic head motion (default: 42)")
    parser.add_argument("--drift_amp", type=float, default=1.5,
                        help="Head drift amplitude in pixels (default: 1.5)")
    parser.add_argument("--drift_freq", type=float, default=0.3,
                        help="Head drift frequency in Hz (default: 0.3)")
    parser.add_argument("--tremor_amp", type=float, default=0.3,
                        help="Micro-tremor amplitude in pixels (default: 0.3)")
    parser.add_argument("--tremor_freq", type=float, default=10.0,
                        help="Micro-tremor frequency in Hz (default: 10.0)")
    parser.add_argument("--jitter_amp", type=float, default=0.5,
                        help="Jitter amplitude in pixels (default: 0.5)")

    args = parser.parse_args()
    
    # Convert BPM to Hz if provided
    f_hz = args.f_hz
    if args.bpm is not None:
        f_hz = args.bpm / 60.0
        print(f"Using {args.bpm} BPM = {f_hz:.3f} Hz")
    
    # Validate paths
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        return 1
    
    if not os.path.exists(args.sam_checkpoint):
        print(f"Error: SAM checkpoint not found: {args.sam_checkpoint}")
        return 1
    
    if not os.path.exists(args.yolo_iris_pupil):
        print(f"Error: YOLO model not found: {args.yolo_iris_pupil}")
        return 1
    
    # Run pipeline
    results = auto_simulate(
        video_path=args.video,
        frame_index=args.frame,
        delta_um=args.delta_um,
        delta_px=args.delta_px,
        use_um=args.use_um,
        iris_diam_mm=args.iris_diam_mm,
        dur_s=args.dur_s,
        f_hz=f_hz,
        out_dir=args.out_dir,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        yolo_iris_pupil_path=args.yolo_iris_pupil,
        yolo_eye_path=args.yolo_eye,
        ref_video=args.ref_video,
        eye=args.eye,
        generate_zoom=args.zoom,
        zoom_size=args.zoom_size,
        center_crop_mode=args.center_crop,
        close_setup_mode=args.close_setup,
        with_resize=args.with_resize,
        # Head motion
        head_motion=args.head_motion,
        head_motion_seed=args.head_motion_seed,
        drift_amp=args.drift_amp,
        drift_freq=args.drift_freq,
        tremor_amp=args.tremor_amp,
        tremor_freq=args.tremor_freq,
        jitter_amp=args.jitter_amp,
        manual_iris_bbox=args.manual_iris,
        manual_pupil_bbox=args.manual_pupil,
        output_fps=args.output_fps
    )
    
    # Print summary
    print("\nSummary:")
    for eye_name, sim in results.get('simulations', {}).items():
        if 'error' in sim:
            print(f"  {eye_name}: FAILED - {sim['error']}")
        else:
            print(f"  {eye_name}: {sim.get('full_video', 'N/A')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
