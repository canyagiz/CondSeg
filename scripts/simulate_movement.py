import os, math, subprocess, json as json_lib, sys
import numpy as np
import cv2
from tqdm import tqdm


# -----------------------------
# Head Motion Noise Generator
# -----------------------------
def generate_head_motion_trajectory(
    n_frames, fps,
    drift_amp=1.5,      # Yavaş sallanma amplitüdü (px)
    drift_freq=0.3,     # Drift frekansı (Hz)
    tremor_amp=0.3,     # Mikro titreme amplitüdü (px)
    tremor_freq=10.0,   # Tremor frekansı (Hz)
    jitter_amp=0.5,     # Düzensiz hareket amplitüdü (px)
    seed=42             # Tekrarlanabilirlik için seed
):
    """
    Fizyolojik kafa hareketi trajektörisi oluşturur.
    
    Bileşenler:
    - Drift: Yavaş kafa sallanması (0.1-0.5 Hz)
    - Tremor: Kas titremesi (8-12 Hz)
    - Jitter: Düzensiz mikro hareketler (deterministik pseudo-random)
    
    Args:
        n_frames: Toplam frame sayısı
        fps: Kare hızı
        drift_amp: Drift amplitüdü (piksel)
        drift_freq: Drift frekansı (Hz)
        tremor_amp: Tremor amplitüdü (piksel)
        tremor_freq: Tremor frekansı (Hz)
        jitter_amp: Jitter amplitüdü (piksel)
        seed: Deterministik sonuç için seed değeri
    
    Returns:
        (dx_array, dy_array): Her frame için x ve y deplasmanları (piksel)
    """
    rng = np.random.default_rng(seed)
    
    # Zaman dizisi oluştur
    t = np.arange(n_frames) / fps
    
    # Rastgele faz offsetleri (deterministik - seed'e bağlı)
    phase_drift_x = rng.random() * 2 * np.pi
    phase_drift_y = rng.random() * 2 * np.pi
    phase_tremor_x = rng.random() * 2 * np.pi
    phase_tremor_y = rng.random() * 2 * np.pi
    
    # Jitter için rastgele frekanslar (deterministik)
    jitter_freq_x1 = 1.5 + rng.random() * 0.5  # 1.5-2.0 Hz
    jitter_freq_x2 = 2.8 + rng.random() * 0.5  # 2.8-3.3 Hz
    jitter_freq_y1 = 1.8 + rng.random() * 0.5  # 1.8-2.3 Hz
    jitter_freq_y2 = 2.5 + rng.random() * 0.5  # 2.5-3.0 Hz
    
    # 1. Drift bileşeni (yavaş sallanma)
    drift_x = drift_amp * np.sin(2 * np.pi * drift_freq * t + phase_drift_x)
    drift_y = drift_amp * 0.8 * np.sin(2 * np.pi * drift_freq * 1.1 * t + phase_drift_y)
    
    # 2. Tremor bileşeni (kas titremesi 8-12 Hz)
    tremor_x = tremor_amp * np.sin(2 * np.pi * tremor_freq * t + phase_tremor_x)
    tremor_y = tremor_amp * 0.9 * np.sin(2 * np.pi * (tremor_freq * 0.95) * t + phase_tremor_y)
    
    # 3. Jitter bileşeni (düzensiz mikro hareketler - beat frequency pattern)
    jitter_x = jitter_amp * np.sin(2 * np.pi * jitter_freq_x1 * t) * np.sin(2 * np.pi * jitter_freq_x2 * t)
    jitter_y = jitter_amp * 0.85 * np.sin(2 * np.pi * jitter_freq_y1 * t) * np.sin(2 * np.pi * jitter_freq_y2 * t)
    
    # Toplam deplasman
    dx = drift_x + tremor_x + jitter_x
    dy = drift_y + tremor_y + jitter_y
    
    return dx.astype(np.float32), dy.astype(np.float32)


def get_video_codec_info(video_path):
    """Use ffprobe to get codec info from a reference video."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed on {video_path}: {result.stderr}")
    info = json_lib.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError(f"No video stream found in {video_path}")
    stream = info["streams"][0]
    return {
        "codec_name": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "pix_fmt": stream.get("pix_fmt"),
        "bit_depth": stream.get("bits_per_raw_sample"),
    }

# -----------------------------
# 1) Radial warp at subpixel
# -----------------------------
# -----------------------------
# 1) Radial warp helpers
# -----------------------------
def precompute_displacement_vectors(h, w, cx, cy, ring_r_px, band_sigma_px=8.0):
    """
    Precompute the unit displacement vectors scaled by the Gaussian profile.
    Returns (Vx, Vy, xx, yy) such that:
       displacement_x = delta_px * Vx
       displacement_y = delta_px * Vy
    
    This avoids recomputing meshgrids, sqrts, and exp() for every frame.
    """
    # Create meshgrid once
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    
    # Distance from center
    # Add epsilon to avoid division by zero at strict center
    rr = np.sqrt(dx*dx + dy*dy) + 1e-6
    
    # Normalized direction vectors (ux, uy)
    ux = dx / rr
    uy = dy / rr
    
    # Gaussian profile p(r) = exp(...)
    # ur = delta_px * p(r)
    # displacement_x = ur * ux = delta_px * p(r) * ux
    # So Vx = p(r) * ux
    
    profile = np.exp(-0.5 * ((rr - ring_r_px) / band_sigma_px) ** 2)
    
    Vx = profile * ux
    Vy = profile * uy
    
    return Vx, Vy, xx, yy

def apply_warp(img, xx, yy, Vx, Vy, delta_px):
    """
    Apply warp using precomputed vectors.
    map_x = xx - delta_px * Vx
    map_y = yy - delta_px * Vy
    
    Works with float32 images to preserve subpixel precision.
    """
    map_x = (xx - delta_px * Vx).astype(np.float32)
    map_y = (yy - delta_px * Vy).astype(np.float32)
    
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)

def warp_radial(img_bgr, cx, cy, ring_r_px, delta_val, use_um=False, iris_diam_mm=11.7, band_sigma_px=8.0):
    """
    One-off wrapper for backward compatibility or single-frame use.
    If use_um is True, delta_val is treated as microns.
    If use_um is False, delta_val is treated as pixels.
    """
    h, w = img_bgr.shape[:2]
    
    delta_px = float(delta_val)
    if use_um:
        # um -> px
        mm_per_px = iris_diam_mm / (2.0 * ring_r_px)
        px_per_um = 1.0 / (mm_per_px * 1000.0)
        delta_px  = float(delta_val) * px_per_um
    
    
    Vx, Vy, xx, yy = precompute_displacement_vectors(h, w, cx, cy, ring_r_px, band_sigma_px)
    return apply_warp(img_bgr, xx, yy, Vx, Vy, delta_px), delta_px

# -----------------------------
# 2) Main: save 4 outputs
# -----------------------------
def run_simulation(
    img_path,
    L, R, U, D,
    delta_um=10.0,
    delta_px=None,
    use_um=False,
    iris_diam_mm=11.7,
    fps=60,
    dur_s=10.0,
    f_hz=1.5,
    out_dir="opa_sim_outputs",
    generate_zoom_video=False,
    zoom_size=200,
    ref_video=None,
    # Head motion parameters
    head_motion=False,
    head_motion_seed=42,
    drift_amp=1.5,
    drift_freq=0.3,
    tremor_amp=0.3,
    tremor_freq=10.0,
    jitter_amp=0.5
):
    # Create subdirectory named after input image
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(out_dir, img_basename)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be read")
    
    # Convert to float32 for high-precision warp processing
    # This preserves subpixel information during interpolation
    # Original 8-bit (0-255) -> float32 (0.0-255.0)
    img_float = img.astype(np.float32)

    # compute iris center and radius from LRUD limits
    cx = (L + R) / 2.0
    cy = (U + D) / 2.0
    r_px = 0.25 * ((R - L) + (D - U))

    # (1) analysis overlay: original vs expanded circle
    # Using the single-shot function for the static outputs
    # Determine basic peak delta_px for static images
    static_delta_val = delta_um if use_um else (delta_px if delta_px is not None else 5.0)
    warped_static, delta_px_peak = warp_radial(img, cx, cy, r_px, static_delta_val, use_um=use_um, iris_diam_mm=iris_diam_mm)
    
    overlay = img.copy()
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(r_px)), (0,0,255), 1)                 # red original
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(r_px + delta_px_peak)), (0,255,0), 1)     # green expanded
    overlay_path = os.path.join(out_dir, "01_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    # (2) warped (static, lossless png)
    warped_path = os.path.join(out_dir, "02_warped_delta.png")
    cv2.imwrite(warped_path, warped_static)

    # (3) full-resolution video (default output)
    n_frames = int(round(dur_s * fps))
    h, w = img.shape[:2]
    
    print(f"    Precomputing displacement maps for {w}x{h} frames... (Optimization)")
    # PRECOMPUTE VECTORS ONCE
    Vx, Vy, xx, yy = precompute_displacement_vectors(h, w, cx, cy, r_px)
    
    # Calculate px_per_um for the loop
    mm_per_px = iris_diam_mm / (2.0 * r_px)
    px_per_um = 1.0 / (mm_per_px * 1000.0)

    # Determine encoding settings
    full_vid_path = os.path.join(out_dir, "03_full.mov")

    if ref_video:
        # Copy codec settings from reference video
        try:
            codec_info = get_video_codec_info(ref_video)
            codec_name = codec_info["codec_name"]
            pix_fmt = codec_info["pix_fmt"] or "yuv422p10le"
            
            # Map codec names to ffmpeg encoder names
            encoder_map = {
                "prores": "prores_ks",
                "h264": "libx264",
                "hevc": "libx265",
                "h265": "libx265",
            }
            encoder = encoder_map.get(codec_name, codec_name)
            
            # Build encoder-specific options
            if encoder == "prores_ks":
                profile = codec_info.get("profile", "")
                # Map ProRes profile names to ffmpeg profile numbers
                profile_map = {
                    "Proxy": "0", "LT": "1", "Standard": "2",
                    "HQ": "3", "4444": "4", "4444 XQ": "5"
                }
                prores_profile = "3"  # Default to HQ
                for name, num in profile_map.items():
                    if name in profile:
                        prores_profile = num
                        break
                # Add vendor tag for Apple compatibility (important on Windows)
                encode_args = ["-c:v", "prores_ks", "-profile:v", prores_profile,
                            "-vendor", "apl0", "-pix_fmt", pix_fmt]
            else:
                encode_args = ["-c:v", encoder, "-pix_fmt", pix_fmt]
        except Exception as e:
            print(f"    Warning: Could not extract codec info from {ref_video}, using default. Error: {e}")
            encode_args = ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-pix_fmt", "yuv422p10le"]
    else:
        # Default: ProRes 422 HQ 10-bit with Apple vendor tag
        encode_args = ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-pix_fmt", "yuv422p10le"]

    # Platform-specific subprocess settings
    # Use DEVNULL for stderr to prevent pipe buffer deadlock on Windows
    # (stderr pipe fills up and blocks stdin writes)
    popen_kwargs = {"stdin": subprocess.PIPE, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        # Prevent console window popup on Windows
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    # Start ffmpeg process for main video
    # Use 16-bit input (bgr48le) to preserve 10-bit precision
    # bgr48le = 16-bit per channel, little-endian
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr48le",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        *encode_args, full_vid_path
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, **popen_kwargs)

    # Optional zoom video (uses simple h264 for smaller file)
    zoom_vid_path = None
    ffmpeg_zoom_proc = None
    zx, zy = 0, 0
    if generate_zoom_video:
        half_zoom = zoom_size // 2
        zx = int(round(L)) - half_zoom
        zy = int(round(cy)) - half_zoom
        zx = max(0, min(w-zoom_size, zx))
        zy = max(0, min(h-zoom_size, zy))
        zoom_vid_path = os.path.join(out_dir, f"04_zoom_{zoom_size}x{zoom_size}.mp4")
        zoom_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{zoom_size}x{zoom_size}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", zoom_vid_path
        ]
        ffmpeg_zoom_proc = subprocess.Popen(zoom_cmd, **popen_kwargs)

    # Generate head motion trajectory if enabled
    head_dx, head_dy = None, None
    if head_motion:
        print(f"    Generating head motion trajectory (seed={head_motion_seed})...")
        head_dx, head_dy = generate_head_motion_trajectory(
            n_frames, fps,
            drift_amp=drift_amp,
            drift_freq=drift_freq,
            tremor_amp=tremor_amp,
            tremor_freq=tremor_freq,
            jitter_amp=jitter_amp,
            seed=head_motion_seed
        )

    # Write frames to ffmpeg - flush after each frame for Windows pipe reliability
    # Use precomputed maps!
    # Determine simulation amplitude value
    sim_amp_val = delta_um if use_um else (delta_px if delta_px is not None else 5.0)

    for i in tqdm(range(n_frames), desc=f"    Rendering {n_frames} frames"):
        t = i / fps
        d_val = sim_amp_val * math.sin(2*math.pi*f_hz*t)
        
        # Calculate current delta_px
        d_px = float(d_val)
        if use_um:
            d_px = float(d_val) * px_per_um
        
        # Apply precomputed warp (uses float32 image for precision)
        frame_float = apply_warp(img_float, xx, yy, Vx, Vy, d_px)
        
        # Apply head motion if enabled
        if head_motion:
            M = np.float32([[1, 0, head_dx[i]], [0, 1, head_dy[i]]])
            frame_float = cv2.warpAffine(frame_float, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
        
        # Convert float32 (0-255) to uint16 (0-65535) for 16-bit output
        # This preserves the subpixel precision in the 10-bit output
        # Scale: 0-255 -> 0-65535 (multiply by 257 = 65535/255)
        frame_16bit = np.clip(frame_float * 257.0, 0, 65535).astype(np.uint16)
        frame_bytes = frame_16bit.tobytes()
        try:
            ffmpeg_proc.stdin.write(frame_bytes)
            ffmpeg_proc.stdin.flush()  # Flush after each frame for Windows pipe reliability
        except BrokenPipeError:
            raise RuntimeError(f"FFmpeg process terminated unexpectedly. Check if ffmpeg is installed correctly.")
        if ffmpeg_zoom_proc is not None:
            # Zoom video uses 8-bit (h264) - convert back from float for this
            frame_8bit = np.clip(frame_float, 0, 255).astype(np.uint8)
            zoom_frame = frame_8bit[zy:zy+zoom_size, zx:zx+zoom_size]
            try:
                ffmpeg_zoom_proc.stdin.write(zoom_frame.tobytes())
                ffmpeg_zoom_proc.stdin.flush()
            except BrokenPipeError:
                raise RuntimeError(f"FFmpeg zoom process terminated unexpectedly.")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with code {ffmpeg_proc.returncode}")

    if ffmpeg_zoom_proc is not None:
        ffmpeg_zoom_proc.stdin.close()
        ffmpeg_zoom_proc.wait()
        if ffmpeg_zoom_proc.returncode != 0:
            raise RuntimeError(f"FFmpeg zoom failed with code {ffmpeg_zoom_proc.returncode}")

    results = {
        "cx": cx, "cy": cy, "r_px": r_px,
        "delta_val": float(sim_amp_val),
        "use_um": use_um,
        "delta_px_at_peak": float(delta_px_peak),
        "overlay_png": overlay_path,
        "warped_png": warped_path,
        "full_video": full_vid_path
    }
    if zoom_vid_path is not None:
        results["zoom_video"] = zoom_vid_path
        results["zoom_roi_xy"] = (zx, zy)
    return results

# Example:
# results = run_simulation("eye.png", L=210, R=430, U=120, D=340, delta_um=10)
# print(results)
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Simulate micron-scale iris radial pulse on a single eye image.")
    parser.add_argument("--ref_video", type=str, required=True, help="Reference video to copy codec settings from.")
    parser.add_argument("--img", required=True, help="Path to input image (eye frame).")
    parser.add_argument("--top", type=int, required=True, help="Iris top limit (px).")
    parser.add_argument("--bot", type=int, required=True, help="Iris bottom limit (px).")
    parser.add_argument("--left", type=int, required=True, help="Iris left limit (px).")
    parser.add_argument("--right", type=int, required=True, help="Iris right limit (px).")

    parser.add_argument("--delta_um", type=float, default=10.0, help="Peak radial displacement in microns (used if --use-um set).")
    parser.add_argument("--delta_px", type=float, default=None, help="Peak radial displacement in pixels (default if --use-um NOT set).")
    parser.add_argument("--use_um", action="store_true", help="Use microns for displacement instead of pixels.")
    parser.add_argument("--iris_diam_mm", type=float, default=11.7, help="Assumed iris diameter in mm for um->px scaling.")
    parser.add_argument("--fps", type=int, default=60, help="Output video FPS.")
    parser.add_argument("--dur_s", type=float, default=10.0, help="Output video duration (seconds).")
    parser.add_argument("--f_hz", type=float, default=1.5, help="Sinusoidal pulse frequency (Hz).")
    parser.add_argument("--out_dir", default="opa_sim_outputs", help="Output directory.")
    parser.add_argument("--zoom_video", action="store_true", help="Also generate a small cropped zoom video (disabled by default).")
    parser.add_argument("--zoom_size", type=int, default=200, help="Zoom video size in pixels (default: 200x200).")

    # Head motion arguments
    parser.add_argument("--head_motion", action="store_true", help="Enable physiological head motion noise.")
    parser.add_argument("--head_motion_seed", type=int, default=42, help="Seed for deterministic head motion (default: 42).")
    parser.add_argument("--drift_amp", type=float, default=1.5, help="Head drift amplitude in pixels (default: 1.5).")
    parser.add_argument("--drift_freq", type=float, default=0.3, help="Head drift frequency in Hz (default: 0.3).")
    parser.add_argument("--tremor_amp", type=float, default=0.3, help="Micro-tremor amplitude in pixels (default: 0.3).")
    parser.add_argument("--tremor_freq", type=float, default=10.0, help="Micro-tremor frequency in Hz (default: 10.0).")
    parser.add_argument("--jitter_amp", type=float, default=0.5, help="Jitter amplitude in pixels (default: 0.5).")

    args = parser.parse_args()

    results = run_simulation(
        img_path=args.img,
        L=args.left, R=args.right, U=args.top, D=args.bot,
        delta_um=args.delta_um,
        delta_px=args.delta_px,
        use_um=args.use_um,
        iris_diam_mm=args.iris_diam_mm,
        fps=args.fps,
        dur_s=args.dur_s,
        f_hz=args.f_hz,
        out_dir=args.out_dir,
        generate_zoom_video=args.zoom_video,
        zoom_size=args.zoom_size,
        ref_video=args.ref_video,
        # Head motion
        head_motion=args.head_motion,
        head_motion_seed=args.head_motion_seed,
        drift_amp=args.drift_amp,
        drift_freq=args.drift_freq,
        tremor_amp=args.tremor_amp,
        tremor_freq=args.tremor_freq,
        jitter_amp=args.jitter_amp
    )

    print(json.dumps(results, indent=2))

