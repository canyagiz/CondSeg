"""
Batch CondSeg Inference + FFT Pipeline
========================================
Self-contained script for processing ALL simulation benchmark videos.

Features:
  - Multi-GPU: uses 2× A100 GPUs via torch.multiprocessing
  - Caching: JSON-based cache for resumability (CSV updates every ~1 min)
  - FFT: computes FFT for all 5 ellipse signals (x0, y0, a, b, theta_rad)
  - Monitoring: TensorBoard dashboard for real-time progress tracking
  - FP16: uses torch.amp.autocast for safe mixed precision

Usage:
    python batch_inference.py
    python batch_inference.py --dry-run
    tensorboard --logdir runs/batch_inference
"""

import os
import sys
import csv
import json
import time
import math
import threading
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

# ═══════════════════════════════════════════════════════════════════════════════
# ██ CONFIG — Edit these paths/values before running on cloud ██
# ═══════════════════════════════════════════════════════════════════════════════

# -- Paths --
MANIFEST_CSV     = "./glaucot-data-storage/simulation-benchmark/manifest.csv"
CHECKPOINT       = "./CondSeg/models/CNN/best_3.pt"
BASE_VIDEO_DIR   = "./glaucot-data-storage/simulation-benchmark"  # prefix stripped from CSV paths

# -- Model --
IMG_SIZE         = 1024
USE_AMP          = False       # KEEP FALSE! FP16 quantizes iris params (~254px → 0.125 steps)

# -- Signal Processing --
SAMPLE_RATE      = 60          # 60 FPS simulation videos
BP_LOWCUT        = 0.5         # bandpass filter lower cutoff Hz
BP_HIGHCUT       = 4.0         # bandpass filter upper cutoff Hz
BP_ORDER         = 4           # Butterworth filter order

# -- Multi-GPU --
NUM_GPUS         = 2           # number of GPUs to use (set to 1 for single-GPU)

# -- Caching & CSV --
CACHE_FILE       = "./glaucot-data-storage/simulation-benchmark/processed_cache.json"
CSV_RELOAD_SEC   = 60          # re-read manifest CSV every N seconds

# -- Monitoring --
TENSORBOARD_LOG  = "./runs/batch_inference"

# -- Output --
SAVE_OVERLAY_VID = False       # save overlay .avi per video (slow, large files)
SIGNAL_NAMES     = ["x0", "y0", "a", "b", "theta_rad"]

# -- Misc --
SENTINEL         = "__DONE__"  # poison pill for workers

# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet normalization constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
# ██ CACHE MANAGEMENT ██
# =============================================================================

def load_cache(cache_path: str) -> dict:
    """Load processed sample cache from JSON file."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"  ⚠ Cache file corrupted, starting fresh: {cache_path}")
            return {}
    return {}


def save_cache(cache_path: str, cache: dict):
    """Atomically save cache to JSON (write to .tmp then rename)."""
    tmp_path = cache_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(cache, f, indent=2)
        # Atomic rename (works on Linux; on Windows, this is close enough)
        if os.path.exists(cache_path):
            os.replace(tmp_path, cache_path)
        else:
            os.rename(tmp_path, cache_path)
    except Exception as e:
        print(f"  ⚠ Failed to save cache: {e}")


# =============================================================================
# ██ CSV PARSING ██
# =============================================================================

def parse_manifest(csv_path: str) -> list:
    """
    Parse manifest CSV and return list of dicts with keys:
      sample_id, video_path, output_dir, video_basename, status
    Only includes rows with status == 'ok'.
    """
    samples = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("status", "").strip()
                if status != "ok":
                    continue

                sample_id = row["sample_id"].strip()
                full_video = row["full_video"].strip()

                # Normalize path:  ./glaucot-data-storage/... → glaucot-data-storage/...
                if full_video.startswith("./"):
                    full_video = full_video[2:]

                video_dir = os.path.dirname(full_video)           # .../000000/frame_60
                video_basename = os.path.splitext(os.path.basename(full_video))[0]  # 03_full
                output_dir = os.path.join(video_dir, "inferences")

                samples.append({
                    "sample_id":      sample_id,
                    "video_path":     full_video,
                    "output_dir":     output_dir,
                    "ffts_dir":       os.path.join(output_dir, "ffts"),
                    "video_basename": video_basename,
                    "f_hz":           float(row.get("f_hz", 0)),
                    "delta_px":       float(row.get("delta_px", 0)),
                    "head_tier":      row.get("head_tier", ""),
                    "gaze_tier":      row.get("gaze_tier", ""),
                })
    except FileNotFoundError:
        print(f"  ✗ Manifest CSV not found: {csv_path}")
    return samples


# =============================================================================
# ██ MODEL LOADING (from inference.py) ██
# =============================================================================

def load_model(checkpoint_path: str, img_size: int, device: torch.device):
    """Load CondSeg model from checkpoint."""
    # Import from local codebase (script runs from CondSeg/ dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from condseg import CondSeg

    model = CondSeg(img_size=img_size, epsilon=0.01, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    model = model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", "N/A")
    print(f"  [GPU-{device.index}] ✓ Model loaded — epoch {epoch}, val_loss {val_loss}")
    return model


# =============================================================================
# ██ VIDEO PREPROCESSING (from inference.py) ██
# =============================================================================

def letterbox_frame(frame_bgr: np.ndarray, img_size: int) -> np.ndarray:
    """Resize frame preserving aspect ratio, pad to square with edge replication."""
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


def preprocess_square_fast(square_bgr: np.ndarray) -> torch.Tensor:
    """
    Minimal CPU work: BGR→RGB + uint8→tensor.
    Normalization happens on GPU (see gpu_normalize).
    """
    rgb = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)
    # (H, W, 3) uint8 → (1, 3, H, W) float32 tensor — but DON'T normalize yet
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


# GPU-side ImageNet normalization constants (created once per worker)
_GPU_MEAN = None
_GPU_STD  = None

def gpu_normalize(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Normalize on GPU: much faster than numpy on CPU.
    tensor: (1, 3, H, W) float32 in [0, 255] range.
    """
    global _GPU_MEAN, _GPU_STD
    if _GPU_MEAN is None or _GPU_MEAN.device != device:
        _GPU_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _GPU_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tensor = tensor / 255.0
    tensor = (tensor - _GPU_MEAN) / _GPU_STD
    return tensor


# =============================================================================
# ██ THREADED FRAME PREFETCHER ██
# =============================================================================

class FramePrefetcher:
    """
    Reads and letterboxes frames in a background thread.
    Main thread can pull ready tensors while GPU processes the previous frame.
    """
    def __init__(self, video_path: str, img_size: int, queue_size: int = 8):
        self.cap = cv2.VideoCapture(video_path)
        self.img_size = img_size
        self.queue = __import__('queue').Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self._ok = self.cap.isOpened()

    def start(self):
        if self._ok:
            self.thread.start()
        return self

    def _worker(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put(None)  # signal end
                break
            square = letterbox_frame(frame, self.img_size)
            tensor = preprocess_square_fast(square)
            self.queue.put(tensor)
        self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    @property
    def is_open(self):
        return self._ok


# =============================================================================
# ██ INFERENCE ON SINGLE VIDEO ██
# =============================================================================

def run_inference_on_video(model, video_path: str, output_dir: str,
                           video_basename: str, device: torch.device,
                           img_size: int = IMG_SIZE,
                           use_amp: bool = USE_AMP) -> dict:
    """
    Run CondSeg inference on a single video file.
    Uses threaded prefetch + GPU-side normalization for speed.
    Saves 5 .npy signal files to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    prefetcher = FramePrefetcher(video_path, img_size).start()
    if not prefetcher.is_open:
        return {"error": f"Cannot open video: {video_path}"}

    frame_idx = 0
    all_params = []
    t_start = time.time()

    # Profiling accumulators
    prof = {"prefetch_wait": 0, "gpu_normalize": 0, "model": 0}
    prof_interval = 100

    with torch.no_grad():
        for tensor_cpu in prefetcher:
            t0 = time.time()

            # Transfer to GPU + normalize there
            tensor = tensor_cpu.to(device, non_blocking=True)
            t1 = time.time()
            tensor = gpu_normalize(tensor, device)
            t2 = time.time()

            # Forward pass
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(tensor)
            else:
                outputs = model(tensor)
            torch.cuda.synchronize(device)
            t3 = time.time()

            iris_params = outputs["iris_params"][0].cpu().float().numpy()
            all_params.append(iris_params.copy())
            frame_idx += 1

            # Accumulate profiling
            prof["prefetch_wait"] += (t1 - t0)
            prof["gpu_normalize"] += (t2 - t1)
            prof["model"]         += (t3 - t2)

            # Print breakdown every N frames
            if frame_idx == prof_interval:
                total_prof = sum(prof.values())
                print(f"  [GPU-{device.index}] ── Timing ({prof_interval}f) ──  "
                      f"wait: {prof['prefetch_wait']/prof_interval*1000:.1f}ms  "
                      f"norm: {prof['gpu_normalize']/prof_interval*1000:.1f}ms  "
                      f"model: {prof['model']/prof_interval*1000:.1f}ms  "
                      f"→ {prof_interval/total_prof:.1f} FPS")

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0

    # Save signals as .npy
    if len(all_params) > 0:
        params_array = np.array(all_params)  # (N, 5)
        for i, name in enumerate(SIGNAL_NAMES):
            npy_path = os.path.join(output_dir, f"{video_basename}_{name}.npy")
            np.save(npy_path, params_array[:, i])

    return {
        "frames": frame_idx,
        "video_fps": prefetcher.fps,
        "inference_fps": round(avg_fps, 1),
        "duration_s": round(elapsed, 1),
    }


# =============================================================================
# ██ FFT ANALYSIS ON SAVED SIGNALS ██
# =============================================================================

# =============================================================================
# ██ SIGNAL PROCESSING (from signal_cropper.py) ██
# =============================================================================

def preprocess_signal(signal: np.ndarray, sample_rate: float,
                      lowcut: float = BP_LOWCUT, highcut: float = BP_HIGHCUT) -> np.ndarray:
    """
    Preprocess signal: DC removal + zero-phase Butterworth bandpass filter.
    Matches signal_cropper.py's SignalCropperApp.preprocess_signal().
    """
    from scipy.signal import butter, sosfiltfilt

    # Step 1: Remove DC component
    signal_no_dc = signal - np.mean(signal)

    # Step 2: Design Butterworth bandpass filter
    nyquist = sample_rate / 2.0

    # Clamp filter frequencies to valid range
    if lowcut >= nyquist or highcut >= nyquist:
        lowcut = min(lowcut, nyquist * 0.9)
        highcut = min(highcut, nyquist * 0.95)
    if lowcut >= highcut:
        return signal_no_dc

    low = lowcut / nyquist
    high = highcut / nyquist

    sos = butter(BP_ORDER, [low, high], btype='band', output='sos')

    # Step 3: Apply zero-phase filtering
    try:
        min_padlen = 3 * max(len(sos) * 2, 1)
        if len(signal_no_dc) <= min_padlen:
            return signal_no_dc
        preprocessed = sosfiltfilt(sos, signal_no_dc)
    except Exception:
        return signal_no_dc

    return preprocessed


def compute_fft(signal: np.ndarray, sample_rate: float):
    """
    Compute FFT with Hanning window.
    Returns: (frequencies, fft_magnitude, fft_magnitude_db)
    Matches signal_cropper.py's SignalCropperApp.compute_fft().
    """
    n = len(signal)

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    windowed_signal = signal * window

    # Compute FFT
    fft_result = np.fft.fft(windowed_signal)
    fft_magnitude = np.abs(fft_result[:n // 2])
    fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-10)

    # Frequency axis
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

    return frequencies, fft_magnitude, fft_magnitude_db


# =============================================================================
# ██ FFT ANALYSIS ON SAVED SIGNALS ██
# =============================================================================

def run_fft_analysis(output_dir: str, ffts_dir: str,
                     video_basename: str, sample_rate: float = SAMPLE_RATE):
    """
    Load .npy signals from output_dir, preprocess, compute FFT, save to ffts_dir.
    Processes all 5 signals (x0, y0, a, b, theta_rad).
    """
    os.makedirs(ffts_dir, exist_ok=True)

    for name in SIGNAL_NAMES:
        npy_path = os.path.join(output_dir, f"{video_basename}_{name}.npy")
        if not os.path.exists(npy_path):
            continue

        signal = np.load(npy_path)

        # Preprocess: DC removal + bandpass filter
        preprocessed = preprocess_signal(signal, sample_rate)

        # Compute FFT
        frequencies, fft_mag, fft_mag_db = compute_fft(preprocessed, sample_rate)

        # Save FFT results
        np.save(os.path.join(ffts_dir, f"{video_basename}_{name}_fft_freq.npy"), frequencies)
        np.save(os.path.join(ffts_dir, f"{video_basename}_{name}_fft_mag.npy"), fft_mag)
        np.save(os.path.join(ffts_dir, f"{video_basename}_{name}_fft_mag_db.npy"), fft_mag_db)


# =============================================================================
# ██ GPU WORKER ██
# =============================================================================

def gpu_worker(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
               cache_lock: mp.Lock, shared_cache_path: str):
    """
    Worker process: pulls samples from queue, runs inference + FFT on assigned GPU.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    print(f"\n  [GPU-{gpu_id}] Starting worker on {torch.cuda.get_device_name(gpu_id)}")
    print(f"  [GPU-{gpu_id}] VRAM: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")

    # Load model on this GPU
    model = load_model(CHECKPOINT, IMG_SIZE, device)

    videos_processed = 0
    total_frames = 0
    t_worker_start = time.time()

    while True:
        try:
            sample = task_queue.get(timeout=5)
        except Exception:
            # Queue empty, check if sentinel was already posted
            continue

        # Poison pill — exit
        if sample == SENTINEL:
            elapsed = time.time() - t_worker_start
            print(f"\n  [GPU-{gpu_id}] ✓ Worker done — {videos_processed} videos, "
                  f"{total_frames} frames in {elapsed:.0f}s")
            result_queue.put({"gpu_id": gpu_id, "type": "done",
                              "videos": videos_processed, "frames": total_frames})
            break

        sample_id = sample["sample_id"]
        video_path = sample["video_path"]
        output_dir = sample["output_dir"]
        ffts_dir = sample["ffts_dir"]
        video_basename = sample["video_basename"]

        try:
            # -- Run inference --
            result = run_inference_on_video(
                model, video_path, output_dir, video_basename, device
            )

            if "error" in result:
                result_queue.put({
                    "gpu_id": gpu_id, "type": "error",
                    "sample_id": sample_id, "error": result["error"]
                })
                continue

            # -- Run FFT analysis --
            run_fft_analysis(output_dir, ffts_dir, video_basename)

            # -- Update cache --
            with cache_lock:
                cache = load_cache(shared_cache_path)
                cache[sample_id] = {
                    "status": "ok",
                    "timestamp": datetime.now().isoformat(),
                    "gpu": gpu_id,
                    "frames": result["frames"],
                    "inference_fps": result["inference_fps"],
                }
                save_cache(shared_cache_path, cache)

            videos_processed += 1
            total_frames += result["frames"]

            # -- Report progress --
            result_queue.put({
                "gpu_id": gpu_id,
                "type": "progress",
                "sample_id": sample_id,
                "frames": result["frames"],
                "inference_fps": result["inference_fps"],
                "duration_s": result["duration_s"],
                "worker_total": videos_processed,
            })

        except Exception as e:
            tb = traceback.format_exc()
            result_queue.put({
                "gpu_id": gpu_id, "type": "error",
                "sample_id": sample_id, "error": str(e), "traceback": tb
            })
            # Still mark as attempted to avoid infinite retry
            with cache_lock:
                cache = load_cache(shared_cache_path)
                cache[sample_id] = {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "gpu": gpu_id,
                    "error": str(e),
                }
                save_cache(shared_cache_path, cache)


# =============================================================================
# ██ CSV FEEDER THREAD ██
# =============================================================================

def csv_feeder(task_queue: mp.Queue, stop_event: threading.Event,
               cache_path: str, stats: dict, num_gpus: int):
    """
    Periodically re-reads the manifest CSV, pushes new (uncached) samples
    into the task queue. Sends sentinels when all work is done.
    """
    already_queued = set()
    reload_count = 0
    idle_rounds = 0  # consecutive reloads with nothing new to queue

    while not stop_event.is_set():
        reload_count += 1
        cache = load_cache(cache_path)
        samples = parse_manifest(MANIFEST_CSV)

        stats["total_in_csv"] = len(samples)
        stats["total_cached"] = len(cache)

        new_count = 0
        pending_uncached = 0
        for sample in samples:
            sid = sample["sample_id"]
            if sid in cache:
                continue
            if sid in already_queued:
                pending_uncached += 1  # queued but not yet cached (still processing)
                continue
            if not os.path.isfile(sample["video_path"]):
                continue

            task_queue.put(sample)
            already_queued.add(sid)
            new_count += 1
            pending_uncached += 1

        if new_count > 0 or reload_count == 1:
            idle_rounds = 0
            print(f"  [Feeder] Reload #{reload_count}: {new_count} new queued "
                  f"(CSV: {len(samples)}, cached: {len(cache)}, "
                  f"pending: {pending_uncached})")
        else:
            idle_rounds += 1

        # If queue is empty AND no pending work AND multiple idle rounds → done
        if pending_uncached == 0 and idle_rounds >= 3 and task_queue.empty():
            print(f"  [Feeder] All samples processed! Sending {num_gpus} stop signals...")
            for _ in range(num_gpus):
                task_queue.put(SENTINEL)
            break

        # Wait for next reload interval (but check stop_event frequently)
        for _ in range(CSV_RELOAD_SEC * 2):
            if stop_event.is_set():
                return
            time.sleep(0.5)

    print("  [Feeder] Stopped.")


# =============================================================================
# ██ TENSORBOARD MONITOR THREAD ██
# =============================================================================

def tensorboard_monitor(result_queue: mp.Queue, stop_event: threading.Event,
                        stats: dict, num_gpus: int):
    """
    Reads worker results from result_queue, logs to TensorBoard,
    and prints progress to console.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(TENSORBOARD_LOG)
        has_tb = True
        print(f"  [Monitor] TensorBoard logging to: {TENSORBOARD_LOG}")
        print(f"  [Monitor] Run: tensorboard --logdir {TENSORBOARD_LOG}")
    except ImportError:
        has_tb = False
        print("  [Monitor] tensorboard not installed, console-only monitoring")
        writer = None

    global_step = 0
    total_processed = 0
    total_errors = 0
    per_gpu_count = {i: 0 for i in range(num_gpus)}
    t_start = time.time()
    recent_times = []  # last N video durations for rolling average
    workers_done = 0

    while not stop_event.is_set() or not result_queue.empty():
        try:
            msg = result_queue.get(timeout=2)
        except Exception:
            continue

        msg_type = msg.get("type", "")
        gpu_id = msg.get("gpu_id", 0)

        if msg_type == "done":
            workers_done += 1
            if workers_done >= num_gpus:
                print(f"\n  [Monitor] All {num_gpus} workers finished!")
                break
            continue

        if msg_type == "error":
            total_errors += 1
            sid = msg.get("sample_id", "?")
            err = msg.get("error", "unknown")
            print(f"  [GPU-{gpu_id}] ✗ ERROR {sid}: {err}")
            if has_tb:
                writer.add_scalar("errors/count", total_errors, global_step)
            continue

        if msg_type == "progress":
            global_step += 1
            total_processed += 1
            per_gpu_count[gpu_id] = per_gpu_count.get(gpu_id, 0) + 1

            sid = msg["sample_id"]
            frames = msg["frames"]
            ifps = msg["inference_fps"]
            dur = msg["duration_s"]
            recent_times.append(dur)
            if len(recent_times) > 50:
                recent_times.pop(0)

            # Compute stats
            elapsed_total = time.time() - t_start
            vids_per_min = total_processed / (elapsed_total / 60) if elapsed_total > 0 else 0
            avg_sec = np.mean(recent_times) if recent_times else 0
            total_csv = stats.get("total_in_csv", 0)
            remaining = max(0, total_csv - total_processed - stats.get("total_cached", 0) + total_processed)
            pct = (total_processed / total_csv * 100) if total_csv > 0 else 0

            # ETA
            if avg_sec > 0 and remaining > 0:
                eta_sec = (remaining * avg_sec) / num_gpus
                eta_h = int(eta_sec // 3600)
                eta_m = int((eta_sec % 3600) // 60)
                eta_str = f"{eta_h}h {eta_m}m"
            else:
                eta_str = "N/A"

            # Console output
            print(f"  [GPU-{gpu_id}] {total_processed}/{total_csv} "
                  f"({pct:.1f}%) | {sid} | {frames}f | "
                  f"{ifps} FPS | {dur:.1f}s | "
                  f"ETA: {eta_str} | Err: {total_errors}")

            # TensorBoard
            if has_tb:
                writer.add_scalar("throughput/videos_per_min", vids_per_min, global_step)
                writer.add_scalar("progress/total_processed", total_processed, global_step)
                writer.add_scalar("progress/total_remaining", remaining, global_step)
                writer.add_scalar("progress/percent_complete", pct, global_step)
                writer.add_scalar("timing/avg_seconds_per_video", avg_sec, global_step)
                writer.add_scalar("timing/inference_fps", ifps, global_step)
                writer.add_scalar(f"gpu_{gpu_id}/videos_processed",
                                  per_gpu_count[gpu_id], global_step)
                writer.add_scalar("errors/count", total_errors, global_step)
                if global_step % 10 == 0:
                    writer.flush()

    if has_tb:
        # Final summary
        writer.add_text("summary", (
            f"Total processed: {total_processed}\n"
            f"Total errors: {total_errors}\n"
            f"Total time: {time.time() - t_start:.0f}s\n"
            f"Avg speed: {total_processed / max(1, (time.time() - t_start) / 60):.1f} vids/min"
        ), global_step)
        writer.flush()
        writer.close()

    print(f"\n{'=' * 60}")
    print(f"  BATCH INFERENCE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total processed : {total_processed}")
    print(f"  Total errors    : {total_errors}")
    print(f"  Total time      : {time.time() - t_start:.0f}s")
    print(f"  Avg speed       : {total_processed / max(1, (time.time() - t_start) / 60):.1f} vids/min")
    for i in range(num_gpus):
        print(f"  GPU-{i}           : {per_gpu_count.get(i, 0)} videos")
    print(f"{'=' * 60}")


# =============================================================================
# ██ DRY RUN ██
# =============================================================================

def dry_run():
    """Preview which samples would be processed without actually running."""
    print("=" * 60)
    print("  DRY RUN — No inference will be performed")
    print("=" * 60)

    cache = load_cache(CACHE_FILE)
    samples = parse_manifest(MANIFEST_CSV)

    total = len(samples)
    cached = sum(1 for s in samples if s["sample_id"] in cache)
    missing_video = 0
    to_process = []

    for s in samples:
        if s["sample_id"] in cache:
            continue
        if not os.path.isfile(s["video_path"]):
            missing_video += 1
            continue
        to_process.append(s)

    print(f"\n  Manifest CSV     : {MANIFEST_CSV}")
    print(f"  Checkpoint       : {CHECKPOINT}")
    print(f"  Total in CSV     : {total}")
    print(f"  Already cached   : {cached}")
    print(f"  Missing video    : {missing_video}")
    print(f"  To process       : {len(to_process)}")
    print(f"  GPUs             : {NUM_GPUS}")

    if len(to_process) > 0:
        print(f"\n  First 10 samples to process:")
        for s in to_process[:10]:
            print(f"    {s['sample_id']} → {s['video_path']}")
        if len(to_process) > 10:
            print(f"    ... and {len(to_process) - 10} more")

    # Estimate time
    avg_sec_per_video = 8.0  # rough estimate for 450-frame video on A100
    est_total_sec = (len(to_process) * avg_sec_per_video) / NUM_GPUS
    est_h = int(est_total_sec // 3600)
    est_m = int((est_total_sec % 3600) // 60)
    print(f"\n  Estimated time   : ~{est_h}h {est_m}m (assuming {avg_sec_per_video}s/video)")
    print("=" * 60)


# =============================================================================
# ██ MAIN ██
# =============================================================================

def main():
    # -- Parse args --
    dry = "--dry-run" in sys.argv or "--dry_run" in sys.argv

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CondSeg Batch Inference + FFT Pipeline                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Manifest CSV   : {MANIFEST_CSV}")
    print(f"  Checkpoint     : {CHECKPOINT}")
    print(f"  Image size     : {IMG_SIZE}")
    print(f"  AMP (FP16)     : {USE_AMP}")
    print(f"  GPUs           : {NUM_GPUS}")
    print(f"  Cache file     : {CACHE_FILE}")
    print(f"  TensorBoard    : {TENSORBOARD_LOG}")
    print(f"  Sample rate    : {SAMPLE_RATE} Hz")
    print(f"  Bandpass       : {BP_LOWCUT}-{BP_HIGHCUT} Hz (order {BP_ORDER})")
    print(f"  CSV reload     : every {CSV_RELOAD_SEC}s")
    print()

    if dry:
        dry_run()
        return

    # -- Validate prerequisites --
    if not os.path.isfile(MANIFEST_CSV):
        print(f"  ✗ Manifest CSV not found: {MANIFEST_CSV}")
        sys.exit(1)
    if not os.path.isfile(CHECKPOINT):
        print(f"  ✗ Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    available_gpus = torch.cuda.device_count()
    actual_gpus = min(NUM_GPUS, available_gpus)
    if actual_gpus == 0:
        print("  ✗ No CUDA GPUs available!")
        sys.exit(1)
    if actual_gpus < NUM_GPUS:
        print(f"  ⚠ Requested {NUM_GPUS} GPUs but only {available_gpus} available. Using {actual_gpus}.")

    for i in range(actual_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU-{i}: {name} ({mem:.1f} GB)")
    print()

    # -- Set up multiprocessing --
    mp.set_start_method("spawn", force=True)

    task_queue = mp.Queue(maxsize=100)
    result_queue = mp.Queue()
    cache_lock = mp.Lock()

    # Shared stats dict for feeder ↔ monitor communication
    stats = {"total_in_csv": 0, "total_cached": 0}

    # -- Start GPU workers --
    workers = []
    for gpu_id in range(actual_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, cache_lock, CACHE_FILE),
            daemon=True
        )
        p.start()
        workers.append(p)
    print(f"  ✓ {actual_gpus} GPU workers started\n")

    # -- Start CSV feeder thread --
    stop_event = threading.Event()
    feeder = threading.Thread(
        target=csv_feeder,
        args=(task_queue, stop_event, CACHE_FILE, stats, actual_gpus),
        daemon=True
    )
    feeder.start()

    # -- Start TensorBoard monitor (runs in main thread) --
    try:
        tensorboard_monitor(result_queue, stop_event, stats, actual_gpus)
    except KeyboardInterrupt:
        print("\n  ⚠ Interrupted! Sending stop signals...")

    # -- Cleanup --
    stop_event.set()

    # Fallback: send sentinels in case feeder didn't (e.g. KeyboardInterrupt)
    for _ in range(actual_gpus):
        try:
            task_queue.put_nowait(SENTINEL)
        except Exception:
            pass

    # Wait for workers
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            print(f"  ⚠ Worker {p.pid} still alive, terminating...")
            p.terminate()

    feeder.join(timeout=10)
    print("\n  ✓ All processes cleaned up. Goodbye!")


if __name__ == "__main__":
    main()
