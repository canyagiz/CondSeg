"""
Raw Signal Diagnostic Scatter Plots
====================================
Ham 'a' sinyalini yükleyip 3 farklı yöntemle FFT hesaplar,
delta_px vs FFT magnitude scatter plotları üretir.

Yöntemler:
  1. Raw FFT (bandpass yok, sadece DC removal + Hanning window)
  2. Bandpass FFT (0.5–4.0 Hz Butterworth + Hanning window)
  3. Harmonics Sum (fundamental + 2., 3., 4., 5. harmoniklerin mag toplamı)

Kullanım:
    python signal_scatter.py
    python signal_scatter.py --base-dir /path/to/data

Çıktı:
    signal_scatter_plots/<video_name>.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

# ── Ayarlar ──────────────────────────────────────────────────────────────
EXCEL_PATH = Path(r"C:\Users\aliya\workspace\CondSeg\benchmark_analiz.xlsx")
BASE_DIR   = Path(r"glaucot-data-storage/simulation-benchmark/videos")
OUTPUT_DIR = Path(r"C:\Users\aliya\workspace\CondSeg\signal_scatter_plots")

SIGNAL_NAME    = "a"
VIDEO_BASENAME = "03_full"
SAMPLE_RATE    = 60.0
BP_LOWCUT      = 0.5
BP_HIGHCUT     = 4.0
BP_ORDER       = 4
N_HARMONICS    = 5   # fundamental + 4 harmonics


# ── Sinyal İşleme Fonksiyonları ──────────────────────────────────────────

def load_raw_signal(sample_id: int) -> np.ndarray | None:
    """Ham sinyal .npy dosyasını yükle."""
    sid_str = f"{sample_id:06d}"
    sig_path = BASE_DIR / sid_str / "frame_60" / "inferences" / f"{VIDEO_BASENAME}_{SIGNAL_NAME}.npy"
    if not sig_path.exists():
        return None
    return np.load(sig_path)


def compute_fft(signal: np.ndarray, sample_rate: float):
    """DC removal + Hanning window + FFT. Returns (frequencies, magnitudes)."""
    signal_no_dc = signal - np.mean(signal)
    n = len(signal_no_dc)
    window = np.hanning(n)
    windowed = signal_no_dc * window
    fft_result = np.fft.fft(windowed)
    magnitudes = np.abs(fft_result[:n // 2])
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
    return frequencies, magnitudes


def apply_bandpass(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    """DC removal + Butterworth bandpass filtre."""
    signal_no_dc = signal - np.mean(signal)
    nyquist = sample_rate / 2.0
    low = BP_LOWCUT / nyquist
    high = BP_HIGHCUT / nyquist
    sos = butter(BP_ORDER, [low, high], btype='band', output='sos')
    try:
        return sosfiltfilt(sos, signal_no_dc)
    except Exception:
        return signal_no_dc


def get_mag_at_freq(frequencies, magnitudes, target_freq):
    """En yakın FFT bin'indeki magnitude."""
    idx = np.argmin(np.abs(frequencies - target_freq))
    return float(magnitudes[idx])


def get_harmonics_sum(frequencies, magnitudes, fundamental_freq, n_harmonics=N_HARMONICS):
    """Fundamental + harmoniklerdeki magnitudelerin toplamı."""
    total = 0.0
    for h in range(1, n_harmonics + 1):
        harm_freq = fundamental_freq * h
        if harm_freq > frequencies[-1]:
            break
        idx = np.argmin(np.abs(frequencies - harm_freq))
        total += magnitudes[idx]
    return total


# ── 3 Yöntemle Analiz ────────────────────────────────────────────────────

def analyze_sample(sample_id: int, target_freq: float):
    """
    Bir sample için 3 yöntemle FFT magnitude hesapla.
    Returns: (raw_mag, bp_mag, harmonics_mag) or None
    """
    signal = load_raw_signal(sample_id)
    if signal is None or len(signal) < 30:
        return None

    # 1. Raw FFT (bandpass yok)
    freqs, mags = compute_fft(signal, SAMPLE_RATE)
    raw_mag = get_mag_at_freq(freqs, mags, target_freq)

    # 2. Bandpass + FFT
    bp_signal = apply_bandpass(signal, SAMPLE_RATE)
    freqs_bp, mags_bp = compute_fft(bp_signal, SAMPLE_RATE)
    bp_mag = get_mag_at_freq(freqs_bp, mags_bp, target_freq)

    # 3. Harmonics Sum (raw FFT üzerinden)
    harm_mag = get_harmonics_sum(freqs, mags, target_freq)

    return raw_mag, bp_mag, harm_mag


# ── Ana Fonksiyon ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Raw signal diagnostic scatter plots")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--excel", type=str, default=None)
    args = parser.parse_args()

    global BASE_DIR, EXCEL_PATH
    if args.base_dir:
        BASE_DIR = Path(args.base_dir)
    if args.excel:
        EXCEL_PATH = Path(args.excel)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Excel'den f=xxxHz sheet'lerini oku ──
    xls = pd.ExcelFile(EXCEL_PATH)
    freq_sheets = [s for s in xls.sheet_names if s.startswith("f=")]
    freq_sheets.sort(key=lambda s: float(s.replace("f=", "").replace("Hz", "")))
    print(f"📊 {len(freq_sheets)} frekans sheet'i bulundu")

    all_data = []
    for sheet_name in freq_sheets:
        f_hz = float(sheet_name.replace("f=", "").replace("Hz", ""))
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
        df = df.dropna(subset=["sample_id"]).copy()
        df["sample_id"] = df["sample_id"].astype(int)
        df["f_hz"] = f_hz
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    freq_values = sorted(combined["f_hz"].unique())
    video_names = sorted(combined["video_name"].unique())

    print(f"   {len(video_names)} video, {len(freq_values)} frekans")

    METHOD_NAMES = ["Raw FFT (bandpass yok)", "Bandpass + FFT", "Harmonics Sum"]
    METHOD_COLORS = ["#dc2626", "#2563eb", "#16a34a"]  # red, blue, green
    METHOD_EDGES  = ["#991b1b", "#1e40af", "#166534"]

    total_plots = 0

    for video_name in video_names:
        video_df = combined[combined["video_name"] == video_name]
        n_freq = len(freq_values)

        # 3 satır (yöntem) × n_freq sütun düzeni → 3 grup × 3×3 grid
        n_cols = 3
        n_method_rows = (n_freq + n_cols - 1) // n_cols  # 3 rows per method
        total_rows = n_method_rows * 3  # 3 methods

        fig, all_axes = plt.subplots(total_rows, n_cols,
                                      figsize=(6 * n_cols, 3.8 * total_rows))
        all_axes = np.array(all_axes).reshape(total_rows, n_cols)

        fig.suptitle(f"{video_name}", fontsize=18, fontweight="bold", y=0.995)

        for method_idx in range(3):
            row_offset = method_idx * n_method_rows

            for i, f_hz in enumerate(freq_values):
                r = row_offset + i // n_cols
                c = i % n_cols
                ax = all_axes[r, c]

                freq_grp = video_df[video_df["f_hz"] == f_hz]
                delta_px_vals = []
                mag_vals = []

                for _, row in freq_grp.iterrows():
                    sid = int(row["sample_id"])
                    dpx = float(row["delta_px"])
                    if dpx <= 0.01:
                        continue

                    result = analyze_sample(sid, f_hz)
                    if result is not None:
                        delta_px_vals.append(dpx)
                        mag_vals.append(result[method_idx])

                if len(delta_px_vals) > 0:
                    ax.scatter(delta_px_vals, mag_vals,
                               s=60, c=METHOD_COLORS[method_idx],
                               edgecolors=METHOD_EDGES[method_idx],
                               alpha=0.8, zorder=3)

                    for x, y in zip(delta_px_vals, mag_vals):
                        ax.annotate(f"{y:.2f}", (x, y),
                                    textcoords="offset points", xytext=(0, 8),
                                    fontsize=6.5, ha="center", color="#374151")

                    ax.set_xscale("log")
                    tick_vals = sorted(set(delta_px_vals))
                    ax.set_xticks(tick_vals)
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
                    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

                # Başlık: ilk satırda yöntem adı + frekans
                if i == 0:
                    ax.set_title(f"{METHOD_NAMES[method_idx]}\nf = {f_hz} Hz",
                                 fontsize=10, fontweight="bold",
                                 color=METHOD_COLORS[method_idx])
                else:
                    ax.set_title(f"f = {f_hz} Hz", fontsize=10, fontweight="bold")

                ax.set_xlabel("Delta PX", fontsize=8)
                ax.set_ylabel("Magnitude", fontsize=8)
                ax.grid(True, alpha=0.3, linestyle="--")
                ax.tick_params(labelsize=7)

            # Kullanılmayan subplot'ları gizle
            for i in range(n_freq, n_method_rows * n_cols):
                r = row_offset + i // n_cols
                c = i % n_cols
                all_axes[r, c].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.98])

        # Dosya adı
        safe_name = (video_name
                     .replace(" ", "_")
                     .replace("ğ", "g").replace("ş", "s")
                     .replace("ç", "c").replace("ü", "u")
                     .replace("ö", "o").replace("ı", "i"))
        fname = f"{safe_name}.png"
        fig.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        total_plots += 1
        print(f"  ✓ {video_name} → {fname}")

    print(f"\n{'═'*60}")
    print(f"  ✅ Toplam {total_plots} diagnostic plot → {OUTPUT_DIR}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
