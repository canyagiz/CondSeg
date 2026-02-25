"""
FFT Magnitude vs Delta PX Scatter Plots
========================================
Excel'deki f=xxxHz sheet'lerini okur, her video × frekans için
delta_px vs FFT peak magnitude scatter plot üretir.

Kullanım:
    python fft_scatter.py
    python fft_scatter.py --base-dir /path/to/data   # farklı data dizini

Çıktı:
    scatter_plots/<video_name>_f<f_hz>Hz.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Ayarlar ──────────────────────────────────────────────────────────────
EXCEL_PATH = Path(r"C:\Users\aliya\workspace\CondSeg\benchmark_analiz.xlsx")
BASE_DIR   = Path(r"glaucot-data-storage/simulation-benchmark/videos")
OUTPUT_DIR = Path(r"C:\Users\aliya\workspace\CondSeg\scatter_plots")

SIGNAL_NAME = "a"  # semi-major axis (iris genişliği)
VIDEO_BASENAME = "03_full"


def get_fft_peak_at_freq(sample_id: int, target_freq: float) -> float | None:
    """
    sample_id'ye ait FFT dosyasını yükle,
    target_freq'e en yakın bin'deki magnitude'u döndür.
    """
    sid_str = f"{sample_id:06d}"
    fft_dir = BASE_DIR / sid_str / "frame_60" / "inferences" / "ffts"

    mag_path  = fft_dir / f"{VIDEO_BASENAME}_{SIGNAL_NAME}_fft_mag.npy"
    freq_path = fft_dir / f"{VIDEO_BASENAME}_{SIGNAL_NAME}_fft_freq.npy"

    if not mag_path.exists() or not freq_path.exists():
        return None

    magnitudes  = np.load(mag_path)
    frequencies = np.load(freq_path)

    # target_freq'e en yakın index
    idx = np.argmin(np.abs(frequencies - target_freq))
    return float(magnitudes[idx])


def main():
    parser = argparse.ArgumentParser(description="FFT Mag vs Delta PX scatter plots")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory for video data (overrides default)")
    parser.add_argument("--excel", type=str, default=None,
                        help="Path to benchmark_analiz.xlsx (overrides default)")
    args = parser.parse_args()

    global BASE_DIR, EXCEL_PATH
    if args.base_dir:
        BASE_DIR = Path(args.base_dir)
    if args.excel:
        EXCEL_PATH = Path(args.excel)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Excel'den f=xxxHz sheet'lerini bul ──
    xls = pd.ExcelFile(EXCEL_PATH)
    freq_sheets = [s for s in xls.sheet_names if s.startswith("f=")]
    print(f"📊 {len(freq_sheets)} frekans sheet'i bulundu: {freq_sheets}")

    total_plots = 0
    total_missing = 0

    for sheet_name in freq_sheets:
        # f=0.8Hz → 0.8
        f_hz = float(sheet_name.replace("f=", "").replace("Hz", ""))
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)

        # Boş ayraç satırlarını temizle
        df = df.dropna(subset=["sample_id"]).copy()
        df["sample_id"] = df["sample_id"].astype(int)

        print(f"\n{'─'*60}")
        print(f"  Sheet: {sheet_name}  |  f_hz = {f_hz}  |  {len(df)} sample")

        # video_name grupları
        for video_name, grp in df.groupby("video_name", sort=True):
            delta_px_vals = []
            fft_mag_vals  = []
            missing = 0

            for _, row in grp.iterrows():
                sid = int(row["sample_id"])
                dpx = float(row["delta_px"])
                peak_mag = get_fft_peak_at_freq(sid, f_hz)

                if peak_mag is not None:
                    delta_px_vals.append(dpx)
                    fft_mag_vals.append(peak_mag)
                else:
                    missing += 1

            total_missing += missing

            if len(delta_px_vals) == 0:
                print(f"    ⚠ {video_name}: tüm FFT dosyaları eksik, atlanıyor")
                continue

            # ── Scatter plot ──
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(delta_px_vals, fft_mag_vals,
                       s=80, c="#2563eb", edgecolors="#1e40af",
                       alpha=0.8, zorder=3)

            # Her noktayı etiketle
            for x, y in zip(delta_px_vals, fft_mag_vals):
                ax.annotate(f"{y:.3f}", (x, y),
                            textcoords="offset points", xytext=(0, 10),
                            fontsize=8, ha="center", color="#374151")

            ax.set_xlabel("Delta PX (piksel)", fontsize=11)
            ax.set_ylabel(f"FFT Peak Magnitude ({SIGNAL_NAME})", fontsize=11)
            ax.set_title(f"{video_name}  —  f = {f_hz} Hz", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xscale("log")

            fig.tight_layout()

            # Dosya adından Türkçe karakter ve boşlukları temizle
            safe_name = (video_name
                         .replace(" ", "_")
                         .replace("ğ", "g").replace("ş", "s")
                         .replace("ç", "c").replace("ü", "u")
                         .replace("ö", "o").replace("ı", "i"))
            fname = f"{safe_name}_f{f_hz}Hz.png"
            fig.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

            total_plots += 1
            if missing:
                print(f"    ✓ {video_name}: {len(delta_px_vals)} nokta, {missing} eksik → {fname}")
            else:
                print(f"    ✓ {video_name}: {len(delta_px_vals)} nokta → {fname}")

    print(f"\n{'═'*60}")
    print(f"  ✅ Toplam {total_plots} scatter plot → {OUTPUT_DIR}")
    if total_missing:
        print(f"  ⚠  {total_missing} eksik FFT dosyası atlandı")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
