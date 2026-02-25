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
import matplotlib.ticker as ticker
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
    freq_sheets.sort(key=lambda s: float(s.replace("f=", "").replace("Hz", "")))
    print(f"📊 {len(freq_sheets)} frekans sheet'i bulundu: {freq_sheets}")

    # ── Tüm sheet'leri oku, büyük bir DataFrame'e birleştir ──
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

    total_plots = 0
    total_missing = 0

    # ── Her video için 1 PNG (tüm frekanslar subplot olarak) ──
    for video_name in video_names:
        video_df = combined[combined["video_name"] == video_name]
        n_freq = len(freq_values)

        # Grid boyutunu hesapla (3 sütun)
        n_cols = 3
        n_rows = (n_freq + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
        axes = np.array(axes).flatten()  # tek boyutlu diziye çevir

        fig.suptitle(f"{video_name}", fontsize=16, fontweight="bold", y=0.98)

        for i, f_hz in enumerate(freq_values):
            ax = axes[i]
            freq_grp = video_df[video_df["f_hz"] == f_hz]

            delta_px_vals = []
            fft_mag_vals  = []

            for _, row in freq_grp.iterrows():
                sid = int(row["sample_id"])
                dpx = float(row["delta_px"])
                if dpx <= 0.01:
                    continue  # delta_px=0.01 atla
                peak_mag = get_fft_peak_at_freq(sid, f_hz)

                if peak_mag is not None:
                    delta_px_vals.append(dpx)
                    fft_mag_vals.append(peak_mag)
                else:
                    total_missing += 1

            if len(delta_px_vals) > 0:
                ax.scatter(delta_px_vals, fft_mag_vals,
                           s=70, c="#2563eb", edgecolors="#1e40af",
                           alpha=0.8, zorder=3)

                for x, y in zip(delta_px_vals, fft_mag_vals):
                    ax.annotate(f"{y:.3f}", (x, y),
                                textcoords="offset points", xytext=(0, 8),
                                fontsize=7, ha="center", color="#374151")

                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())

            ax.set_title(f"f = {f_hz} Hz", fontsize=11, fontweight="bold")
            ax.set_xlabel("Delta PX", fontsize=9)
            ax.set_ylabel(f"FFT Mag ({SIGNAL_NAME})", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=8)

        # Kullanılmayan subplot'ları gizle
        for j in range(n_freq, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

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
    print(f"  ✅ Toplam {total_plots} scatter plot → {OUTPUT_DIR}")
    if total_missing:
        print(f"  ⚠  {total_missing} eksik FFT dosyası atlandı")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
