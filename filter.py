import pandas as pd
from pathlib import Path

# ── Ayarlar ──────────────────────────────────────────────────────────────
CSV_PATH = Path(r"C:\Users\aliya\workspace\CondSeg\simulation-benchmark_manifest.csv")
OUTPUT_XLSX = CSV_PATH.parent / "benchmark_analiz.xlsx"

# Gösterilecek sütunlar (okunabilirlik için gereksiz path'leri kısalt)
DISPLAY_COLS = ['sample_id', 'video_name', 'eye', 'f_hz', 'delta_px',
                'head_tier', 'gaze_tier', 'r_px', 'delta_px_at_peak', 'status']

df = pd.read_csv(CSV_PATH)
# Sadece gösterilecek sütunları tut
display_df = df[DISPLAY_COLS].copy()

# ── Filtre tanımları ─────────────────────────────────────────────────────
# Her bir noise kategorisi için filtre: (sheet_adı, head_tier filtresi, gaze_tier filtresi)
NOISE_TIERS = ['none', 'light', 'moderate', 'heavy']

filters = {}

# 1) Temiz veri: hiç noise yok
filters['Temiz (noise yok)'] = display_df[
    (df['head_tier'] == 'none') & (df['gaze_tier'] == 'none')
].copy()

# 2) Sadece Head Noise (gaze=none, head != none)
for tier in ['light', 'moderate', 'heavy']:
    name = f'Head {tier.capitalize()}'
    filters[name] = display_df[
        (df['head_tier'] == tier) & (df['gaze_tier'] == 'none')
    ].copy()

# 3) Sadece Gaze Noise (head=none, gaze != none)
for tier in ['light', 'moderate', 'heavy']:
    name = f'Gaze {tier.capitalize()}'
    filters[name] = display_df[
        (df['head_tier'] == 'none') & (df['gaze_tier'] == tier)
    ].copy()

# 4) Kombine Noise (head != none AND gaze != none)
for h_tier in ['light', 'moderate', 'heavy']:
    for g_tier in ['light', 'moderate', 'heavy']:
        name = f'H-{h_tier[:3]} G-{g_tier[:3]}'
        filters[name] = display_df[
            (df['head_tier'] == h_tier) & (df['gaze_tier'] == g_tier)
        ].copy()

# ── Özet tablo: her kategori için satır sayısı + f_hz/delta_px coverage ─
ozet_rows = []
for name, sub_df in filters.items():
    ozet_rows.append({
        'Kategori': name,
        'Sample Sayısı': len(sub_df),
        'Benzersiz f_hz': str(sorted(sub_df['f_hz'].unique().tolist())) if len(sub_df) else '-',
        'Benzersiz delta_px': str(sorted(sub_df['delta_px'].unique().tolist())) if len(sub_df) else '-',
        'Video Sayısı': sub_df['video_name'].nunique() if len(sub_df) else 0,
    })
ozet_df = pd.DataFrame(ozet_rows)

# ── Genel istatistikler ─────────────────────────────────────────────────
clean_count = len(filters['Temiz (noise yok)'])
genel = pd.DataFrame({
    'Metrik': [
        'Toplam satır',
        'Temiz (noise yok)',
        'Sadece Head noise',
        'Sadece Gaze noise',
        'Kombine noise (head+gaze)',
        'f_hz değerleri',
        'delta_px değerleri',
        'Benzersiz video',
    ],
    'Değer': [
        len(df),
        clean_count,
        sum(len(filters[f'Head {t.capitalize()}']) for t in ['light','moderate','heavy']),
        sum(len(filters[f'Gaze {t.capitalize()}']) for t in ['light','moderate','heavy']),
        sum(len(v) for k, v in filters.items() if k.startswith('H-')),
        str(sorted(df['f_hz'].unique().tolist())),
        str(sorted(df['delta_px'].unique().tolist())),
        df['video_name'].nunique(),
    ],
})

# ── f_hz × delta_px pivot (temiz veri) ───────────────────────────────────
clean_df = filters['Temiz (noise yok)']
if len(clean_df) > 0:
    pivot = (
        clean_df.groupby(['f_hz', 'delta_px'])['sample_id']
        .count().reset_index().rename(columns={'sample_id': 'count'})
        .pivot(index='f_hz', columns='delta_px', values='count')
        .fillna(0).astype(int)
    )
else:
    pivot = pd.DataFrame()

# ── Excel'e yaz ─────────────────────────────────────────────────────────
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    # Genel özet + kategori bazlı sayılar
    genel.to_excel(writer, sheet_name='Genel Özet', index=False)
    ozet_df.to_excel(writer, sheet_name='Kategori Özet', index=False)
    if len(pivot) > 0:
        pivot.to_excel(writer, sheet_name='f_hz × delta_px (Temiz)')

    # Her noise kategorisini ayrı sheet'e yaz
    for name, sub_df in filters.items():
        # Excel sheet adı max 31 karakter
        sheet_name = name[:31]
        sub_df.sort_values(['f_hz', 'delta_px', 'video_name']).to_excel(
            writer, sheet_name=sheet_name, index=False
        )

    # ── Frekans bazlı sheet'ler (sadece temiz veri: head=none, gaze=none) ─
    clean_all = display_df[
        (df['head_tier'] == 'none') & (df['gaze_tier'] == 'none')
    ].copy()

    for f_val in sorted(clean_all['f_hz'].unique()):
        freq_df = clean_all[clean_all['f_hz'] == f_val].copy()
        freq_df = freq_df.sort_values(['video_name', 'delta_px'])
        sheet_name = f'f={f_val}Hz'[:31]

        # video_name grupları arasına boş satır ekle
        grouped_parts = []
        for vname, grp in freq_df.groupby('video_name', sort=True):
            grouped_parts.append(grp)
            # Boş ayraç satırı
            blank = pd.DataFrame({c: [''] for c in freq_df.columns})
            grouped_parts.append(blank)
        if grouped_parts:
            grouped_parts.pop()  # son boş satırı kaldır
        freq_grouped = pd.concat(grouped_parts, ignore_index=True)
        freq_grouped.to_excel(writer, sheet_name=sheet_name, index=False)

    # Kolon genişliklerini otomatik ayarla
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            col_letter = col[0].column_letter
            max_len = max((len(str(c.value or '')) for c in col), default=8)
            ws.column_dimensions[col_letter].width = min(max_len + 3, 55)

print(f"✅ Analiz dosyası: {OUTPUT_XLSX}")
print(f"   Toplam: {len(df)} satır")
print(f"\n📋 Sheet bazında dağılım:")
for name, sub_df in filters.items():
    print(f"   {name:30s} → {len(sub_df):4d} sample")