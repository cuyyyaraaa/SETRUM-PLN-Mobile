"""
prepare_data.py — Menyiapkan CSV latih dari Report Review Negatif PLN
====================================================================
Membaca file Excel berlabel manual tim PLN (sheet DATA) dan menghasilkan
CSV terpisah per target: sentimen, layer1, layer2, layer3.

Teks dinormalisasi dulu agar konsisten dengan yang dipakai saat inferensi.

Cara pakai:
    python prepare_data.py --xlsx "20260520_-_Report_Review_Negatif__OK_.xlsx"

Output: data_sentiment.csv, data_layer1.csv, data_layer2.csv, data_layer3.csv
Catatan: semua baris di sini berlabel NEGATIF (sesuai fokus PLN), jadi
data_sentiment.csv perlu ditambah contoh positif/netral dari scraping
agar model sentimen bisa membedakan (lihat README).
"""
import argparse, sys, os, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.agents.normalizer import normalize
from training.reconcile_labels import reconcile_dataframe, clean_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--min_count', type=int, default=10,
                    help='kelas dgn sampel < ini digabung jadi "Lainnya" (untuk L3)')
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name='DATA', header=2)
    df.columns = [str(c).strip() for c in df.columns]
    # kolom: ISI ULASAN, LAYER 1..5
    df = df.rename(columns={'ISI ULASAN': 'text'})
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str).map(normalize)
    df = df[df['text'].str.split().str.len() >= 1]

    df['LAYER 1'] = df['LAYER 1'].map(clean_label)
    df['LAYER 2'] = df['LAYER 2'].map(clean_label)
    df, report = reconcile_dataframe(df)
    if len(report):
        review = report[report['method'] == 'NEEDS_MANUAL_REVIEW']
        print(f"[Rekonsiliasi] {len(report)} label direkonsiliasi otomatis "
              f"({dict(report['method'].value_counts())}).")
        if len(review):
            report.to_csv('reconcile_report.csv', index=False)
            print(f"  ⚠️  {len(review)} butuh review manual -> lihat reconcile_report.csv "
                  f"SEBELUM training (baris ini dipakai sebagai ground truth).")

    for layer, col in [('layer1', 'LAYER 1'), ('layer2', 'LAYER 2'), ('layer3', 'LAYER 3')]:
        if col not in df.columns:
            print(f"  kolom {col} tidak ada, lewati"); continue
        sub = df[['text', col]].copy()
        sub['label'] = sub[col]  # sudah dibersihkan + direkonsiliasi di atas
        sub = sub.dropna(subset=['label'])[['text', 'label']]

        if layer == 'layer3':
            vc = sub['label'].value_counts()
            rare = vc[vc < args.min_count].index
            sub.loc[sub['label'].isin(rare), 'label'] = 'Lainnya'
            print(f"  L3: {len(rare)} kelas langka digabung -> 'Lainnya'")

        out = f'data_{layer}.csv'
        sub.to_csv(out, index=False)
        print(f"{out}: {len(sub)} baris, {sub['label'].nunique()} kelas")

    # sentimen: semua negatif (perlu ditambah positif/netral manual)
    sent = df[['text']].copy(); sent['label'] = 'Negatif'
    sent.to_csv('data_sentiment_neg_only.csv', index=False)
    print(f"data_sentiment_neg_only.csv: {len(sent)} baris (SEMUA negatif — "
          f"tambah contoh positif/netral sebelum training sentimen)")


if __name__ == '__main__':
    main()
