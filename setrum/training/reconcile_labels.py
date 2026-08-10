"""
reconcile_labels.py — Rekonsiliasi label Excel PLN <-> hierarki resmi JSON
============================================================================
MASALAH: Excel labeling manual dari tim PLN kadang tidak persis sama dengan
hierarchy_pln_clean.json (JSON = sumber kebenaran / source of truth), karena:
  - newline/spasi ganda tersisip saat copy-paste ke Excel
  - beda kata sedikit ("Gagal Charging / Partially Charging" vs
    "Gagal / Partial Charging / Error Aplikasi")
Kalau dibiarkan, label ini jadi "kelas hantu" sendiri saat training, dan saat
inferensi hierarchy_loader gagal cocokkan opsi L4/L5 -> escalator jatuh ke
fallback yang salah.

STRATEGI (escalating, murah dulu baru mahal):
  1. EXACT match (setelah normalisasi whitespace + strip prefix angka)
     terhadap opsi yang VALID untuk path parent baris itu (l1,l2[,l3,l4]).
  2. FUZZY match (difflib, gratis, tanpa model) -- tangani typo/beda tipis.
  3. LLM zero-shot LOKAL (BART-MNLI, sudah dipakai escalator.py) -- hanya
     dipanggil untuk sisa kasus yang gagal di langkah 1-2, dan opsi yang
     ditawarkan ke model SELALU di-scope ke path parent yang benar (bukan
     nebak dari semua 96 label L5 sekaligus).

OUTPUT: reconcile_report.csv berisi setiap baris yang TIDAK exact-match,
metode yang dipakai, dan skor -- WAJIB direview manual sebelum dipakai
sebagai ground truth training, karena ini label yang menentukan kualitas
model, bukan sekadar file cache.

Pakai:
    python training/reconcile_labels.py --xlsx Dataaaa_PLNN.xlsx \
        --out data_labeled_reconciled.csv
"""
import argparse, os, re, sys, difflib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.agents.hierarchy_loader import get_hierarchy

FUZZY_CUTOFF = 0.72  # di bawah ini, serahkan ke LLM zero-shot


def clean_label(x):
    """Strip prefix angka ('1. ' / '1.1.1 ') + normalisasi whitespace/newline."""
    if x is None or (isinstance(x, float)):
        return None
    s = str(x).strip()
    s = re.sub(r'^\d+(\.\d+)*\.?\s*', '', s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s or None


_zs = None
_zs_tried = False


def _load_zeroshot():
    global _zs, _zs_tried
    if _zs is not None:
        return _zs
    if _zs_tried:
        return None
    _zs_tried = True
    try:
        from transformers import pipeline
        _zs = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=-1)
        print("[Reconcile] zero-shot lokal dimuat untuk rekonsiliasi label.")
    except Exception as e:
        print(f"[Reconcile] zero-shot tidak tersedia ({e}) -> sisa kasus ditandai NEEDS_MANUAL_REVIEW.")
        _zs = None
    return _zs


def _reconcile_one(raw_label, valid_options, context: str):
    """Return (matched_label, method, score). context = string path untuk log."""
    cleaned = clean_label(raw_label)
    if cleaned is None:
        return None, 'empty', 0.0
    if not valid_options:
        return cleaned, 'no_hierarchy_options', 0.0

    # 1. Exact match
    if cleaned in valid_options:
        return cleaned, 'exact', 1.0

    # 2. Fuzzy match (murah, tanpa model)
    close = difflib.get_close_matches(cleaned, valid_options, n=1, cutoff=FUZZY_CUTOFF)
    if close:
        score = difflib.SequenceMatcher(None, cleaned, close[0]).ratio()
        return close[0], 'fuzzy', round(score, 3)

    # 3. LLM zero-shot lokal, opsi di-scope ke valid_options saja
    zs = _load_zeroshot()
    if zs is not None:
        try:
            res = zs(cleaned, valid_options, multi_label=False)
            return res['labels'][0], 'llm_zeroshot', round(res['scores'][0], 3)
        except Exception as e:
            print(f"[Reconcile] LLM gagal utk '{cleaned}' ({context}): {e}")

    # Tidak ada cara otomatis yang berhasil -> tandai untuk review manual
    return cleaned, 'NEEDS_MANUAL_REVIEW', 0.0


def reconcile_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = get_hierarchy()
    report_rows = []
    out = df.copy()

    for idx, row in df.iterrows():
        l1 = clean_label(row.get('LAYER 1'))
        l2 = clean_label(row.get('LAYER 2'))
        l3_raw = row.get('LAYER 3')
        l4_raw = row.get('LAYER 4')
        l5_raw = row.get('LAYER 5')

        # L3: opsi valid = anak dari (l1,l2)
        l3_opts = h.get_l3_options(l1, l2) if (l1 and l2) else []
        l3, m3, s3 = _reconcile_one(l3_raw, l3_opts, f"{l1}>{l2}")
        out.at[idx, 'LAYER 3'] = l3
        if m3 not in ('exact', 'empty') and l3_raw is not None:
            report_rows.append({'row': idx, 'layer': 'L3', 'raw': str(l3_raw).strip(),
                                'matched': l3, 'method': m3, 'score': s3, 'context': f"{l1}>{l2}"})

        # L4: opsi valid = anak dari (l1,l2,l3) -- pakai l3 HASIL rekonsiliasi
        l4_opts = h.get_l4_options(l1, l2, l3) if (l1 and l2 and l3) else []
        l4, m4, s4 = _reconcile_one(l4_raw, l4_opts, f"{l1}>{l2}>{l3}")
        out.at[idx, 'LAYER 4'] = l4
        if m4 not in ('exact', 'empty') and l4_raw is not None:
            report_rows.append({'row': idx, 'layer': 'L4', 'raw': str(l4_raw).strip(),
                                'matched': l4, 'method': m4, 'score': s4, 'context': f"{l1}>{l2}>{l3}"})

        # L5: opsi valid = anak dari (l1,l2,l3,l4)
        l5_opts = h.get_l5_options(l1, l2, l3, l4) if (l1 and l2 and l3 and l4) else []
        l5, m5, s5 = _reconcile_one(l5_raw, l5_opts, f"{l1}>{l2}>{l3}>{l4}")
        out.at[idx, 'LAYER 5'] = l5
        if m5 not in ('exact', 'empty') and l5_raw is not None:
            report_rows.append({'row': idx, 'layer': 'L5', 'raw': str(l5_raw).strip(),
                                'matched': l5, 'method': m5, 'score': s5, 'context': f"{l1}>{l2}>{l3}>{l4}"})

    report = pd.DataFrame(report_rows)
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--out', default='data_labeled_reconciled.csv')
    ap.add_argument('--report', default='reconcile_report.csv')
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name='DATA', header=2)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=['ISI ULASAN'])
    df['LAYER 1'] = df['LAYER 1'].map(clean_label)
    df['LAYER 2'] = df['LAYER 2'].map(clean_label)

    out, report = reconcile_dataframe(df)
    out.to_csv(args.out, index=False)
    report.to_csv(args.report, index=False)

    print(f"\n{args.out}: {len(out)} baris tersimpan.")
    if len(report):
        print(f"{args.report}: {len(report)} label tidak exact-match, breakdown metode:")
        print(report['method'].value_counts())
        needs_review = report[report['method'] == 'NEEDS_MANUAL_REVIEW']
        if len(needs_review):
            print(f"\n⚠️  {len(needs_review)} baris butuh review manual (tidak ada opsi hierarki valid "
                  f"utk path tsb, atau LLM tidak tersedia). Cek {args.report}.")
    else:
        print("Semua label exact-match ke hierarki. Tidak ada yang perlu direkonsiliasi.")


if __name__ == '__main__':
    main()
