"""
evaluate.py — Pengujian sistem terhadap ground truth label PLN
==============================================================
Membandingkan prediksi sistem SETRUM dengan label manual tim PLN (data uji),
menghitung Accuracy, Macro-F1, Weighted-F1, Cohen's Kappa, dan confusion matrix.

Ini menghasilkan ANGKA NYATA untuk Bab IV — bukan ilustrasi.

Cara pakai:
    python evaluate.py --test data_layer1.csv --target layer1
    python evaluate.py --test data_layer3.csv --target layer3
    python evaluate.py --test data_sentiment_test.csv --target sentiment
"""
import argparse, sys, os, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             classification_report, confusion_matrix)
from backend.agents.normalizer import normalize
from backend.agents.sentiment import analyze_sentiment
from backend.agents.classifier import predict_l1_l3


def kappa_label(k):
    if k < 0.21: return 'Buruk'
    if k < 0.41: return 'Cukup'
    if k < 0.61: return 'Sedang'
    if k < 0.81: return 'Baik'
    return 'Sangat Baik'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True, help='CSV kolom text,label (data uji)')
    ap.add_argument('--target', required=True, choices=['sentiment','layer1','layer2','layer3'])
    args = ap.parse_args()

    df = pd.read_csv(args.test).dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        norm = normalize(row['text'])
        if args.target == 'sentiment':
            pred = analyze_sentiment(norm)['sentiment']
        else:
            pred = predict_l1_l3(norm).get(args.target, '')
        y_true.append(str(row['label'])); y_pred.append(str(pred))

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    wf1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kap = cohen_kappa_score(y_true, y_pred)

    print(f"\n=== EVALUASI {args.target.upper()} (n={len(df)}) ===")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro-F1    : {mf1:.4f}")
    print(f"  Weighted-F1 : {wf1:.4f}")
    print(f"  Cohen Kappa : {kap:.4f}  ({kappa_label(kap)})")
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, zero_division=0))

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("--- Confusion Matrix (baris=aktual, kolom=prediksi) ---")
    print("labels:", labels)
    print(cm)

    # simpan untuk dipakai di Bab IV / dashboard
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(f'cm_{args.target}.csv')
    print(f"\nConfusion matrix disimpan: cm_{args.target}.csv")


if __name__ == '__main__':
    main()
