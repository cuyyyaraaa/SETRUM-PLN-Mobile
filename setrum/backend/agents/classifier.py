"""
Agent 3 — Hierarchical Classifier Layer 1-3
===========================================
Mengklasifikasikan keluhan negatif ke Layer 1, 2, 3 PLN Mobile.

Prioritas:
  1. Model IndoBERT fine-tuned (models/layer1, models/layer2, models/layer3)
     -> dijalankan LOKAL, akurat. Ini yang dipakai di skripsi.
  2. Bila model belum ada -> keyword classifier (agar sistem tetap jalan saat
     development sebelum training selesai).

CATATAN SKRIPSI: hasil akhir HARUS pakai model fine-tuned. Keyword hanya
jaring pengaman pengembangan; jangan dilaporkan sebagai "IndoBERT".
"""
import os, re

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
_models, _tried = {}, {}


def _load(layer: str):
    if layer in _models:
        return _models[layer]
    if _tried.get(layer):
        return None
    _tried[layer] = True
    path = os.path.join(_MODELS_DIR, layer)
    if not os.path.isdir(path):
        print(f"[Clf] model {layer} belum ada -> keyword fallback")
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path)
        mdl.eval()
        _models[layer] = {'tok': tok, 'mdl': mdl, 'torch': torch}
        print(f"[Clf] model {layer} dimuat (fine-tuned)")
        return _models[layer]
    except Exception as e:
        print(f"[Clf] gagal muat {layer}: {e} -> keyword")
        return None


def predict_layer(text: str, layer: str):
    m = _load(layer)
    if m is None:
        return None, 0.0
    try:
        torch = m['torch']
        inp = m['tok'](text, return_tensors='pt', truncation=True,
                       max_length=128, padding=True)
        with torch.no_grad():
            logits = m['mdl'](**inp).logits
            probs = torch.softmax(logits, dim=-1)
            conf, idx = torch.max(probs, dim=-1)
            label = m['mdl'].config.id2label[int(idx)]
        return label, float(conf)
    except Exception as e:
        print(f"[Clf] error {layer}: {e}")
        return None, 0.0


# ---- keyword fallback (development) -----------------------------------------
KW = {
    'Token & Pembayaran': ['token','bayar','pembayaran','tagihan','transaksi','saldo','top up','topup'],
    'Akun PLN Mobile': ['login','akun','registrasi','daftar','otp','password','verifikasi','email'],
    'Pengaduan': ['padam','mati','gangguan','listrik','byar pet','nyala','trip','meteran','meter'],
    'ICONNET': ['iconnet','internet','wifi','indihome'],
    'Catat Meter': ['stand meter','catat meter','angka meter'],
}
L1_KW = {
    'APLIKASI': ['aplikasi','app','login','error','crash','loading','token','bayar','akun'],
    'LAYANAN': ['padam','mati','gangguan','petugas','pengaduan','respon','cs','layanan'],
}


def classify(text: str) -> dict:
    """Keyword fallback klasifikasi (dipakai bila model belum ada)."""
    low = text.lower()
    # L3
    best_l3, best_score = 'Pengaduan', 0
    for cat, kws in KW.items():
        s = sum(1 for k in kws if k in low)
        if s > best_score:
            best_l3, best_score = cat, s
    # L1
    l1 = 'APLIKASI'
    a = sum(1 for k in L1_KW['APLIKASI'] if k in low)
    b = sum(1 for k in L1_KW['LAYANAN'] if k in low)
    if b > a: l1 = 'LAYANAN'
    conf = min(0.55 + best_score*0.1, 0.85) if best_score else 0.50
    return {'layer1': l1, 'layer2': 'KETENAGALISTRIKAN', 'layer3': best_l3,
            'confidence': conf, 'source': 'keyword'}


def predict_l1_l3(text: str) -> dict:
    """Pipeline L1-L3: pakai fine-tuned bila ada, else keyword."""
    out = classify(text)  # default keyword
    for layer in ['layer1', 'layer2', 'layer3']:
        lab, c = predict_layer(text, layer)
        if lab:
            out[layer] = lab
            out[f'{layer}_conf'] = c
            out['source'] = 'IndoBERT'
    return out


if __name__ == '__main__':
    for t in ["sudah bayar token tapi tidak masuk",
              "listrik padam dari kemarin belum nyala"]:
        print(t, '->', predict_l1_l3(t))
