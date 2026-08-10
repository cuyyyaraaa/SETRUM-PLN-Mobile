"""
Agent 4 — LLM Escalation (Layer 4-5)
====================================
L4-L5 datanya terlalu sedikit per kelas untuk fine-tuning, jadi pakai LLM
zero-shot. Pilihan kategori DIPERSEMPIT dari hierarki resmi (hanya L4 yang valid
untuk L3 tsb) -> akurasi & efisiensi naik.

Dua opsi backend LLM (pilih lewat .env: LLM_BACKEND):
  1. "local"  -> model zero-shot lokal (transformers pipeline zero-shot-
                 classification, mis. BART-MNLI atau model NLI Indonesia).
                 Tidak butuh internet saat inferensi.
  2. "ollama" -> server Ollama lokal (mis. llama3/qwen). Stabil, tanpa API key.

Jika keduanya gagal -> fallback: ambil L4/L5 pertama dari hierarki (heuristik).
Sistem TIDAK PERNAH crash karena eskalasi gagal.
"""
import os
from .hierarchy_loader import get_hierarchy

LLM_BACKEND = os.getenv('LLM_BACKEND', 'local')  # 'local' | 'ollama'
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
        model = os.getenv('ZS_MODEL', 'facebook/bart-large-mnli')
        _zs = pipeline('zero-shot-classification', model=model, device=-1)
        print(f"[Escalator] zero-shot lokal dimuat: {model}")
        return _zs
    except Exception as e:
        print(f"[Escalator] zero-shot lokal gagal ({e})")
        return None


def _ollama_pick(text, options, layer_name):
    """Tanya Ollama memilih satu opsi. Return string opsi atau None."""
    import json, urllib.request
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    model = os.getenv('OLLAMA_MODEL', 'llama3')
    opts = "\n".join(f"- {o}" for o in options)
    prompt = (f"Ulasan pelanggan PLN Mobile:\n\"{text}\"\n\n"
              f"Pilih SATU kategori {layer_name} yang paling sesuai dari daftar berikut. "
              f"Jawab HANYA dengan teks kategori persis seperti di daftar.\n{opts}")
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    try:
        req = urllib.request.Request(f"{host}/api/generate", data=body,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as r:
            ans = json.loads(r.read())['response'].strip()
        # cocokkan ke opsi terdekat
        for o in options:
            if o.lower() in ans.lower() or ans.lower() in o.lower():
                return o
        return options[0] if options else None
    except Exception as e:
        print(f"[Escalator] ollama gagal: {e}")
        return None


def _zeroshot_pick(text, options, _layer):
    zs = _load_zeroshot()
    if zs is None or not options:
        return None
    try:
        res = zs(text, options, multi_label=False)
        return res['labels'][0]
    except Exception as e:
        print(f"[Escalator] zero-shot error: {e}")
        return None


def classify_l4_l5(text, l1, l2, l3, verbose=False):
    """Klasifikasi L4 lalu L5 berdasar PATH LENGKAP l1->l2->l3.

    Path lengkap wajib (bukan cuma l3) karena beberapa label L3/L4 di
    hierarki resmi PLN muncul di lebih dari satu cabang (mis. L3="Pengaduan"
    ada di bawah LAYANAN maupun APLIKASI) -- tanpa l1/l2, opsi L4 yang
    ditawarkan ke LLM bisa gabungan dua cabang yang tidak valid.
    """
    h = get_hierarchy()
    l4_opts = h.get_l4_options(l1, l2, l3)
    if not l4_opts:
        return {'layer4': '', 'layer5': '', 'param': ''}

    pick = _ollama_pick if LLM_BACKEND == 'ollama' else _zeroshot_pick
    l4 = pick(text, l4_opts, 'Layer 4') or (l4_opts[0] if l4_opts else '')

    l5_opts = h.get_l5_options(l1, l2, l3, l4)
    l5 = pick(text, l5_opts, 'Layer 5') if l5_opts else ''
    l5 = l5 or (l5_opts[0] if l5_opts else '')
    param = h.get_parameter(l1, l2, l3, l4, l5)
    if verbose:
        print(f"[Escalator] {l1}>{l2}>{l3} -> L4={l4} -> L5={l5} ({param})")
    return {'layer4': l4, 'layer5': l5, 'param': param}
