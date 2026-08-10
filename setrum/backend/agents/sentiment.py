"""
Agent 2 — Sentiment (GERBANG/FILTER)
====================================
Desain: sentimen sebagai GERBANG — hanya ulasan NEGATIF yang diteruskan
ke klasifikasi keluhan L1-L5. Positif disaring dan berhenti di sini.

Override logic:
- Netral → Negatif (konteks PLN didominasi keluhan)
- Negatif + kata positif sangat kuat + TANPA keluhan eksplisit → Positif
  (mencegah ulasan pujian panjang yang mengandung kata domain PLN seperti
  "gangguan", "pemadaman", "kendala" salah masuk sebagai keluhan)

Catatan: kasus "Rating 1 tapi teks positif" SENGAJA dibiarkan masuk pipeline
sebagai Negatif, karena itu akan diklasifikasi ke L3 = "Salah Mengartikan
Bintang 1" — kelas yang valid di hierarki PLN.

Model: mdhugol/indonesia-bert-sentiment-classification (lokal, bukan API).
Fallback: lexicon rule-based bila model tidak tersedia.
"""
import os, re

_LOCAL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'sentiment')
_HF_NAME   = 'mdhugol/indonesia-bert-sentiment-classification'
LABEL_MAP  = {'LABEL_0': 'Positif', 'LABEL_1': 'Netral', 'LABEL_2': 'Negatif'}

_pipe       = None
_load_tried = False


def contains_any(text: str, phrases) -> bool:
    """Cek apakah salah satu frasa muncul sebagai KATA/FRASA UTUH di teks,
    bukan cuma substring di tengah kata lain.
    Contoh bug yang dicegah: 'ga bisa' TIDAK boleh match di dalam 'juga bisa'
    (karena 'juga bisa' mengandung substring 'ga bisa' secara kebetulan)."""
    return any(re.search(r'\b' + re.escape(p) + r'\b', text) for p in phrases)

# ── Override: kata positif sangat kuat ───────────────────────────────────────
# Jika salah satu frasa ini muncul di teks DAN tidak ada keluhan eksplisit,
# sentimen dipaksa Positif meskipun model bilang Negatif.
STRONG_POS = [
    'sangat membantu', 'sangat bagus', 'sangat baik', 'sangat mudah',
    'sangat cepat', 'sangat puas', 'sangat senang', 'sangat memuaskan',
    'sangat canggih', 'sangat lengkap', 'sangat praktis', 'sangat mempermudah',
    'sangat terbantu', 'terbantu sekali', 'membantu sekali', 'bagus sekali',
    'baik sekali', 'mudah sekali', 'cepat sekali',
    'luar biasa', 'terbaik', 'sempurna', 'top banget', 'bagus banget',
    'keren banget', 'mantap banget', 'membantu banget', 'mudah banget',
    'cepat banget', 'mantap', 'mantaap', 'mantaaap', 'mantaaaap', 'mantaaaap',
    'keren', 'mantul', 'kece', 'josss', 'josss banget',
    'aplikasi terbaik', 'aplikasi bagus', 'aplikasi keren', 'aplikasi mantap',
    'terima kasih pln', 'makasih pln', 'thanks pln', 'terimakasih pln',
    'puas dengan', 'puas banget', 'sangat puas',
    'recommend', 'direkomendasikan', 'rekomendasikan',
    'memudahkan', 'mempermudah', 'mempercepat',
]

# ── Override: keluhan eksplisit ───────────────────────────────────────────────
# Jika salah satu frasa ini ada, override positif TIDAK aktif —
# biarkan model memutuskan.
EXPLICIT_NEG = [
    'tidak bisa', 'tidak mau', 'tidak berfungsi', 'tidak bekerja',
    'tidak jalan', 'tidak berhasil', 'tidak masuk', 'tidak nyala',
    'tidak keluar', 'tidak muncul', 'tidak bisa dibuka', 'tidak bisa login',
    'tidak bisa bayar', 'tidak bisa transfer', 'tidak bisa top up',
    'ga bisa', 'gak bisa', 'ngga bisa',
    'error', 'eror', 'gagal', 'crash', 'hang', 'force close',
    'lemot banget', 'lambat banget', 'loading terus', 'loading lama',
    # NOTE: 'padam', 'mati listrik', 'listrik mati' SENGAJA dihapus dari
    # daftar ini (per analisis Juli 2026) — kata-kata ini netral secara
    # domain PLN (dipakai baik di keluhan MAUPUN pujian soal kemudahan
    # lapor pemadaman), jadi bikin banyak false-positive kalau dianggap
    # otomatis sebagai keluhan eksplisit. Byar pet tetap dipertahankan
    # karena secara konotasi jelas negatif (listrik naik-turun berulang).
    'byar pet', 'byarpet',
    'kecewa', 'mengecewakan', 'sangat kecewa', 'sangat mengecewakan',
    'tolong perbaiki', 'mohon diperbaiki', 'harap diperbaiki', 'segera perbaiki',
    'tolong diperbaiki', 'minta diperbaiki',
    'buruk', 'jelek', 'parah', 'sampah', 'zonk', 'bohong', 'penipuan',
    'tidak berguna', 'tidak ada gunanya', 'tidak membantu',
    'susah banget', 'sulit banget', 'ribet banget', 'bermasalah terus',
    'kok tidak', 'kok gak', 'kok ga', 'masa iya', 'kenapa harus',
    'kapan diperbaiki', 'kapan beres', 'sudah lapor tapi',
    'otp tidak', 'otp ga', 'token tidak masuk', 'token ga masuk',
]


def _load_pipeline():
    """Muat model sentimen lokal sekali. Return None → fallback lexicon."""
    global _pipe, _load_tried
    if _pipe is not None:
        return _pipe
    if _load_tried:
        return None
    _load_tried = True
    try:
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification,
                                  TextClassificationPipeline)
        import torch
        src  = _LOCAL_DIR if os.path.isdir(_LOCAL_DIR) else _HF_NAME
        tok  = AutoTokenizer.from_pretrained(src)
        mdl  = AutoModelForSequenceClassification.from_pretrained(src)
        mdl.eval()
        _pipe = TextClassificationPipeline(
            model=mdl, tokenizer=tok,
            top_k=None, device=-1, truncation=True, max_length=256)
        print(f"[Sentiment] Model dimuat dari: {src}")
        return _pipe
    except Exception as e:
        print(f"[Sentiment] Gagal muat model ({e}). Pakai fallback lexicon.")
        return None


# ── Lexicon fallback ──────────────────────────────────────────────────────────
NEG = {
    'error':3,'eror':3,'gagal':3,'tidak bisa':3,'tidak masuk':3,'crash':3,
    'mati':3,'padam':3,'rusak':3,'bermasalah':3,'penipuan':3,'tipu':3,
    'kecewa':2,'buruk':2,'jelek':2,'parah':2,'lambat':2,'lemot':2,'susah':2,
    'sulit':2,'ribet':2,'mahal':2,'gangguan':2,'menyebalkan':2,
    'lama':1,'kurang':1,'belum':1,
}
POS = {
    'sangat bagus':3,'sangat membantu':3,'luar biasa':3,'terbaik':3,'sempurna':3,
    'bagus':2,'baik':2,'mantap':2,'mudah':2,'cepat':2,'lancar':2,'membantu':2,
    'puas':2,'terima kasih':2,'makasih':2,'keren':2,'mantaap':2,'recommend':2,
    'oke':1,'lumayan':1,'berhasil':1,
}
NEGASI = ['tidak','ga','gak','ngga','nggak','bukan','jangan','belum']


def _fallback(text: str) -> dict:
    words = text.lower().split()
    low   = text.lower()
    ns = ps = 0
    for ph, w in {**NEG, **POS}.items():
        if ' ' in ph and ph in low:
            if ph in NEG: ns += w
            else:         ps += w
    for i, wd in enumerate(words):
        neg = any(words[j] in NEGASI for j in range(max(0, i-2), i))
        if wd in NEG and ' ' not in wd:
            if neg: ps += NEG[wd] * 0.5
            else:   ns += NEG[wd]
        elif wd in POS and ' ' not in wd:
            if neg: ns += POS[wd] * 0.5
            else:   ps += POS[wd]
    tot = ns + ps
    if tot == 0:
        return {'sentiment': 'Negatif', 'confidence': 0.60, 'source': 'fallback'}
    if ns > ps:
        return {'sentiment': 'Negatif',
                'confidence': min(round(0.55 + ns/tot*0.38, 3), 0.92),
                'source': 'fallback'}
    if ps > ns:
        return {'sentiment': 'Positif',
                'confidence': min(round(0.55 + ps/tot*0.38, 3), 0.92),
                'source': 'fallback'}
    return {'sentiment': 'Negatif', 'confidence': 0.62, 'source': 'fallback'}


def _apply_override(sent: str, text: str) -> str:
    """
    Terapkan override kata kunci setelah model berjalan:
    1. Netral → Negatif (selalu)
    2. Negatif + STRONG_POS + tanpa EXPLICIT_NEG → Positif
       (ulasan pujian panjang yang kebetulan sebut kata domain PLN)

    TIDAK mengubah kasus rating 1 + teks positif — itu sengaja
    dibiarkan Negatif supaya bisa masuk L3 "Salah Mengartikan Bintang 1".
    Override hanya aktif kalau memang teksnya murni positif.
    """
    low = text.lower()

    # Rule 1: Netral → Negatif
    if sent == 'Netral':
        return 'Negatif'

    # Rule 2: Negatif → Positif jika teks murni positif
    if sent == 'Negatif':
        has_strong_pos   = contains_any(low, STRONG_POS)
        has_explicit_neg = contains_any(low, EXPLICIT_NEG)
        if has_strong_pos and not has_explicit_neg:
            return 'Positif'

    return sent


def analyze_sentiment(text: str) -> dict:
    """Return {'sentiment', 'confidence', 'source'}."""
    pipe = _load_pipeline()

    if pipe is None:
        result = _fallback(text)
        result['sentiment'] = _apply_override(result['sentiment'], text)
        return result

    try:
        scores = pipe(text[:256])[0]
        best   = max(scores, key=lambda x: x['score'])
        sent   = LABEL_MAP.get(best['label'], best['label'])
        sent   = _apply_override(sent, text)
        return {
            'sentiment':  sent,
            'confidence': round(float(best['score']), 4),
            'source':     'IndoBERT-local',
        }
    except Exception as e:
        print(f"[Sentiment] Error prediksi: {e} -> fallback")
        result = _fallback(text)
        result['sentiment'] = _apply_override(result['sentiment'], text)
        return result


def is_negative(text: str, threshold: float = 0.50):
    """Gerbang: True bila bukan Positif (Negatif masuk klasifikasi)."""
    r = analyze_sentiment(text)
    return (r['sentiment'] != 'Positif'), r


if __name__ == '__main__':
    tests = [
        # Harus POSITIF (teks murni pujian)
        ("aplikasi sangat membantu bisa mengetahui info tagihan listrik pemadaman pemasangan baru mantaaap", "POSITIF"),
        ("bagus sekali aplikasinya mudah digunakan terima kasih PLN",                                        "POSITIF"),
        ("mantap aplikasinya sangat membantu sekali pln mobile terbaik",                                     "POSITIF"),
        ("luar biasa aplikasinya recommend banget",                                                          "POSITIF"),
        # Harus NEGATIF (ada keluhan eksplisit)
        ("aplikasi error terus tidak bisa beli token sudah 3 hari",                                          "NEGATIF"),
        ("kecewa banget aplikasi sering crash dan gagal bayar",                                              "NEGATIF"),
        ("listrik mati 2 hari tidak ada pemberitahuan sama sekali",                                          "NEGATIF"),
        ("sangat membantu tapi sering error dan tidak bisa login",                                           "NEGATIF"),
        ("otp tidak masuk email sudah dicoba berkali kali",                                                  "NEGATIF"),
        # Kasus khusus: rating 1 tapi teks positif → NEGATIF (masuk Salah Mengartikan Bintang 1)
        ("aplikasi bagus sekali sangat membantu bayar listrik jadi mudah",                                   "NEGATIF — tergantung model, kelas Salah Mengartikan Bintang 1"),
    ]
    print("=" * 70)
    for text, expected in tests:
        result = analyze_sentiment(text)
        gerbang, _ = is_negative(text)
        gate = "MASUK klasifikasi" if gerbang else "DISARING"
        icon = "✅" if result['sentiment'] in expected else "⚠️"
        print(f"{icon} [{result['sentiment']} {result['confidence']:.0%}] {gate}")
        print(f"   Expected : {expected}")
        print(f"   Text     : \"{text[:75]}{'...' if len(text)>75 else ''}\"")
        print()