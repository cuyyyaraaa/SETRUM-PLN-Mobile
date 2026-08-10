"""
Orchestrator — koordinasi pipeline HMAS (sentimen = gerbang)

Alur:
  Ulasan + Rating
    | (1) Normalisasi
    | (2) Sentimen (IndoBERT + override kata kunci)
    |      → Positif : SELESAI
    |      → Negatif : lanjut
    |        ↓
    | (2b) RATING RESCUE — rating tinggi (4-5) tapi model bilang Negatif
    |      TANPA keluhan eksplisit sama sekali
    |      → dianggap salah baca model (bias kata domain: "gangguan",
    |        "pengaduan", "tagihan", dst yang muncul di ulasan positif)
    |      → dipaksa jadi Positif, SELESAI
    |        ↓
    | (3) Klasifikasi L1-L3 (IndoBERT fine-tuned)
    |        ↓
    | (3b) RATING OVERRIDE — setelah dapat sentimen & L3:
    |      Rating 1-2 + sentimen Positif (model ragu)
    |      → L1=UMUM, L2=UMUM, L3="Salah Mengartikan Bintang 1"
    |      Rating 1-2 + sentimen Negatif → pipeline normal
    |      Rating 3 → tidak ada override
    |        ↓
    | (4) Eskalasi L4-L5 (BART-MNLI zero-shot)
    |      hanya bila L1 != UMUM
    v
  Simpan → Dashboard

Peran Rating:
- Rating BUKAN gerbang utama (sentiment tetap yang memutuskan) — TAPI
  dipakai sebagai sinyal koreksi di 2 ujung ekstrem (rating 1-2 dan 4-5),
  karena rating adalah data yang diberikan pengguna sendiri sehingga lebih
  bisa dipercaya dibanding model sentimen generic yang tidak dilatih
  khusus untuk domain kelistrikan PLN.
- Rating 1-2 dipakai untuk mendeteksi "Salah Mengartikan Bintang 1":
  ulasan yang diberi bintang rendah tapi isi teksnya positif/pujian.
  Ini kelas khusus di hierarki PLN untuk menandai inkonsistensi
  antara rating dan konten ulasan.
- Rating 4-5 dipakai untuk "rating rescue": ulasan dengan rating tinggi
  yang oleh model sentimen (generic, bukan fine-tuned domain PLN)
  salah dibaca sebagai Negatif hanya karena menyebut istilah domain
  ("gangguan", "pengaduan", "tagihan", dst) — TANPA ada keluhan
  eksplisit apapun di teksnya. Ditemukan lewat analisis manual: dari
  275 ulasan rating-5 yang salah diklasifikasi Negatif, 98% TIDAK
  mengandung kata keluhan eksplisit sama sekali.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.agents.normalizer import normalize
from backend.agents.sentiment  import analyze_sentiment, STRONG_POS, EXPLICIT_NEG, contains_any
from backend.agents.classifier import predict_l1_l3
from backend.agents.escalator  import classify_l4_l5


def _is_positive_text(text: str) -> bool:
    """Cek apakah teks mengandung kata positif kuat tanpa keluhan eksplisit."""
    low = text.lower()
    has_strong_pos   = contains_any(low, STRONG_POS)
    has_explicit_neg = contains_any(low, EXPLICIT_NEG)
    return has_strong_pos and not has_explicit_neg


def analyze_review(text: str, rating: int = None, verbose=False) -> dict:
    """Jalankan pipeline HMAS lengkap. Return dict hasil analisis."""

    # (1) Normalisasi
    norm = normalize(text)
    if verbose:
        print(f"\n=== {text[:60]}")
        print(f"[1] norm: {norm[:60]}")

    result = {
        'normalized':                norm,
        'sentiment':                 '',
        'sentiment_confidence':      0.0,
        'sentiment_source':          '',
        'category':                  '',
        'subcategory':               '',
        'classification_confidence': 0.0,
        'layer1': '', 'layer2': '', 'layer3': '',
        'layer4': '', 'layer5': '',
        'quality_param': '',
        'escalated':    False,
        'is_complaint': False,
        'confidence':   0.0,
        'reason':       '',
    }

    # (2) Sentimen — gerbang utama
    sent      = analyze_sentiment(norm)
    sentiment = sent['sentiment']
    sconf     = sent['confidence']
    if verbose:
        print(f"[2] sentimen: {sentiment} ({sconf*100:.0f}%) [{sent['source']}]")

    result.update({
        'sentiment':            sentiment,
        'sentiment_confidence': round(sconf, 3),
        'sentiment_source':     sent['source'],
        'confidence':           round(sconf, 3),
    })

    # Hanya NEGATIF yang lanjut ke klasifikasi
    if sentiment != 'Negatif':
        result['reason'] = f"{sentiment} review — not classified as a complaint."
        if verbose: print("    -> bukan negatif, selesai.")
        return result

    # (2b) RATING RESCUE — rating tinggi (4-5) tapi model bilang Negatif
    # TANPA keluhan eksplisit sama sekali di teksnya. Ditemukan lewat
    # analisis manual bahwa model generic sering salah baca istilah domain
    # PLN ("gangguan", "pengaduan", "tagihan", dst) sebagai sinyal negatif,
    # padahal ulasan tersebut sebenarnya memuji aplikasi.
    if rating is not None and int(rating) >= 4:
        has_explicit_neg = contains_any(norm.lower(), EXPLICIT_NEG)
        if not has_explicit_neg:
            result.update({'sentiment': 'Positif',
                           'sentiment_source': sent['source'] + '+rating_rescue'})
            result['reason'] = (
                f"Rating {rating}★ tanpa keluhan eksplisit — model sentimen awal "
                f"salah baca istilah domain PLN, dikoreksi jadi Positif."
            )
            if verbose:
                print(f"[2b] Rating rescue: {rating}★ + tanpa keluhan eksplisit -> Positif")
            return result

    result['is_complaint'] = True

    # (3) Klasifikasi L1-L3
    cls   = predict_l1_l3(norm)
    l1    = cls['layer1']
    l2    = cls['layer2']
    l3    = cls['layer3']
    cconf = cls.get('layer3_conf', cls.get('confidence', 0.5))

    result.update({
        'layer1':                    l1,
        'layer2':                    l2,
        'layer3':                    l3,
        'category':                  l1,
        'subcategory':               l3,
        'classification_confidence': round(cconf, 3),
    })
    if verbose:
        print(f"[3] L1-L3: {l1} > {l2} > {l3} ({cconf*100:.0f}%) [{cls['source']}]")

    # (3b) RATING OVERRIDE — deteksi "Salah Mengartikan Bintang 1"
    # Terjadi bila: rating rendah (1-2) DAN teks ulasan sebenarnya positif
    # Ini menangkap inkonsistensi rating vs konten yang disebut di paper Asri et al.
    if rating is not None and int(rating) <= 2 and _is_positive_text(norm):
        result.update({
            'layer1':                    'UMUM',
            'layer2':                    'UMUM',
            'layer3':                    'Salah Mengartikan Bintang 1',
            'layer4':                    '',
            'layer5':                    '',
            'category':                  'UMUM',
            'subcategory':               'Salah Mengartikan Bintang 1',
            'classification_confidence': 0.85,
            'quality_param':             '',
            'escalated':                 False,
        })
        result['reason'] = (
            f"Rating {rating}★ dengan teks positif — terdeteksi sebagai "
            f"'Salah Mengartikan Bintang 1' (inkonsistensi rating vs konten)."
        )
        result['confidence'] = round(min(sconf, 0.85), 3)
        if verbose:
            print(f"[3b] Rating override: {rating}★ + teks positif -> UMUM > Salah Mengartikan Bintang 1")
        return result

    # (4) Eskalasi L4-L5 — hanya bila L1 bukan UMUM dan teks cukup panjang
    if l1 and 'UMUM' not in l1.upper() and len(norm.split()) >= 3:
        esc = classify_l4_l5(norm, l1, l2, l3, verbose=verbose)
        result.update({
            'layer4':        esc['layer4'],
            'layer5':        esc['layer5'],
            'quality_param': esc['param'],
            'escalated':     True,
        })

    result['reason'] = f"Negative complaint: {l1} > {l3}" + (
        f" > {result['layer4']}" if result['layer4'] else "")
    result['confidence'] = round(min(sconf, cconf), 3)
    return result


if __name__ == '__main__':
    tests = [
        # Rating rendah + teks positif → Salah Mengartikan Bintang 1
        ("Aplikasi sangat membantu mantap sekali terima kasih PLN", 1),
        ("bagus banget aplikasinya mudah digunakan pln mobile terbaik", 2),
        # Rating rendah + teks negatif → pipeline normal
        ("Aplikasi error terus tidak bisa beli token sudah 3 hari", 1),
        ("listrik mati 2 hari tidak ada pemberitahuan", 2),
        # Rating tinggi + teks negatif → pipeline normal
        ("token tidak masuk sudah bayar tapi gagal terus", 4),
        # Rating tinggi + model salah baca istilah domain (TANPA keluhan asli)
        # → HARUS dikoreksi jadi Positif oleh rating rescue
        ("pengaduan gangguan listrik melalui PLN mobile", 5),
        ("pln mobile membantu", 5),
        ("lapor lewat aplikasi gampang dan cepat", 5),
        # Rating tinggi TAPI ada keluhan eksplisit asli → tetap Negatif
        ("awalnya bintang 5 tapi sekarang gak bisa masuk ke aplikasi sama sekali", 5),
        # Tanpa rating → pipeline normal (tidak ada rescue apapun)
        ("aplikasi sering crash dan tidak bisa login", None),
    ]
    print("=" * 70)
    for text, rating in tests:
        out = analyze_review(text, rating=rating, verbose=True)
        print(f"  HASIL  : {out['sentiment']} | {out['layer1']} > {out['layer3']}")
        print(f"  REASON : {out['reason']}\n")