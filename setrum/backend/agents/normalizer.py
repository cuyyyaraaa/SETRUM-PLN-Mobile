"""
Agent 1 — Smart Text Normalization (v3)
=======================================
Pipeline normalisasi teks ulasan PLN Mobile:
  1. Emoji → kata (sinyal sentimen)
  2. Cleaning: URL, karakter noise
  3. Fix pengulangan huruf & reduplikasi
  4. Kamus slang (dari slang_dict.json — load sekali)
  5. Sastrawi stemmer — opsional, hanya untuk kata yang tidak dikenal
  6. Restore akronim domain PLN

Sastrawi di-load LAZY (sekali saja saat pertama dipanggil) → tidak berat.
slang_dict.json di-load dari folder yang sama dengan file ini,
fallback ke kamus bawaan bila file tidak ditemukan.
"""
import re, os, json

# ── Lazy-load Sastrawi ────────────────────────────────────────────────────────
_stemmer = None

def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            _stemmer = StemmerFactory().create_stemmer()
            print("[Normalizer] Sastrawi stemmer loaded.")
        except ImportError:
            _stemmer = False
            print("[Normalizer] Sastrawi not installed — stemming disabled.")
    return _stemmer if _stemmer else None


# ── Load kamus slang dari JSON (fallback ke dict bawaan) ─────────────────────
def _load_slang() -> dict:
    candidates = [
        os.path.join(os.path.dirname(__file__), 'slang_dict.json'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'slang_dict.json'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slang_dict.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                slang = json.load(f)
            print(f"[Normalizer] Slang dict loaded: {len(slang)} entries from {path}")
            return slang
    # fallback minimal
    print("[Normalizer] slang_dict.json not found — using built-in minimal dict.")
    return {
        'gak':'tidak','ga':'tidak','ngga':'tidak','nggak':'tidak','tdk':'tidak',
        'udah':'sudah','udh':'sudah','blm':'belum','emang':'memang',
        'bgt':'banget','yg':'yang','dgn':'dengan','utk':'untuk',
        'krn':'karena','klo':'kalau','kalo':'kalau','tp':'tapi',
        'bs':'bisa','hrs':'harus','msh':'masih','sgr':'segera',
        'tlg':'tolong','eror':'error','apk':'aplikasi','trs':'terus',
        'mulu':'terus','lemot':'lambat','notif':'notifikasi',
        'makasih':'terima kasih','thx':'terima kasih',
    }

_SLANG: dict = None

def _get_slang() -> dict:
    global _SLANG
    if _SLANG is None:
        _SLANG = _load_slang()
    return _SLANG


# ── Konstanta ─────────────────────────────────────────────────────────────────
EMOJI_MAP = {
    '😡': ' marah ', '😠': ' marah ', '🤬': ' marah ', '😤': ' kesal ',
    '😭': ' kecewa ', '😢': ' sedih ', '😞': ' kecewa ', '😔': ' kecewa ',
    '😩': ' frustrasi ', '🤯': ' frustrasi ', '👎': ' buruk ',
    '❌': ' gagal ', '⚠️': ' peringatan ',
    '😊': ' senang ', '👍': ' bagus ', '🙏': ' mohon ',
    '😍': ' suka ', '⭐': ' bintang ', '🔥': ' panas ',
}

PRESERVE_UPPER = {
    'pln','otp','idpel','spklu','splu','spbklu','kwh','kva',
    'lsp','slo','nidi','p2tl','pb','pd','ps','ev','plts',
    'bpjs','qris','cc','faq','url','ui','ux','api','sms',
}

# Kata-kata yang TIDAK boleh di-stem karena bermakna untuk klasifikasi
NO_STEM = {
    'error','loading','token','tagihan','bayar','listrik','padam','gangguan',
    'pengaduan','aplikasi','pembayaran','registrasi','verifikasi','notifikasi',
    'password','akun','login','install','update','server','database',
    'tidak','sudah','belum','karena','kalau','tapi','saja','tolong',
    'segera','sangat','banget','masih','terus','masuk','keluar',
    'perbaiki','diperbaiki','perbarui','diperbarui',
}

PURE_FILLERS = {
    'wkwk','wkwkwk','wkwkwkwk','haha','hehe','hihi','huhu',
    'xixi','hahaha','hehehehe','hmm','mmm','umm',
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _replace_emoji(text: str) -> str:
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, word)
    return text


def _fix_repeat(text: str) -> str:
    """bagusss → bagus (potong ke 1 karakter, bukan 2)."""
    return re.sub(r'(.)\1{2,}', r'\1', text)


def _handle_reduplication(text: str) -> str:
    """brkali2 → berkali-kali, cepet2 → cepat-cepat."""
    text = re.sub(r'\b(berkali|brkali)2\b', 'berkali-kali', text)
    text = re.sub(r'\b(\w{3,})2\b', r'\1-\1', text)
    return text


def _separate_number_word(text: str) -> str:
    """3hari → 3 hari, 2jam → 2 jam."""
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    return text


def _fix_punctuation(text: str) -> str:
    text = re.sub(r'([!?])\1+', r'\1', text)
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'\s([.,!?])', r'\1', text)
    return text


def _apply_slang(words: list, slang: dict) -> list:
    """Replace slang, bigram dulu baru unigram, buang pure filler."""
    out = []
    i = 0
    while i < len(words):
        # Coba bigram
        if i < len(words) - 1:
            bigram = re.sub(r'[^\w\s]', '', words[i] + ' ' + words[i+1]).strip().lower()
            if bigram in slang:
                rep = slang[bigram]
                if rep: out.extend(rep.split())
                i += 2
                continue
        # Unigram
        w = words[i]
        cw = re.sub(r'[^\w]', '', w).lower()
        if cw in PURE_FILLERS:
            i += 1
            continue
        if cw in slang:
            rep = slang[cw]
            if rep: out.extend(rep.split())
        else:
            out.append(w)
        i += 1
    return out


def _selective_stem(words: list) -> list:
    """
    Stem hanya kata yang:
    - Bukan dalam NO_STEM list
    - Bukan akronim (tidak semua huruf kapital)
    - Panjang > 4 karakter
    - Bukan angka
    Ini mencegah over-stemming yang merusak makna.
    """
    stemmer = _get_stemmer()
    if not stemmer:
        return words
    out = []
    for w in words:
        cw = re.sub(r'[^\w]', '', w).lower()
        # Skip kondisi
        if (not cw or
            cw in NO_STEM or
            cw in PRESERVE_UPPER or
            cw.isdigit() or
            len(cw) <= 4):
            out.append(w)
            continue
        stemmed = stemmer.stem(cw)
        # Hanya pakai hasil stem kalau lebih pendek dan bukan string kosong
        if stemmed and len(stemmed) < len(cw) and len(stemmed) >= 3:
            out.append(stemmed)
        else:
            out.append(w)
    return out


def _restore_upper(words: list) -> list:
    """Kembalikan akronim domain ke huruf kapital."""
    out = []
    for w in words:
        cw = re.sub(r'[^\w]', '', w).lower()
        if cw in PRESERVE_UPPER:
            out.append(cw.upper())
        else:
            out.append(w)
    return out


# ── Main normalize ────────────────────────────────────────────────────────────
def normalize(text: str, use_stemmer: bool = True) -> str:
    """
    Normalisasi teks ulasan PLN Mobile.
    
    Args:
        text: teks ulasan asli
        use_stemmer: aktifkan Sastrawi (default True)
    
    Returns:
        teks yang sudah dinormalisasi
    """
    if not text or not text.strip():
        return ''

    slang = _get_slang()

    # 1. Emoji → kata
    result = _replace_emoji(text)

    # 2. Buang URL
    result = re.sub(r'https?://\S+|www\.\S+', '', result)

    # 3. Lowercase sementara
    result = result.lower().strip()

    # 4. Handle reduplikasi sebelum pisah angka
    result = _handle_reduplication(result)

    # 5. Pisahkan angka-kata
    result = _separate_number_word(result)

    # 6. Fix pengulangan huruf: bagusss → bagus
    result = _fix_repeat(result)

    # 7. Buang karakter noise (kecuali alfanumerik & tanda baca dasar)
    result = re.sub(r'[^\w\s.,!?\-/]', ' ', result, flags=re.UNICODE)

    # 8. Fix tanda baca
    result = _fix_punctuation(result)

    # 9. Normalisasi spasi
    result = re.sub(r'\s+', ' ', result).strip()

    # 10. Ganti slang (kamus JSON)
    words = result.split()
    words = _apply_slang(words, slang)

    # 11. Selective stemming (Sastrawi)
    if use_stemmer:
        words = _selective_stem(words)

    # 12. Restore akronim penting
    words = _restore_upper(words)

    # 13. Gabung & kapitalisasi awal
    result = re.sub(r'\s+', ' ', ' '.join(words)).strip()
    if result:
        result = result[0].upper() + result[1:]

    return result


if __name__ == '__main__':
    tests = [
        ("Aplikasi eror mulu gak bisa beli token udh 3hari!!! tlg diperbaiki donk 😡",
         "→ error, token, 3 hari, emoji, slang"),
        ("loadinggg mulu sm sekali ga bisa kebuka wkwkwk",
         "→ repeat char, slang, filler"),
        ("otp tdk msk ke email udh dicoba brkali2 tp tetep ga bisa login",
         "→ OTP kapital, berkali-kali"),
        ("pln kok mati mulu sih pdhl udh bayar listrik kmrn, sgr perbaiki!!!",
         "→ PLN kapital, partikel"),
        ("tghn bulan ini bnyk bgt pdhl pemakaian sm aja, knp ya???",
         "→ tagihan, banyak, banget, kenapa"),
        ("aplikas ga bisa dibuka dr tadi, lemot bgt, tolong update dong",
         "→ typo aplikas, lambat, perbarui"),
        ("makasih pln mobile sangad membantu byr tagihan jd gampang bgt",
         "→ sentimen positif"),
        ("eror mulu apk ny, msh loading trs gak ada notif sama sekali 😤",
         "→ apk→aplikasi, notif→notifikasi, emoji"),
        ("pembayaran gagal tapi saldo sdh terpotong, mhn segera dikembalikan",
         "→ domain PLN, stemming"),
        ("listrik mati 2hari gak ada pemberitahuan dr pln sama sekali",
         "→ 2hari pisah, PLN kapital"),
    ]
    print("=" * 70)
    for text, note in tests:
        out = normalize(text)
        print(f"NOTE: {note}")
        print(f"IN : {text}")
        print(f"OUT: {out}")
        print()