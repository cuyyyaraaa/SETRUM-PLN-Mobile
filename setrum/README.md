# SETRUM — Hierarchical Multi-Agent System untuk Analisis Ulasan PLN Mobile

Sistem analisis ulasan pelanggan PLN Mobile berbasis multi-agent: normalisasi →
sentimen (gerbang) → klasifikasi keluhan hierarkis L1–L3 (IndoBERT) → eskalasi
L4–L5 (LLM zero-shot) → dashboard interaktif.

> **Perubahan desain penting (saran dosen):** PLN hanya mengeksekusi ulasan
> **negatif**, dan data validasi pun negatif. Maka **sentimen dijadikan
> gerbang**: hanya ulasan negatif yang diklasifikasikan ke keluhan L1–L5.
> Ulasan positif/netral disisihkan. Lihat bagian "Dampak ke Bab 3" di bawah.

---

## Struktur Proyek
```
setrum/
├── backend/
│   ├── app.py                 # server Flask + API dashboard
│   ├── database.py            # SQLite (reviews, analysis_results)
│   └── agents/
│       ├── normalizer.py      # Agent 1: normalisasi teks
│       ├── sentiment.py       # Agent 2: sentimen (GERBANG), model lokal
│       ├── classifier.py      # Agent 3: klasifikasi L1-L3 (IndoBERT fine-tuned)
│       ├── escalator.py       # Agent 4: eskalasi L4-L5 (LLM zero-shot)
│       ├── hierarchy_loader.py# pemuat hierarki resmi PLN
│       └── orchestrator.py    # koordinator pipeline
├── frontend/src/index.html    # dashboard interaktif (Chart.js)
├── training/
│   ├── prepare_data.py        # siapkan CSV latih dari Excel PLN
│   ├── train_indobert.py      # fine-tuning IndoBERT (class weighting)
│   └── evaluate.py            # uji vs ground truth -> Acc/F1/Kappa
├── data/
│   └── hierarchy_pln_clean.json  # hierarki 5-layer resmi PLN (sudah diekstrak)
├── models/                    # (kosong) tempat model fine-tuned hasil training
├── requirements.txt
├── .env.example
├── PANDUAN_HUGGINGFACE.md     # anti-korup model — WAJIB baca
├── PANDUAN_PENGUJIAN_UAT.md   # rencana pengujian + UAT di PLN
└── README.md
```

---

## A. Cara Menjalankan Dashboard (paling cepat)

### 1. Siapkan environment
```bash
cd setrum
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

### 2. Jalankan server
```bash
python backend/app.py
```
Buka browser ke **http://localhost:5000**

### 3. Pakai dashboard
- **Engine Control → Scraping**: tarik ulasan dari Play Store.
- **Engine Control → Analisis Batch**: jalankan pipeline pada ulasan.
- **Dashboard**: lihat grafik sentimen & keluhan.
- **Detail Hasil**: telusuri hasil per ulasan.
- **Analisis Manual**: uji satu teks ulasan.

> Tanpa model fine-tuned, sistem tetap jalan memakai fallback (keyword/lexicon)
> agar alur bisa didemokan. Untuk hasil skripsi, selesaikan training (bagian B).

---

## B. Cara Training IndoBERT (untuk hasil skripsi)

Dilakukan di **Google Colab (GPU)**. Ringkas:

### 1. Siapkan data latih dari Excel PLN
```bash
python training/prepare_data.py --xlsx "20260520_-_Report_Review_Negatif__OK_.xlsx"
# menghasilkan: data_layer1.csv, data_layer2.csv, data_layer3.csv
```

### 2. Fine-tuning tiap layer (di Colab)
```bash
pip install transformers datasets scikit-learn pandas torch
python training/train_indobert.py --data data_layer1.csv --out models/layer1 --epochs 4
python training/train_indobert.py --data data_layer2.csv --out models/layer2 --epochs 4
python training/train_indobert.py --data data_layer3.csv --out models/layer3 --epochs 5
```
Script otomatis: split stratified 80/20, **class weighting** (untuk L3 timpang),
dan mencetak Accuracy / Macro-F1 / Weighted-F1 / Kappa + classification report.

### 3. Pindahkan model ke laptop
Download folder `models/layerX` dari Colab/Drive **sebagai ZIP** lalu:
```bash
unzip layer3.zip -d models/
```
Cek model tidak korup: lihat **PANDUAN_HUGGINGFACE.md** bagian 2.

### 4. Verifikasi sistem memakai model fine-tuned
```bash
python backend/agents/orchestrator.py
# log harus: [Clf] model layer3 dimuat (fine-tuned)  (bukan "keyword fallback")
```

---

## C. Cara Pengujian (menghasilkan angka Bab IV)

```bash
python training/evaluate.py --test data_layer1.csv --target layer1
python training/evaluate.py --test data_layer3.csv --target layer3
```
Output: Accuracy, Macro-F1, Weighted-F1, Cohen's Kappa, classification report,
dan confusion matrix (disimpan `cm_*.csv`). **Ini angka nyata, bukan ilustrasi.**

Rencana pengujian lengkap + UAT di PLN: lihat **PANDUAN_PENGUJIAN_UAT.md**.

---

## Dampak ke Bab 3 (yang perlu kamu sebut ke dosen)

Karena sentimen kini jadi **gerbang**, ada penyesuaian narasi di Bab 3:
1. Pada **Activity/Sequence Diagram**, keputusan pertama setelah normalisasi
   adalah **"sentimen negatif?"** — bila tidak, proses berhenti (tidak masuk
   klasifikasi keluhan). Diagram di draf Bab 3 perlu menambah cabang ini.
2. Pada **3.4.3 Modeling**, tegaskan bahwa model klasifikasi L1–L3 **dilatih dan
   diuji hanya pada ulasan negatif** (sesuai populasi data PLN), sehingga
   evaluasi adil.
3. Pada **Data Preparation**, untuk melatih *model sentimen* dibutuhkan juga
   contoh positif/netral (dari scraping), karena data PLN semuanya negatif —
   ini perlu disebut agar tidak janggal saat ditanya penguji.
