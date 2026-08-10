# PANDUAN MODEL & HUGGINGFACE — Anti-Korup

Dokumen ini menjawab masalah yang sering kamu alami: **"HuggingFace tidak
terkoneksi"** dan **"model korup"**. Baca sebelum training & sebelum demo.

---

## 1. Kenapa pendekatan lama bermasalah

Kode lama memakai **HuggingFace Inference API** (memanggil model lewat internet
setiap kali). Ini sumber masalahmu:

- Model bisa **"loading"** lama (cold start) → timeout.
- Endpoint bisa **berubah / deprecated** sewaktu-waktu.
- Butuh **internet stabil** saat demo (berisiko saat sidang).
- Rate limit → request ditolak.

**Solusi sistem baru: model dijalankan LOKAL.** Unduh sekali, jalankan dari
disk. Tidak ada panggilan internet saat inferensi. Stabil saat demo.

---

## 2. Apa itu "model korup" & tanda-tandanya

Model korup = file model terunduh **tidak lengkap / rusak**, biasanya karena
download terputus. Tanda-tandanya:

| Gejala | Penyebab |
|---|---|
| `OSError: Unable to load weights ...` | file `.safetensors`/`.bin` tidak lengkap |
| `safetensors_rust.SafetensorError: ... header` | file terpotong saat download |
| `EOFError` / `RuntimeError: unexpected EOF` | unduhan terputus |
| Ukuran file model jauh lebih kecil dari seharusnya | download gagal di tengah |
| Prediksi selalu kelas yang sama / acak | bobot tidak termuat benar |

### Cara cek cepat apakah model utuh
```bash
# IndoBERT base ~ 440 MB. Jika file model jauh lebih kecil → korup.
ls -lh models/layer3/
# harus ada: config.json, tokenizer files, dan model.safetensors (atau pytorch_model.bin)
python -c "from transformers import AutoModelForSequenceClassification as M; M.from_pretrained('models/layer3'); print('OK utuh')"
```
Jika baris terakhir mencetak `OK utuh`, model tidak korup.

---

## 3. Cara mengunduh model dengan AMAN (anti-korup)

### A. Unduh penuh & verifikasi (disarankan)
```bash
pip install -U "huggingface_hub[cli]"
# unduh snapshot lengkap ke cache lokal (resume otomatis bila putus)
huggingface-cli download indobenchmark/indobert-base-p1
huggingface-cli download mdhugol/indonesia-bert-sentiment-classification
```
`huggingface-cli download` otomatis **melanjutkan** unduhan yang terputus dan
**memverifikasi hash** — ini cara paling aman menghindari korup.

### B. Kalau sudah terlanjur korup → bersihkan cache lalu unduh ulang
```bash
# hapus cache model yang rusak
rm -rf ~/.cache/huggingface/hub/models--indobenchmark--indobert-base-p1
# unduh ulang
huggingface-cli download indobenchmark/indobert-base-p1
```

### C. Saat training di Colab: simpan model ke Google Drive, jangan unduh ulang
```python
from google.colab import drive; drive.mount('/content/drive')
# setelah training:
model.save_pretrained('/content/drive/MyDrive/setrum_models/layer3')
tokenizer.save_pretrained('/content/drive/MyDrive/setrum_models/layer3')
```
Lalu **download folder itu** dari Drive ke `models/layer3/` di laptop.
Pindahkan sebagai **ZIP** agar tidak ada file yang putus saat transfer:
```bash
# di Drive: kompres folder jadi zip, download zip, lalu:
unzip layer3.zip -d models/
```

---

## 4. Aturan agar sistem TIDAK PERNAH crash

Sistem dirancang berlapis (graceful fallback):

1. **Sentimen**: model lokal → bila gagal muat → lexicon rule-based.
2. **Klasifikasi L1–L3**: model fine-tuned lokal → bila belum ada → keyword.
3. **Eskalasi L4–L5**: zero-shot lokal / Ollama → bila gagal → ambil opsi
   pertama dari hierarki.

Artinya: walau model belum siap, dashboard tetap jalan untuk demo alur.
**TAPI** untuk hasil skripsi, pastikan model fine-tuned benar-benar termuat
(cek log: harus muncul `[Clf] model layer3 dimuat (fine-tuned)`).

---

## 5. Checklist sebelum demo / sidang

- [ ] `python -c "from transformers import AutoModel"` jalan tanpa error
- [ ] Folder `models/layer1, layer2, layer3, sentiment` terisi & lolos cek utuh (bagian 2)
- [ ] Jalankan `python backend/agents/orchestrator.py` → log menyebut model fine-tuned, bukan keyword
- [ ] Matikan internet, jalankan analisis manual → tetap berfungsi (bukti lokal)
- [ ] `evaluate.py` menghasilkan angka, bukan error
