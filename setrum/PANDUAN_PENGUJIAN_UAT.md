# PANDUAN PENGUJIAN & UAT — Sistem SETRUM

Dokumen ini menjelaskan strategi pengujian agar hasil Bab IV kuat dan tahan
kritik penguji, serta rencana User Acceptance Test (UAT) di PLN.

Pengujian dibagi tiga lapis: **(1) pengujian model**, **(2) pengujian sistem
(fungsional)**, **(3) UAT (penerimaan pengguna)**. Ketiganya saling melengkapi.

---

## 1. PENGUJIAN MODEL (kuantitatif — inti Bab IV)

Tujuan: membuktikan akurasi klasifikasi terhadap **ground truth label tim PLN**.

### 1.1 Metrik
Gunakan kombinasi, bukan hanya akurasi:
- **Accuracy** — proporsi benar (kurang andal saat kelas timpang).
- **Macro-F1** — rata-rata F1 antar kelas; mengungkap performa kelas minoritas.
- **Weighted-F1** — F1 berbobot frekuensi; performa keseluruhan.
- **Cohen's Kappa** — kesepakatan sistem vs label PLN di luar kebetulan.

> **Lapor Macro-F1 dan Weighted-F1 BERDAMPINGAN.** Selisih besar antara keduanya
> di Layer 3 itu WAJAR (data timpang) dan justru menunjukkan kejujuran. Jangan
> sembunyikan; jelaskan penyebabnya (kelas minoritas sedikit sampel).

### 1.2 Protokol
- Split **stratified 80/20** (sudah otomatis di `train_indobert.py`).
- Data uji = 20% yang **tidak pernah dilihat** saat training.
- Jalankan `evaluate.py` per target → simpan confusion matrix.
- Interpretasi Kappa: <0,21 buruk; 0,21–0,40 cukup; 0,41–0,60 sedang;
  0,61–0,80 baik; 0,81–1,00 sangat baik (Landis & Koch, 1977).

### 1.3 Kriteria keberhasilan (tetapkan di Bab III)
- F1-Score ≥ 0,75 dan Cohen's Kappa ≥ 0,61 → kategori "baik".

### 1.4 Tabel hasil yang harus diisi (contoh format)
| Target | Accuracy | Macro-F1 | Weighted-F1 | Kappa | Interpretasi |
|---|---|---|---|---|---|
| Sentimen | ... | ... | ... | ... | ... |
| Layer 1 | ... | ... | ... | ... | ... |
| Layer 2 | ... | ... | ... | ... | ... |
| Layer 3 | ... | ... | ... | ... | ... |

> Isi dengan angka nyata dari `evaluate.py`. Jangan pakai angka ilustrasi.

---

## 2. PENGUJIAN SISTEM (fungsional — Black Box)

Tujuan: membuktikan tiap fungsi dashboard berjalan sesuai harapan.
Gunakan metode **Black Box Testing** dengan tabel kasus uji.

| ID | Skenario | Input | Hasil Diharapkan | Status |
|---|---|---|---|---|
| TC-01 | Scraping ulasan | jumlah=200, sort=terbaru | 200 ulasan tersimpan di DB | |
| TC-02 | Analisis batch | 100 ulasan pending | Semua terproses, status 100% | |
| TC-03 | Gerbang sentimen | ulasan positif | Tidak masuk klasifikasi keluhan | |
| TC-04 | Klasifikasi negatif | "token tidak masuk" | L1=APLIKASI, L3=Token & Pembayaran | |
| TC-05 | Eskalasi L4-L5 | keluhan spesifik | L4 & L5 terisi sesuai hierarki | |
| TC-06 | L1=UMUM | ulasan umum | Tidak eskalasi L4-L5 | |
| TC-07 | Dashboard grafik | data tersedia | Grafik sentimen & kategori tampil | |
| TC-08 | Filter detail | filter=Negatif | Hanya ulasan negatif tampil | |
| TC-09 | Analisis manual | satu teks | Hasil sentimen + klasifikasi tampil | |
| TC-10 | Model korup/absen | model dihapus | Sistem fallback, tidak crash | |

---

## 3. USER ACCEPTANCE TEST (UAT) DI PLN

Tujuan: membuktikan sistem **diterima dan bermanfaat** bagi pengguna nyata
(tim Divisi Customer Experience PLN). Ini nilai tambah besar untuk TA Sistem
Informasi — menunjukkan dampak organisasi, bukan sekadar model.

### 3.1 Responden
Pilih 3–10 orang dari tim yang benar-benar memakai klasifikasi ulasan
(analis / PIC evaluasi customer journey). Catat peran mereka.

### 3.2 Skenario tugas UAT
Minta responden melakukan tugas nyata memakai dashboard, lalu menilai:
1. Menarik ulasan terbaru (scraping).
2. Menjalankan analisis batch.
3. Membaca distribusi keluhan di dashboard.
4. Menelusuri 5 ulasan dan menilai apakah klasifikasi sistem **sesuai**
   dengan penilaian manual mereka.
5. Menguji 3 ulasan via Analisis Manual.

### 3.3 Instrumen kuesioner
Gunakan skala Likert 1–5 (1=sangat tidak setuju, 5=sangat setuju). Disarankan
memakai kerangka baku agar bisa dikutip:

**Opsi A — System Usability Scale (SUS), 10 pernyataan baku** (Brooke, 1996).
Skor SUS 0–100; >68 = di atas rata-rata. Cocok untuk menilai kemudahan pakai.

**Opsi B — Technology Acceptance Model (TAM)** (Davis, 1989), dua dimensi:
- *Perceived Usefulness* (kebermanfaatan): "Sistem mempercepat analisis ulasan."
- *Perceived Ease of Use* (kemudahan): "Sistem mudah dioperasikan."

Contoh pernyataan UAT (sesuaikan):
1. Sistem mempercepat proses klasifikasi ulasan dibanding cara manual.
2. Klasifikasi yang dihasilkan sesuai dengan hierarki keluhan PLN.
3. Hasil sentimen sesuai dengan isi ulasan.
4. Dashboard mudah dipahami dan dioperasikan.
5. Visualisasi membantu mengenali area keluhan prioritas.
6. Saya bersedia menggunakan sistem ini dalam pekerjaan.

### 3.4 Analisis hasil UAT
- Hitung rata-rata skor per pernyataan dan keseluruhan.
- Bila pakai SUS, hitung skor SUS standar.
- Tampilkan dalam tabel + grafik batang di Bab IV.
- Sertakan kutipan kualitatif (saran responden) sebagai bahan "Saran".

### 3.5 Bukti UAT (lampiran)
- Lembar kuesioner terisi (atau tangkapan layar Google Form).
- Daftar hadir / nama & peran responden.
- Dokumentasi sesi (foto/tangkapan layar Zoom).

---

## 4. Triangulasi: kenapa tiga lapis ini kuat

- Pengujian **model** menjawab: "seberapa akurat?" (objektif, angka).
- Pengujian **fungsional** menjawab: "apakah semua fitur jalan?" (sistem).
- **UAT** menjawab: "apakah berguna & diterima pengguna?" (dampak nyata).

Penguji SI sangat menghargai kombinasi ini karena menutup celah umum: TA yang
hanya melaporkan akurasi tanpa membuktikan kebermanfaatan, atau sebaliknya.

---

## Referensi instrumen (verifikasi sendiri sebelum sitasi)
- Brooke, J. (1996). SUS: A quick and dirty usability scale.
- Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user
  acceptance of information technology. MIS Quarterly.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement
  for categorical data. Biometrics.
