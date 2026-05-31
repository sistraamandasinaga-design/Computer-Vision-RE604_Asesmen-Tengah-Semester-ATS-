# 🔤 Klasifikasi Karakter EMNIST dengan HOG + SVM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn&logoColor=white)
![scikit-image](https://img.shields.io/badge/scikit--image-0.19+-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Klasifikasi Karakter pada Dataset EMNIST menggunakan HOG Feature Extraction dan Support Vector Machine (SVM) dengan Evaluasi Leave-One-Out Cross-Validation (LOOCV)**

</div>

---

## 📋 Informasi Proyek

| Atribut | Detail |
|---|---|
| **Nama** | Sistra Amanda Sinaga |
| **NIM** | 4222301011 |
| **Mata Kuliah** | Computer Vision (RE604) |
| **Kelas** | 6A Pagi |
| **Program Studi** | Teknik Robotika |
| **Institusi** | Politeknik Negeri Batam |
| **Dosen** | Eko Rudiawan Jamzuri |
| **Tahun Akademik** | Genap 2026 |

---

## 📖 Deskripsi Proyek

Proyek ini mengimplementasikan pipeline klasifikasi karakter alfabet (**a–z**, 26 kelas) menggunakan dataset **EMNIST Letters**. Pendekatan yang digunakan menggabungkan:

1. **HOG (Histogram of Oriented Gradients)** — untuk mengekstrak fitur bentuk dan tepi karakter dari citra grayscale 28×28 piksel.
2. **SVM (Support Vector Machine)** dengan kernel RBF — sebagai model klasifikasi multi-kelas.
3. **LOOCV (Leave-One-Out Cross-Validation)** — sebagai strategi evaluasi yang tidak bias untuk dataset berukuran kecil.

---

## 🗂️ Struktur Direktori

```
📦 emnist-hog-svm/
├── 📓 emnist_hog_svm_.ipynb       # Notebook utama
├── 📄 emnist-letters-train.csv    # Dataset EMNIST Letters (diunduh terpisah)
├── 📄 README.md                   # Dokumentasi proyek
├── 🖼️ sample_dataset.png          # Output: contoh gambar tiap kelas
├── 🖼️ hog_visualization.png       # Output: visualisasi fitur HOG
├── 🖼️ confusion_matrix.png        # Output: confusion matrix LOOCV
├── 🖼️ metrics_bar.png             # Output: grafik metrik evaluasi
└── 🖼️ sample_predictions.png      # Output: contoh hasil prediksi
```

---

## 📦 Dataset

Dataset yang digunakan adalah **EMNIST Letters** (subset dari EMNIST).

- **Sumber:** [EMNIST Dataset – NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) / [Kaggle EMNIST](https://www.kaggle.com/datasets/crawford/emnist)
- **Format:** CSV (`emnist-letters-train.csv`)
- **Struktur:** Kolom pertama = label (1–26), kolom 2–785 = nilai piksel (28×28)
- **Kelas:** 26 huruf alfabet lowercase (a–z)

> ⚠️ **Catatan:** File CSV tidak disertakan dalam repositori karena ukurannya yang besar. Unduh secara terpisah dan letakkan di folder yang sama dengan notebook.

### Langkah Download Dataset

1. Buka [Kaggle EMNIST](https://www.kaggle.com/datasets/crawford/emnist)
2. Unduh dan ekstrak arsip
3. Salin file `emnist-letters-train.csv` ke direktori proyek

---

## ⚙️ Instalasi & Persyaratan

### Prasyarat

- Python 3.8 atau lebih baru
- Jupyter Notebook / JupyterLab

### Install Dependensi

```bash
pip install scikit-image scikit-learn matplotlib seaborn tqdm numpy
```

Atau menggunakan file requirements (jika tersedia):

```bash
pip install -r requirements.txt
```

### Dependensi Lengkap

| Library | Fungsi |
|---|---|
| `numpy` | Operasi array dan matriks |
| `scikit-image` | Ekstraksi fitur HOG |
| `scikit-learn` | SVM, LOOCV, metrik evaluasi |
| `matplotlib` | Visualisasi gambar dan grafik |
| `seaborn` | Heatmap confusion matrix |
| `tqdm` | Progress bar saat ekstraksi HOG |

---

## 🚀 Cara Menjalankan

1. **Clone repositori:**
   ```bash
   git clone https://github.com/username/emnist-hog-svm.git
   cd emnist-hog-svm
   ```

2. **Siapkan dataset:**
   Letakkan file `emnist-letters-train.csv` di folder yang sama dengan notebook.

3. **Buka notebook:**
   ```bash
   jupyter notebook emnist_hog_svm_.ipynb
   ```

4. **Jalankan semua sel** dari atas ke bawah (`Kernel > Restart & Run All`).

> ⏱️ Estimasi waktu eksekusi: **2–3 menit** untuk proses LOOCV (390 iterasi dengan `n_jobs=-1`).

---

## 🔬 Metodologi

### 1. Preprocessing Data

- Membaca CSV dan memisahkan label serta piksel
- Reshape piksel menjadi gambar 28×28
- **Koreksi orientasi EMNIST:** rotasi 270° + horizontal flip
- Normalisasi nilai piksel ke rentang [0, 1]
- Pengambilan subset seimbang: **15 sampel × 26 kelas = 390 sampel**

### 2. Ekstraksi Fitur HOG

HOG menghitung distribusi arah gradien pada sel-sel lokal untuk menangkap informasi bentuk dan tepi karakter.

| Parameter HOG | Nilai | Keterangan |
|---|---|---|
| `orientations` | 9 | Jumlah bin arah gradien |
| `pixels_per_cell` | (4, 4) | Ukuran cell dalam piksel |
| `cells_per_block` | (2, 2) | Jumlah cell per block normalisasi |
| `block_norm` | L2-Hys | Metode normalisasi |
| `transform_sqrt` | True | Gamma correction sebelum HOG |

Dimensi vektor fitur HOG yang dihasilkan per gambar: **`324` dimensi**

### 3. Klasifikasi dengan SVM

| Parameter SVM | Nilai | Keterangan |
|---|---|---|
| `kernel` | rbf | Radial Basis Function |
| `C` | 10.0 | Parameter regularisasi |
| `gamma` | scale | Koefisien kernel otomatis |
| `decision_function_shape` | ovr | One-vs-Rest multi-kelas |

### 4. Evaluasi LOOCV

LOOCV bekerja dengan cara:
- Diberikan **N** total sampel → lakukan **N iterasi**
- Setiap iterasi: **1 sampel** sebagai data uji, **N-1 sampel** sisanya sebagai data latih
- Setelah semua iterasi → hitung metrik dari seluruh prediksi

**Keunggulan LOOCV:**
- Tidak ada data yang terbuang (semua dipakai untuk training)
- Estimasi performa paling tidak bias
- Cocok untuk dataset berukuran kecil

---

## 📊 Hasil Evaluasi

Evaluasi dilakukan dengan LOOCV pada **390 sampel** (15 per kelas × 26 kelas).

| Metrik | Nilai |
|---|---|
| **Accuracy** | *lihat output notebook* |
| **Precision** (macro-avg) | *lihat output notebook* |
| **F1-Score** (macro-avg) | *lihat output notebook* |

> 📌 Nilai aktual bergantung pada hasil eksekusi notebook dengan dataset yang disiapkan.

### Output yang Dihasilkan

| File | Deskripsi |
|---|---|
| `sample_dataset.png` | Contoh satu gambar per kelas (a–z) |
| `hog_visualization.png` | Perbandingan gambar asli vs fitur HOG untuk 5 huruf |
| `confusion_matrix.png` | Confusion matrix hasil LOOCV (26×26) |
| `metrics_bar.png` | Bar chart Accuracy, Precision, F1-Score |
| `sample_predictions.png` | 20 contoh prediksi (hijau = benar, merah = salah) |

---

## 🖼️ Contoh Visualisasi

### HOG Feature Visualization
Fitur HOG mengekstrak informasi arah gradien yang merepresentasikan bentuk dan tepi karakter secara lokal.

### Confusion Matrix
Confusion matrix 26×26 menunjukkan performa per kelas dan pola kesalahan prediksi antar huruf yang mirip.

---

## 📚 Referensi

- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR.
- Cohen, G., et al. (2017). *EMNIST: Extending MNIST to handwritten letters*. IJCNN.
- Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine Learning.
- [scikit-image HOG Documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog)
- [scikit-learn SVC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [scikit-learn LeaveOneOut Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik — **Asesmen Akhir Semester UTS Computer Vision (RE604)**, Politeknik Negeri Batam.

---

<div align="center">
  Made with ❤️ by <strong>Sistra Amanda Sinaga</strong> · Politeknik Negeri Batam · 2026
</div>
