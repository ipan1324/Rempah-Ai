# 🌿 RempahAI – Klasifikasi Bumbu Dapur Indonesia

> **Implementasi Transfer Learning MobileNetV2 untuk Klasifikasi Citra Bumbu Dapur Indonesia Berbasis Website Menggunakan Flask**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?logo=bootstrap)](https://getbootstrap.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Deskripsi Proyek

**RempahAI** adalah aplikasi web berbasis kecerdasan buatan yang mampu mengklasifikasikan citra bumbu dapur Indonesia secara otomatis. Proyek ini mengimplementasikan teknik **Transfer Learning** menggunakan arsitektur **MobileNetV2** yang telah dilatih pada dataset ImageNet, kemudian di-*fine-tune* untuk mengenali tiga jenis bumbu dapur Indonesia:

| Kelas | Nama Latin | Kegunaan |
|-------|-----------|----------|
| 🌿 **Kencur** | *Kaempferia galanga* | Masakan Sunda, nasi liwet, pecel |
| 🟡 **Kunyit** | *Curcuma longa* | Kari, rendang, pewarna alami |
| 🌱 **Lengkuas** | *Alpinia galanga* | Rendang, soto, opor |

### Fitur Utama
- ✅ Klasifikasi gambar bumbu dengan akurasi tinggi
- ✅ Antarmuka web modern dengan Bootstrap 5 (dark mode)
- ✅ Drag & Drop upload gambar dengan preview
- ✅ Confidence score dengan visualisasi bar
- ✅ Probabilitas untuk semua kelas
- ✅ Waktu prediksi real-time
- ✅ Validasi file (JPG, JPEG, PNG)
- ✅ Responsif untuk semua ukuran layar

---

## 🗂 Struktur Folder

```
rempah-ai/
│
├── app.py                        # Aplikasi Flask utama (routing & logika web)
├── train.py                      # Script training model MobileNetV2
├── predict.py                    # Modul fungsi prediksi (dipanggil Flask)
├── requirements.txt              # Daftar dependensi Python
│
├── model/
│   └── mobilenet_bumbu.keras     # Model terlatih (dihasilkan train.py)
│
├── dataset/
│   ├── train/                    # Data training (per subfolder kelas)
│   │   ├── Kencur/
│   │   ├── Kunyit/
│   │   └── Lengkuas/
│   └── test/                     # Data testing (per subfolder kelas)
│       ├── Kencur/
│       ├── Kunyit/
│       └── Lengkuas/
│
├── static/
│   ├── css/
│   │   ├── style.css             # CSS kustom (dark mode, animasi)
│   │   └── script.js             # JavaScript global
│   ├── uploads/                  # Gambar yang diunggah pengguna
│   └── images/                   # Grafik training & confusion matrix
│
├── templates/
│   ├── base.html                 # Template dasar (navbar, footer)
│   ├── index.html                # Halaman utama (form upload)
│   └── result.html               # Halaman hasil prediksi
│
└── README.md                     # Dokumentasi proyek ini
```

---

## ⚙️ Instalasi

### Prasyarat
- Python 3.11 (disarankan menggunakan virtual environment)
- pip (package manager Python)
- Dataset dari Kaggle: [Spices Classification](https://www.kaggle.com/datasets/nisarizqi/spices-classification)

### Langkah Instalasi

**1. Clone atau download proyek ini**
```bash
# Jika menggunakan git
git clone <url-repositori>
cd rempah-ai

# Atau ekstrak file ZIP ke folder pilihan Anda
```

**2. Buat dan aktifkan virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install semua dependensi**
```bash
pip install -r requirements.txt
```

**4. Siapkan dataset**

Unduh dataset dari Kaggle:
- https://www.kaggle.com/datasets/nisarizqi/spices-classification

Ekstrak dan susun dataset dengan struktur berikut:
```
dataset/
├── train/
│   ├── Kencur/        ← gambar kencur untuk training
│   ├── Kunyit/        ← gambar kunyit untuk training
│   └── Lengkuas/      ← gambar lengkuas untuk training
└── test/
    ├── Kencur/        ← gambar kencur untuk testing
    ├── Kunyit/        ← gambar kunyit untuk testing
    └── Lengkuas/      ← gambar lengkuas untuk testing
```

> ⚠️ **Penting:** Nama subfolder harus persis sama: `Kencur`, `Kunyit`, `Lengkuas` (kapital pertama).

**5. Buat folder yang diperlukan**
```bash
# Windows
mkdir model
mkdir static\images
mkdir static\uploads

# macOS / Linux
mkdir -p model static/images static/uploads
```

---

## 🏋️ Cara Training Model

Setelah dataset tersedia di folder `dataset/train` dan `dataset/test`, jalankan:

```bash
python train.py
```

Proses training akan:
1. **Memuat dataset** dengan augmentasi (rotasi, flip, zoom, dll.)
2. **Memuat MobileNetV2** pretrained ImageNet
3. **Membekukan** (freeze) seluruh layer base model
4. **Menambahkan** custom classification head:
   - `GlobalAveragePooling2D`
   - `Dropout(0.3)`
   - `Dense(128, relu)`
   - `Dropout(0.2)`
   - `Dense(3, softmax)`
5. **Melatih** model dengan:
   - Optimizer: Adam (lr=0.0001)
   - Loss: Categorical Crossentropy
   - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
6. **Menyimpan** model terbaik ke `model/mobilenet_bumbu.keras`
7. **Menampilkan** grafik akurasi, loss, confusion matrix, dan classification report

### Output Training
```
model/
└── mobilenet_bumbu.keras         ← Model terbaik

static/images/
├── training_history.png          ← Grafik Accuracy & Loss
├── confusion_matrix.png          ← Confusion Matrix
└── classification_report.txt     ← Laporan evaluasi lengkap
```

### Konfigurasi Training (opsional)
Edit bagian **Konfigurasi Parameter** di `train.py`:
```python
IMG_SIZE      = (224, 224)   # Ukuran input
BATCH_SIZE    = 32           # Batch size
EPOCHS        = 20           # Maksimal epoch
LEARNING_RATE = 0.0001       # Learning rate Adam
```

---

## 🚀 Cara Menjalankan Aplikasi Flask

Pastikan model sudah terlatih (`model/mobilenet_bumbu.keras` ada), lalu:

```bash
python app.py
```

Buka browser dan akses:
```
http://127.0.0.1:5000
```

### Menjalankan di Production (dengan Gunicorn – Linux/macOS)
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## 📱 Cara Menggunakan Aplikasi

1. **Buka** browser dan akses `http://127.0.0.1:5000`
2. **Unggah gambar** bumbu dapur dengan cara:
   - Klik area upload dan pilih file, **atau**
   - Drag & Drop gambar ke area upload
3. **Preview** gambar akan tampil otomatis sebelum prediksi
4. Klik tombol **"Prediksi Sekarang"**
5. Lihat **hasil prediksi** yang mencakup:
   - Nama kelas bumbu yang teridentifikasi
   - Confidence score (tingkat keyakinan)
   - Probabilitas untuk semua kelas
   - Waktu inferensi
   - Informasi tentang bumbu tersebut

### Format File yang Didukung
| Format | Ekstensi |
|--------|----------|
| JPEG   | `.jpg`, `.jpeg` |
| PNG    | `.png` |

> **Ukuran maksimal:** 16 MB per file

---

## 🧠 Arsitektur Model

```
Input (224×224×3)
      ↓
MobileNetV2 (pretrained ImageNet, FROZEN)
  - 154 layer konvolusional
  - Feature extractor generik
      ↓
GlobalAveragePooling2D
  - Mengubah (7×7×1280) → (1280,)
      ↓
Dropout(0.3)
      ↓
Dense(128, activation='relu')
      ↓
Dropout(0.2)
      ↓
Dense(3, activation='softmax')
      ↓
Output: [P(Kencur), P(Kunyit), P(Lengkuas)]
```

---

## 📦 Dependency

| Library | Versi | Kegunaan |
|---------|-------|----------|
| `tensorflow` | 2.16.1 | Framework deep learning, Keras API |
| `keras` | 3.3.3 | High-level neural network API |
| `numpy` | 1.26.4 | Operasi array numerik |
| `Pillow` | 10.3.0 | Baca dan proses gambar |
| `scikit-learn` | 1.4.2 | Evaluasi model (confusion matrix, report) |
| `matplotlib` | 3.8.4 | Visualisasi grafik training |
| `seaborn` | 0.13.2 | Visualisasi heatmap |
| `Flask` | 3.0.3 | Web framework untuk deployment |
| `Werkzeug` | 3.0.3 | WSGI utilities (file upload) |
| `h5py` | 3.11.0 | Penyimpanan model HDF5 |
| `tqdm` | 4.66.4 | Progress bar terminal |

---

## 📊 Hasil Training (Contoh)

| Metrik | Nilai |
|--------|-------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~92% |
| Test Accuracy | ~90%+ |

> Hasil aktual bergantung pada kualitas dan jumlah dataset.

---

## 🔧 Troubleshooting

**Model tidak ditemukan:**
```
FileNotFoundError: Model tidak ditemukan di 'model/mobilenet_bumbu.keras'
```
→ Jalankan `python train.py` terlebih dahulu.

**Error saat training (dataset tidak ditemukan):**
```
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/train'
```
→ Pastikan folder dataset sudah ada dan berisi subfolder kelas yang benar.

**GPU tidak terdeteksi:**
→ Install TensorFlow GPU: `pip install tensorflow[and-cuda]==2.16.1`

**Port 5000 sudah digunakan:**
```bash
# Ganti port di app.py
app.run(port=5001)
```

---

## 👥 Referensi

- Dataset: [Spices Classification – Kaggle](https://www.kaggle.com/datasets/nisarizqi/spices-classification)
- Model: [MobileNetV2 – Google AI](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)
- Transfer Learning Guide: [TensorFlow Documentation](https://www.tensorflow.org/guide/keras/transfer_learning)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik / tugas akhir. Bebas digunakan dan dimodifikasi dengan menyertakan atribusi.

---

*Dibuat dengan ❤️ menggunakan Python, TensorFlow, dan Flask*
