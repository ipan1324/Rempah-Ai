# ============================================================
# predict.py - Modul Fungsi Prediksi Gambar
# Digunakan oleh app.py (Flask) untuk inferensi model
# ============================================================

import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf

# ─── Konfigurasi ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)
MODEL_PATH  = 'model/mobilenet_bumbu.keras'

# Nama kelas sesuai urutan subfolder (sorted alphabetically)
CLASS_NAMES = ['Kencur', 'Kunyit', 'Lengkuas']

# Deskripsi singkat masing-masing kelas
CLASS_INFO = {
    'Kencur': {
        'nama_latin': 'Kaempferia galanga',
        'kegunaan'  : 'Banyak digunakan dalam masakan Sunda dan Jawa, '
                      'seperti nasi liwet dan pecel. Dikenal dengan aroma '
                      'yang kuat dan rasa yang sedikit pedas.',
        'warna'     : '#4CAF50'
    },
    'Kunyit': {
        'nama_latin': 'Curcuma longa',
        'kegunaan'  : 'Digunakan sebagai pewarna alami dan bumbu masakan '
                      'kari, rendang, dan gulai. Mengandung kurkumin '
                      'yang bermanfaat sebagai antiinflamasi.',
        'warna'     : '#FF9800'
    },
    'Lengkuas': {
        'nama_latin': 'Alpinia galanga',
        'kegunaan'  : 'Bumbu wajib dalam masakan rendang, soto, dan '
                      'berbagai masakan Indonesia. Memberikan aroma '
                      'khas yang harum dan segar.',
        'warna'     : '#9C27B0'
    }
}

# ─── Load Model ───────────────────────────────────────────────
_model = None   # Gunakan cache agar model tidak dimuat berulang kali

def load_model():
    """
    Memuat model Keras dari file .keras.
    Model hanya dimuat sekali (singleton pattern).
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model tidak ditemukan di '{MODEL_PATH}'. "
                "Jalankan train.py terlebih dahulu."
            )
        print(f"  [Predict] Memuat model dari: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("  [Predict] Model berhasil dimuat!")
    return _model


# ─── Fungsi Preprocessing Gambar ─────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """
    Membaca dan memproses gambar menjadi tensor siap prediksi.

    Args:
        image_path (str): Path lengkap ke file gambar.

    Returns:
        np.ndarray: Array dengan shape (1, 224, 224, 3),
                    nilai piksel dinormalisasi ke [0, 1].
    """
    # Buka gambar, konversi ke RGB (hindari RGBA / grayscale)
    img = Image.open(image_path).convert('RGB')

    # Resize ke ukuran input MobileNetV2
    img = img.resize(IMG_SIZE)

    # Konversi ke array NumPy
    img_array = np.array(img, dtype=np.float32)

    # Normalisasi piksel ke rentang [0, 1]
    img_array = img_array / 255.0

    # Tambah dimensi batch: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ─── Fungsi Utama Prediksi ────────────────────────────────────
def predict_image(image_path: str) -> dict:
    """
    Melakukan prediksi kelas bumbu dapur dari gambar.

    Args:
        image_path (str): Path ke file gambar yang akan diprediksi.

    Returns:
        dict: Dictionary hasil prediksi berisi:
            - predicted_class (str) : Nama kelas yang diprediksi
            - confidence (float)    : Skor kepercayaan 0.0 – 1.0
            - confidence_pct (str)  : Confidence dalam format persen
            - all_probabilities (list): Probabilitas semua kelas
            - prediction_time (str) : Waktu inferensi dalam milidetik
            - class_info (dict)     : Informasi tambahan kelas
            - success (bool)        : Status prediksi berhasil/gagal
    """
    try:
        # Catat waktu mulai inferensi
        start_time = time.time()

        # Load model (dari cache jika sudah dimuat)
        model = load_model()

        # Preprocess gambar
        img_array = preprocess_image(image_path)

        # Jalankan prediksi
        predictions = model.predict(img_array, verbose=0)

        # Hitung waktu inferensi
        elapsed_ms = (time.time() - start_time) * 1000

        # Ambil indeks kelas dengan probabilitas tertinggi
        class_idx   = int(np.argmax(predictions[0]))
        confidence  = float(predictions[0][class_idx])

        # Nama kelas hasil prediksi
        predicted_class = CLASS_NAMES[class_idx]

        # Susun probabilitas semua kelas
        all_probs = [
            {
                'class'      : CLASS_NAMES[i],
                'probability': float(predictions[0][i]),
                'pct'        : f"{predictions[0][i]*100:.2f}%"
            }
            for i in range(len(CLASS_NAMES))
        ]

        # Urutkan dari probabilitas tertinggi
        all_probs.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'success'          : True,
            'predicted_class'  : predicted_class,
            'confidence'       : confidence,
            'confidence_pct'   : f"{confidence*100:.2f}%",
            'all_probabilities': all_probs,
            'prediction_time'  : f"{elapsed_ms:.1f} ms",
            'class_info'       : CLASS_INFO.get(predicted_class, {})
        }

    except FileNotFoundError as e:
        return {
            'success': False,
            'error'  : str(e)
        }
    except Exception as e:
        return {
            'success': False,
            'error'  : f"Terjadi kesalahan saat prediksi: {str(e)}"
        }


# ─── Test Standalone (jalankan langsung) ─────────────────────
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Penggunaan: python predict.py <path_gambar>")
        print("Contoh    : python predict.py dataset/test/Kencur/img001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' tidak ditemukan.")
        sys.exit(1)

    print(f"\nMemproses gambar: {image_path}")
    result = predict_image(image_path)

    if result['success']:
        print(f"\n  Hasil Prediksi  : {result['predicted_class']}")
        print(f"  Confidence      : {result['confidence_pct']}")
        print(f"  Waktu prediksi  : {result['prediction_time']}")
        print(f"\n  Probabilitas semua kelas:")
        for prob in result['all_probabilities']:
            bar = '█' * int(prob['probability'] * 30)
            print(f"    {prob['class']:10s} : {bar} {prob['pct']}")
    else:
        print(f"\n  Error: {result['error']}")
