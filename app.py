# ============================================================
# app.py - Aplikasi Flask untuk Klasifikasi Bumbu Dapur
# ============================================================

import os
import uuid
from datetime import datetime
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash
)
from werkzeug.utils import secure_filename
from predict import predict_image

# ─── Inisialisasi Flask ───────────────────────────────────────
app = Flask(__name__)

# Secret key untuk session & flash messages
app.secret_key = 'rempah_ai_secret_key_2024'

# ─── Konfigurasi Upload ───────────────────────────────────────
UPLOAD_FOLDER   = 'static/uploads'
ALLOWED_EXT     = {'jpg', 'jpeg', 'png'}   # Ekstensi yang diizinkan
MAX_CONTENT_MB  = 16                        # Batas ukuran file (MB)

app.config['UPLOAD_FOLDER']    = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/images', exist_ok=True)


# ─── Helper Functions ─────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """
    Memeriksa apakah ekstensi file termasuk yang diizinkan.

    Args:
        filename (str): Nama file yang akan diperiksa.

    Returns:
        bool: True jika ekstensi valid (jpg/jpeg/png), False jika tidak.
    """
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT
    )


def get_unique_filename(original_name: str) -> str:
    """
    Membuat nama file unik menggunakan UUID untuk menghindari
    tabrakan nama file di folder upload.

    Args:
        original_name (str): Nama file asli.

    Returns:
        str: Nama file baru yang unik.
    """
    ext         = original_name.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    return unique_name


# ─── Route: Halaman Utama ─────────────────────────────────────
@app.route('/')
def index():
    """
    Halaman utama aplikasi.
    Menampilkan form upload gambar bumbu dapur.
    """
    return render_template('index.html')


# ─── Route: Prediksi ─────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    Menerima gambar yang diunggah, memvalidasi, kemudian
    menjalankan prediksi menggunakan model MobileNetV2.

    Method: POST
    Form data:
        file (FileStorage): Gambar bumbu dapur (jpg/jpeg/png)

    Returns:
        render_template: Halaman result dengan hasil prediksi,
        atau redirect ke index jika terjadi error validasi.
    """
    # ── Validasi: Pastikan ada file dalam request ──
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih.', 'danger')
        return redirect(url_for('index'))

    file = request.files['file']

    # ── Validasi: Pastikan file tidak kosong ──
    if file.filename == '':
        flash('Tidak ada file yang dipilih.', 'danger')
        return redirect(url_for('index'))

    # ── Validasi: Cek ekstensi file ──
    if not allowed_file(file.filename):
        flash(
            'Format file tidak valid! '
            'Hanya mendukung JPG, JPEG, dan PNG.',
            'danger'
        )
        return redirect(url_for('index'))

    # ── Simpan file ke folder upload ──
    try:
        # Amankan nama file dan buat nama unik
        safe_name    = secure_filename(file.filename)
        unique_name  = get_unique_filename(safe_name)
        save_path    = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

        file.save(save_path)

    except Exception as e:
        flash(f'Gagal menyimpan file: {str(e)}', 'danger')
        return redirect(url_for('index'))

    # ── Jalankan Prediksi ──
    result = predict_image(save_path)

    if not result['success']:
        # Hapus file jika prediksi gagal
        if os.path.exists(save_path):
            os.remove(save_path)
        flash(f"Prediksi gagal: {result['error']}", 'danger')
        return redirect(url_for('index'))

    # ── Tambahkan informasi tambahan ke hasil ──
    result['image_filename'] = unique_name
    result['image_url']      = url_for('static',
                                       filename=f'uploads/{unique_name}')
    result['upload_time']    = datetime.now().strftime('%d %B %Y, %H:%M:%S')
    result['original_name']  = safe_name

    return render_template('result.html', result=result)


# ─── Error Handler ────────────────────────────────────────────
@app.errorhandler(413)
def file_too_large(e):
    """Tangani error file terlalu besar (> 16MB)."""
    flash(
        f'File terlalu besar! Maksimal ukuran file adalah '
        f'{MAX_CONTENT_MB}MB.',
        'danger'
    )
    return redirect(url_for('index'))


@app.errorhandler(404)
def page_not_found(e):
    """Tangani error halaman tidak ditemukan."""
    return render_template('index.html'), 404


# ─── Jalankan Aplikasi ────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Aplikasi Klasifikasi Bumbu Dapur Indonesia")
    print("  Model: MobileNetV2 Transfer Learning")
    print("=" * 60)
    print(f"  Akses di: http://127.0.0.1:5000")
    print(f"  Tekan CTRL+C untuk menghentikan server")
    print("=" * 60)

    app.run(
        debug=True,       # Mode debug (matikan saat production)
        host='0.0.0.0',  # Dapat diakses dari jaringan lokal
        port=5000
    )
