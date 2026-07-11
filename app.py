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

# --- Inisialisasi Flask ---
app = Flask(__name__)

# Secret key untuk session & flash messages
app.secret_key = 'rempah_ai_secret_key_2024'

# --- Konfigurasi Upload ---
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER   = os.path.join(BASE_DIR, 'static', 'uploads')
STATIC_IMAGES   = os.path.join(BASE_DIR, 'static', 'images')
ALLOWED_EXT     = {'jpg', 'jpeg', 'png'}
MAX_CONTENT_MB  = 16

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMAGES, exist_ok=True)

# --- Helper Functions ---
def allowed_file(filename: str) -> bool:
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT)

def get_unique_filename(original_name: str) -> str:
    ext         = original_name.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    return unique_name

# --- Route: Halaman Utama ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Route: Prediksi ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih.', 'danger')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('Tidak ada file yang dipilih.', 'danger')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Format file tidak valid! Hanya mendukung JPG, JPEG, dan PNG.', 'danger')
        return redirect(url_for('index'))

    try:
        safe_name    = secure_filename(file.filename)
        unique_name  = get_unique_filename(safe_name)
        save_path    = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

        file.save(save_path)
    except Exception as e:
        flash(f'Gagal menyimpan file: {str(e)}', 'danger')
        return redirect(url_for('index'))

    result = predict_image(save_path)

    if not result.get('success', False):
        if os.path.exists(save_path):
            os.remove(save_path)
        flash(f"Prediksi gagal: {result.get('error', 'Unknown Error')}", 'danger')
        return redirect(url_for('index'))

    result['image_filename'] = unique_name
    result['image_url']      = url_for('static', filename=f'uploads/{unique_name}')
    result['upload_time']    = datetime.now().strftime('%d %B %Y, %H:%M:%S')
    result['original_name']  = safe_name

    return render_template('result.html', result=result)

# --- Error Handler ---
@app.errorhandler(413)
def file_too_large(e):
    flash(f'File terlalu besar! Maksimal ukuran file adalah {MAX_CONTENT_MB}MB.', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    print("=" * 60)
    print("  Aplikasi Klasifikasi Bumbu Dapur Indonesia")
    print("  Model: MobileNetV2 Transfer Learning")
    print("=" * 60)
    print(f"  Akses di: http://127.0.0.1:5000")
    print(f"  Tekan CTRL+C untuk menghentikan server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
