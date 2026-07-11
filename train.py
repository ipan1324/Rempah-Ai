# ============================================================
# train.py - Script Training Model MobileNetV2
# Klasifikasi Bumbu Dapur Indonesia (Kencur, Kunyit, Lengkuas)
# ============================================================

# --- 1. Import Library ---------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# Scikit-learn untuk evaluasi
from sklearn.metrics import (
    confusion_matrix, classification_report
)

print("=" * 60)
print("  TRAINING - Klasifikasi Bumbu Dapur Indonesia")
print("  Model: MobileNetV2 (Transfer Learning)")
print("=" * 60)
print(f"  TensorFlow versi : {tf.__version__}")
print(f"  GPU tersedia     : {len(tf.config.list_physical_devices('GPU')) > 0}")
print("=" * 60)

# --- 2. Konfigurasi Parameter --------------------------------
IMG_SIZE    = (224, 224)   # Ukuran input MobileNetV2
BATCH_SIZE  = 92           # Jumlah sampel per iterasi
EPOCHS      = 10           # Maksimal epoch training
LEARNING_RATE = 0.0001     # Learning rate Adam optimizer

# Path dataset
TRAIN_DIR   = 'dataset/train'
TEST_DIR    = 'dataset/test'

# Path penyimpanan model
MODEL_DIR   = 'model'
MODEL_PATH  = os.path.join(MODEL_DIR, 'mobilenet_bumbu.keras')

# Nama kelas (sesuai subfolder dataset)
CLASS_NAMES = ['Jahe', 'Kencur', 'Kunyit', 'Lengkuas']
NUM_CLASSES = len(CLASS_NAMES)

# Buat folder model jika belum ada
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

print(f"\n  Kelas yang dikenali : {CLASS_NAMES}")
print(f"  Ukuran gambar       : {IMG_SIZE}")
print(f"  Batch size          : {BATCH_SIZE}")
print(f"  Epoch maksimal      : {EPOCHS}")
print(f"  Learning rate       : {LEARNING_RATE}\n")

# --- 3. Preprocessing & Augmentasi Data ----------------------
print("  [1/6] Mempersiapkan data generator...")

# Generator untuk data TRAINING (dengan augmentasi)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalisasi piksel ke [0, 1]
    rotation_range=20,           # Rotasi acak ±20 derajat
    width_shift_range=0.15,      # Geser horizontal
    height_shift_range=0.15,     # Geser vertikal
    shear_range=0.1,             # Transformasi geser
    zoom_range=0.15,             # Zoom acak
    horizontal_flip=True,        # Flip horizontal
    brightness_range=[0.8, 1.2], # Variasi kecerahan
    fill_mode='nearest'          # Pengisian piksel kosong
)

# Generator untuk data TEST (hanya normalisasi, tanpa augmentasi)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Load data training dari folder
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # One-hot encoding untuk multi-kelas
    shuffle=True,
    seed=42
)

# Load data testing dari folder
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False               # Jangan diacak agar evaluasi konsisten
)

print(f"\n  Data Training  : {train_generator.samples} gambar")
print(f"  Data Testing   : {test_generator.samples} gambar")
print(f"  Kelas terdeteksi: {train_generator.class_indices}")

# --- 4. Load MobileNetV2 Pretrained --------------------------
print("\n  [2/6] Memuat model MobileNetV2 pretrained...")

# Load base model MobileNetV2 dengan bobot ImageNet
base_model = MobileNetV2(
    input_shape=(224, 224, 3),  # (tinggi, lebar, channel)
    include_top=False,           # Tidak sertakan head klasifikasi asli
    weights='imagenet'           # Gunakan bobot pretrained ImageNet
)

# FREEZE semua layer base model (tidak dilatih ulang)
base_model.trainable = False

print(f"  Layer di base model  : {len(base_model.layers)}")
print(f"  Status base model    : Frozen (tidak dilatih)")

# --- 5. Tambah Custom Classification Head --------------------
print("\n  [3/6] Membangun arsitektur model...")

# Ambil output dari base model
x = base_model.output

# GlobalAveragePooling untuk meratakan feature maps
x = GlobalAveragePooling2D(name='global_avg_pool')(x)

# Dropout pertama untuk mencegah overfitting
x = Dropout(0.3, name='dropout_1')(x)

# Dense layer dengan 128 neuron
x = Dense(128, activation='relu', name='dense_128')(x)

# Dropout kedua
x = Dropout(0.2, name='dropout_2')(x)

# Output layer - 3 kelas (softmax untuk probabilitas)
predictions = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

# Rangkai menjadi model lengkap
model = Model(inputs=base_model.input, outputs=predictions)

# --- 6. Compile Model ----------------------------------------
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',   # Cocok untuk multi-kelas one-hot
    metrics=['accuracy']
)

# Tampilkan ringkasan model (hanya layer kustom)
print("\n  --- Ringkasan Model (Top Layers) ---")
model.summary()

total_params     = model.count_params()
trainable_params = sum([
    tf.size(w).numpy() for w in model.trainable_weights
])
print(f"\n  Total parameter     : {total_params:,}")
print(f"  Parameter dilatih   : {trainable_params:,}")

# --- 7. Callbacks --------------------------------------------
print("\n  [4/6] Menyiapkan callbacks...")

callbacks = [
    # Simpan model terbaik berdasarkan val_accuracy
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    # Hentikan training jika tidak ada peningkatan selama 5 epoch
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Kurangi learning rate jika val_loss stagnan
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# --- 8. Training Model ---------------------------------------
print("\n  [5/6] Memulai proses training...")
print("  " + "-" * 56)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n  [OK] Training selesai!")

# --- 9. Evaluasi Model ---------------------------------------
print("\n  [6/6] Evaluasi model pada data test...")

# Evaluasi dasar
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"\n  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

# --- 10. Visualisasi Grafik Training -------------------------
print("\n  Membuat grafik akurasi dan loss...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Hasil Training - Klasifikasi Bumbu Dapur Indonesia\n(MobileNetV2 Transfer Learning)',
    fontsize=13, fontweight='bold', y=1.02
)

epochs_ran = range(1, len(history.history['accuracy']) + 1)

# -- Grafik Akurasi --
axes[0].plot(epochs_ran, history.history['accuracy'],
             'b-o', label='Train Accuracy', linewidth=2, markersize=4)
axes[0].plot(epochs_ran, history.history['val_accuracy'],
             'r-o', label='Val Accuracy', linewidth=2, markersize=4)
axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1.05])

# -- Grafik Loss --
axes[1].plot(epochs_ran, history.history['loss'],
             'b-o', label='Train Loss', linewidth=2, markersize=4)
axes[1].plot(epochs_ran, history.history['val_loss'],
             'r-o', label='Val Loss', linewidth=2, markersize=4)
axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('static/images/training_history.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("  [OK] Grafik disimpan: static/images/training_history.png")

# --- 11. Confusion Matrix ------------------------------------
print("\n  Membuat confusion matrix...")

# Reset generator agar prediksi dari awal
test_generator.reset()

# Prediksi seluruh data test
y_pred_probs = model.predict(test_generator, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)   # Kelas prediksi
y_true       = test_generator.classes             # Label asli

# Hitung confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Confusion Matrix\nKlasifikasi Bumbu Dapur Indonesia',
          fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Label Asli', fontsize=11)
plt.xlabel('Label Prediksi', fontsize=11)
plt.tight_layout()
plt.savefig('static/images/confusion_matrix.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("  [OK] Confusion matrix disimpan: static/images/confusion_matrix.png")

# --- 12. Classification Report -------------------------------
print("\n  --- Classification Report ---")
report = classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES
)
print(report)

# Simpan report ke file teks
with open('static/images/classification_report.txt', 'w') as f:
    f.write("Classification Report - Klasifikasi Bumbu Dapur Indonesia\n")
    f.write("=" * 60 + "\n")
    f.write(report)

print("  [OK] Classification report disimpan")

# --- 13. Simpan Model ----------------------------------------
# Model terbaik sudah otomatis disimpan oleh ModelCheckpoint
# Berikut memastikan model akhir juga tersimpan
model.save(MODEL_PATH)

print(f"\n{'=' * 60}")
print(f"  [OK] Model disimpan di: {MODEL_PATH}")
print(f"  [OK] Test Accuracy    : {test_acc*100:.2f}%")
print(f"  Training selesai dengan sukses!")
print(f"{'=' * 60}")
