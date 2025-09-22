import os
import shutil
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# KONFIGURASI (SAMA SEPERTI SEBELUMNYA)
# ==============================================================================
SOURCE_CSV_PATH = r'D:\Kuliah\File Pelajaran\Semester 8\TA\Dataset\PICD_ImageComposition-main\labels_PICD.csv'
SOURCE_IMAGE_DIR = r'D:\Kuliah\File Pelajaran\Semester 8\TA\Dataset\single_labels'
TARGET_BASE_DIR = r'./datasets/PICD_Simple'
CATEGORY_MAPPING = {
    'rule_of_thirds': [1, 2, 9],
    'centered': [3, 4],
    'horizontal': [7, 8, 10]
}

# ==============================================================================
# FUNGSI UTAMA (FINAL & DIPERBAIKI)
# ==============================================================================
def prepare_dataset():
    print("Memulai proses persiapan dataset...")

    if not os.path.exists(SOURCE_CSV_PATH):
        print(f"Error: File CSV tidak ditemukan di '{SOURCE_CSV_PATH}'")
        return
    if not os.path.exists(SOURCE_IMAGE_DIR):
        print(f"Error: Direktori gambar sumber tidak ditemukan di '{SOURCE_IMAGE_DIR}'")
        return
        
    # Hapus dan buat ulang direktori target untuk memastikan kebersihan
    if os.path.exists(TARGET_BASE_DIR):
        print(f"Menghapus direktori lama: '{TARGET_BASE_DIR}'")
        shutil.rmtree(TARGET_BASE_DIR)

    print(f"Membuat direktori target baru di: '{TARGET_BASE_DIR}'")
    for category_name in CATEGORY_MAPPING.keys():
        target_path = os.path.join(TARGET_BASE_DIR, category_name)
        os.makedirs(target_path, exist_ok=True)
    
    print(f"Membaca file anotasi: {SOURCE_CSV_PATH}")
    try:
        df = pd.read_csv(SOURCE_CSV_PATH)
        if 'folder_nar' in df.columns:
            df = df.rename(columns={'folder_nar': 'folder_name'})
    except Exception as e:
        print(f"Error saat membaca file CSV: {e}")
        return

    reverse_mapping = {}
    for target_cat, source_ids in CATEGORY_MAPPING.items():
        for source_id in source_ids:
            reverse_mapping[source_id] = target_cat
            
    print("\nMemproses gambar dan menyalin file (ini mungkin memakan waktu)...")
    copied_files_count = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Menyalin gambar"):
        # PERBAIKAN UTAMA #1: Konversi category_id ke integer
        try:
            category_id = int(row['category_id'])
        except (ValueError, TypeError):
            # Lewati baris jika category_id tidak valid
            continue

        folder_name = row['folder_name']
        img_id = row['img_id']

        target_category = reverse_mapping.get(category_id)

        if target_category:
            source_path = os.path.join(SOURCE_IMAGE_DIR, str(folder_name), img_id)
            dest_path = os.path.join(TARGET_BASE_DIR, target_category, img_id)

            # PERBAIKAN UTAMA #2: Hanya salin jika file benar-benar ada
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)
                copied_files_count += 1

    print("\n==========================================================")
    print(f"Proses selesai!")
    print(f"Berhasil menyalin {copied_files_count} gambar ke direktori '{TARGET_BASE_DIR}'.")
    print("==========================================================")


# ==============================================================================
# EKSEKUSI SKRIP
# ==============================================================================
if __name__ == '__main__':
    prepare_dataset()