import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomPICDDataset(Dataset):
    """
    Kelas Dataset kustom untuk dataset PICD_Simple kita.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Direktori utama dataset (misal, './datasets/PICD_Simple').
            transform (callable, optional): Transformasi yang akan diterapkan pada gambar.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = self._make_dataset()

    def _make_dataset(self):
        """Membaca semua path gambar dan menetapkan label integer."""
        samples = []
        print("Membaca dataset...")
        for target_class in self.classes:
            class_idx = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            
            for fname in os.listdir(target_dir):
                path = os.path.join(target_dir, fname)
                item = (path, class_idx)
                samples.append(item)
        print(f"Dataset ditemukan. Total gambar: {len(samples)}. Total kelas: {len(self.classes)}")
        print(f"Pemetaan kelas: {self.class_to_idx}")
        return samples

    def __len__(self):
        """Mengembalikan jumlah total gambar dalam dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Mengambil satu item (gambar dan label) dari dataset.
        
        Args:
            idx (int): Indeks dari item yang akan diambil.
        
        Returns:
            tuple: (gambar, label) di mana gambar adalah tensor dan label adalah integer.
        """
        # 1. Ambil path gambar dan label dari daftar 'samples'
        img_path, label = self.samples[idx]
        
        # 2. Buka gambar menggunakan PIL (Python Imaging Library)
        # Konversi ke 'RGB' untuk memastikan gambar memiliki 3 channel (menghindari gambar grayscale)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error saat membuka gambar {img_path}: {e}")
            # Jika gambar rusak, kita bisa mengembalikan gambar dummy atau gambar lain
            # Untuk sekarang, kita kembalikan gambar hitam dan labelnya
            return torch.zeros(3, 224, 224), label

        # 3. Terapkan transformasi (resize, to tensor, normalize) jika ada
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==============================================================================
# BLOK UNTUK MENGUJI DATALOADER KITA (DENGAN DATA AUGMENTATION)
# ==============================================================================
if __name__ == '__main__':
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224

    # 1. Transformasi untuk TRAINING (dengan augmentasi)
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5), # 50% kemungkinan dibalik horizontal
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    ])

    # 2. Transformasi untuk VALIDASI/TESTING (tanpa augmentasi)
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    ])
    
    dataset_path = './datasets/PICD_Simple'

    # Buat instance dataset menggunakan transformasi training
    print("--- Menguji Dataset dengan Transformasi Training ---")
    train_dataset = CustomPICDDataset(root_dir=dataset_path, transform=train_transforms)
    
    # Buat DataLoader dari train_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    print("\n--- Menguji DataLoader (Training) ---")
    try:
        images, labels = next(iter(train_dataloader))
        print(f"Berhasil mengambil satu batch data training.")
        print(f"Bentuk tensor gambar: {images.shape}")
        # Jika Anda ingin melihat efek augmentasi, Anda bisa menyimpan salah satu gambar dari batch ini
    except Exception as e:
        print(f"Error saat mengambil data dari DataLoader: {e}")