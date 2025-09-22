import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Impor modul-modul kustom kita
from model import CompositionGuidanceNet
from CustomPICDDataset import CustomPICDDataset

# ==============================================================================
# 1. KONFIGURASI PELATIHAN
# ==============================================================================
# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32 # Anda bisa turunkan ke 16 jika mengalami 'Out of Memory' di GPU lokal
NUM_EPOCHS = 50 # Jumlah total epoch pelatihan
NUM_CLASSES = 3

# Paths
DATASET_DIR = './datasets/PICD_Simple'
OUTPUT_DIR = './experiments/composition_model' # Folder untuk menyimpan hasil
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# Konfigurasi Data Augmentation
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# ==============================================================================
# 2. FUNGSI UTAMA PELATIHAN
# ==============================================================================
def train():
    # --- Setup Awal ---
    # Pilih perangkat (GPU jika tersedia, jika tidak CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    # Buat direktori output jika belum ada
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Inisialisasi TensorBoard writer
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"Log TensorBoard akan disimpan di: {LOG_DIR}")

    # --- Persiapan Dataset ---
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    ])

    # Buat dataset penuh
    full_dataset = CustomPICDDataset(root_dir=DATASET_DIR)

    # Bagi dataset menjadi training (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Terapkan transformasi yang sesuai untuk setiap set
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    
    print(f"Ukuran dataset: {len(full_dataset)} gambar")
    print(f"  - Training set: {len(train_dataset)} gambar")
    print(f"  - Validation set: {len(val_dataset)} gambar")

    # Buat DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Inisialisasi Model, Loss, dan Optimizer ---
    model = CompositionGuidanceNet(num_classes=NUM_CLASSES, loadweights=True).to(device)
    
    criterion = nn.CrossEntropyLoss() # Loss function untuk klasifikasi
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Optimizer

    # --- Loop Pelatihan Utama ---
    best_val_accuracy = 0.0 # Untuk melacak akurasi terbaik

    for epoch in range(NUM_EPOCHS):
        # --- Fase Training ---
        model.train()
        running_loss = 0.0
        correct_preds_train = 0
        total_preds_train = 0

        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        for images, labels in progress_bar_train:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits.data, 1)
            total_preds_train += labels.size(0)
            correct_preds_train += (predicted == labels).sum().item()
            
            progress_bar_train.set_postfix(loss=loss.item())

        # --- Fase Validation ---
        model.eval()
        val_loss = 0.0
        correct_preds_val = 0
        total_preds_val = 0
        
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
            for images, labels in progress_bar_val:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits.data, 1)
                total_preds_val += labels.size(0)
                correct_preds_val += (predicted == labels).sum().item()

        # --- Logging dan Checkpointing ---
        epoch_loss_train = running_loss / len(train_dataset)
        epoch_acc_train = correct_preds_train / total_preds_train
        epoch_loss_val = val_loss / len(val_dataset)
        epoch_acc_val = correct_preds_val / total_preds_val

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> "
              f"Train Loss: {epoch_loss_train:.4f}, Train Acc: {epoch_acc_train:.4f} | "
              f"Val Loss: {epoch_loss_val:.4f}, Val Acc: {epoch_acc_val:.4f}")

        # Log ke TensorBoard
        writer.add_scalar('Loss/train', epoch_loss_train, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc_train, epoch)
        writer.add_scalar('Loss/val', epoch_loss_val, epoch)
        writer.add_scalar('Accuracy/val', epoch_acc_val, epoch)

        # Simpan model jika akurasi validasi membaik
        if epoch_acc_val > best_val_accuracy:
            best_val_accuracy = epoch_acc_val
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Model terbaik baru disimpan di {checkpoint_path} (Akurasi: {best_val_accuracy:.4f})")

    writer.close()
    print("Pelatihan selesai.")

# ==============================================================================
# 3. BLOK EKSEKUSI
# ==============================================================================
if __name__ == '__main__':
    train()