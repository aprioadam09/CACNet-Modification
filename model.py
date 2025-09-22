# File: model.py
# Deskripsi: Arsitektur final untuk Composition Guidance Net kita.
# Ini adalah versi CACNet yang dimodifikasi dengan backbone MobileNetV2 dan
# tanpa cabang cropping.

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import einops

# ==============================================================================
# 1. BACKBONE (MobileNetV2)
# ==============================================================================
class MobileNetV2_base(nn.Module):
    def __init__(self, loadweights=True):
        super(MobileNetV2_base, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if loadweights else None
        mobilenet = models.mobilenet_v2(weights=weights)
        
        features = mobilenet.features
        
        self.f2_layer = features[:4]  # Stride /4
        self.f3_layer = features[4:7] # Stride /8
        self.f4_layer = features[7:14]# Stride /16

    def forward(self, x):
        f2 = self.f2_layer(x)
        f3 = self.f3_layer(f2)
        f4 = self.f4_layer(f3)
        return f2, f3, f4

# ==============================================================================
# 2. MODUL KOMPOSISI (Sama seperti yang kita modifikasi sebelumnya)
# ==============================================================================
class CompositionModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CompositionModel, self).__init__()
        self.comp_types = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU(True))
        self.GAP = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.fc_layer = nn.Linear(128, self.comp_types, bias=True)

    def forward(self, f2, f3, f4):
        x = self.conv1(f4)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f3
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f2
        x = self.conv4(x)
        gap = self.GAP(x)
        logits = self.fc_layer(gap)
        
        # Logika KCM (Heatmap)
        conf   = F.softmax(logits, dim=1)
        with torch.no_grad():
            B,C,H,W = x.shape
            w  = self.fc_layer.weight.data
            trans_w = einops.repeat(w, 'n c -> b n c', b=B)
            trans_x = einops.rearrange(x, 'b c h w -> b c (h w)')
            cam = torch.matmul(trans_w, trans_x)
            cam = cam - cam.min(dim=-1, keepdim=True)[0]
            cam = cam / (cam.max(dim=-1, keepdim=True)[0] + 1e-12)
            cam = einops.rearrange(cam, 'b n (h w) -> b n h w', h=H, w=W)
            kcm = torch.sum(conf.unsqueeze(-1).unsqueeze(-1) * cam, dim=1, keepdim=True)
            kcm = F.interpolate(kcm, scale_factor=4, mode='bilinear', align_corners=True)
            
        return logits, kcm

# ==============================================================================
# 3. ARSITEKTUR UTAMA (BERSIH & FOKUS) - Kita beri nama baru
# ==============================================================================
class CompositionGuidanceNet(nn.Module):
    def __init__(self, num_classes=3, loadweights=True):
        super(CompositionGuidanceNet, self).__init__()
        
        print(f"Menginisialisasi CompositionGuidanceNet dengan backbone: mobilenetv2, kelas: {num_classes}")

        # Bagian Backbone
        self.backbone = MobileNetV2_base(loadweights=loadweights)
        
        # Bagian Lapisan Adaptasi
        self.adapter_f2 = nn.Conv2d(24, 128, kernel_size=1)
        self.adapter_f3 = nn.Conv2d(32, 256, kernel_size=1)
        self.adapter_f4 = nn.Conv2d(96, 512, kernel_size=1)

        # Bagian Modul Komposisi
        self.composition_module = CompositionModel(num_classes=num_classes)
        
        # TIDAK ADA LAGI CroppingModel atau PostProcess

    def forward(self, im):
        # Alur kerja yang sudah disederhanakan
        f2_raw, f3_raw, f4_raw = self.backbone(im)
        
        f2 = self.adapter_f2(f2_raw)
        f3 = self.adapter_f3(f3_raw)
        f4 = self.adapter_f4(f4_raw)
        
        # Langsung dapatkan logits dan KCM
        logits, kcm = self.composition_module(f2, f3, f4)
        
        return logits, kcm # Hanya mengembalikan 2 item, bukan 3