import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
import einops
import numpy as np
from torchvision.ops import roi_pool

from config_cropping import cfg

# ==============================================================================
# 1. BACKBONE LAMA (VGG-16) - KITA SIMPAN SEBAGAI REFERENSI (TIDAK DIGUNAKAN)
# ==============================================================================
class vgg_base(nn.Module):
    def __init__(self, loadweights=True):
        super(vgg_base, self).__init__()
        vgg = models.vgg16(pretrained=loadweights)
        self.feature1 = nn.Sequential(vgg.features[:6])      # /2
        self.feature2 = nn.Sequential(vgg.features[6:10])    # /4, out channels: 128
        self.feature3 = nn.Sequential(vgg.features[10:17])   # /8, out channels: 256
        self.feature4 = nn.Sequential(vgg.features[17:30])   # /16, out channels: 512

    def forward(self, x):
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)
        return f2, f3, f4

# ==============================================================================
# 2. BACKBONE BARU (MobileNetV2) - INI YANG AKAN KITA GUNAKAN
# ==============================================================================
class MobileNetV2_base(nn.Module):
    def __init__(self, loadweights=True):
        super(MobileNetV2_base, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if loadweights else None
        mobilenet = models.mobilenet_v2(weights=weights)
        
        # Ekstrak lapisan-lapisan dari MobileNetV2
        features = mobilenet.features
        
        # Kita akan mengambil output dari blok-blok yang menghasilkan stride /4, /8, dan /16
        # Stride /4 -> output dari features[3], channels: 24
        self.f2_layer = features[:4]
        # Stride /8 -> output dari features[6], channels: 32
        self.f3_layer = features[4:7]
        # Stride /16 -> output dari features[13], channels: 96
        self.f4_layer = features[7:14]

    def forward(self, x):
        # Kita tidak bisa mengembalikan feature map secara langsung karena modul lain
        # mengharapkan input dari output sebelumnya. Jadi kita jalankan secara sekuensial.
        f2 = self.f2_layer(x)
        f3 = self.f3_layer(f2)
        f4 = self.f4_layer(f3)
        return f2, f3, f4

# ==============================================================================
# SISA KODE - TIDAK ADA PERUBAHAN DI SINI
# ==============================================================================
class CompositionModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CompositionModel, self).__init__()
        # ... (isi kelas ini sama persis seperti aslinya)
        self.comp_types = num_classes # Nanti akan kita ubah menjadi 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            nn.ReLU(True)
        )
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1))
        self.fc_layer = nn.Linear(128, self.comp_types, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f2, f3, f4):
        # ... (isi fungsi ini sama persis seperti aslinya)
        x = self.conv1(f4)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f3
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + f2
        x = self.conv4(x)
        gap = self.GAP(x)
        logits = self.fc_layer(gap)
        conf   = F.softmax(logits, dim=1)
        with torch.no_grad():
            B,C,H,W = x.shape
            w  = self.fc_layer.weight.data # cls_num, channels
            trans_w = einops.repeat(w, 'n c -> b n c', b=B)
            trans_x = einops.rearrange(x, 'b c h w -> b c (h w)')
            cam = torch.matmul(trans_w, trans_x) # b n hw
            cam = cam - cam.min(dim=-1)[0].unsqueeze(-1)
            cam = cam / (cam.max(dim=-1)[0].unsqueeze(-1) + 1e-12)
            cam = einops.rearrange(cam, 'b n (h w) -> b n h w', h=H, w=W)
            kcm = torch.sum(conf[:,:,None,None] * cam, dim=1, keepdim=True)
            kcm = F.interpolate(kcm, scale_factor=4, mode='bilinear', align_corners=True)
        return logits, kcm

# ... (Kelas CroppingModel, generate_anchors, shift, PostProcess, ComClassifier tetap sama persis)
class CroppingModel(nn.Module):
    # ... (Sama seperti asli)
    def __init__(self, anchor_stride):
        super(CroppingModel, self).__init__()
        self.anchor_stride = anchor_stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        out_channel = int((16 / anchor_stride)**2 * 4)
        self.output = nn.Conv2d(256, out_channel, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.output(x)
        return out
def generate_anchors(anchor_stride):
    # ... (Sama seperti asli)
    assert anchor_stride <= 16, 'not implement for anchor_stride{} > 16'.format(anchor_stride)
    P_h = np.array([2+i*4 for i in range(16 // anchor_stride)])
    P_w = np.array([2+i*4 for i in range(16 // anchor_stride)])
    num_anchors = len(P_h) * len(P_h)
    anchors = torch.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = float (P_w[j])
            anchors[k,0] = float (P_h[i])
            k += 1
    return anchors
def shift(shape, stride, anchors):
    # ... (Sama seperti asli)
    shift_w = torch.arange(0, shape[0]) * stride
    shift_h = torch.arange(0, shape[1]) * stride
    shift_w, shift_h = torch.meshgrid([shift_w, shift_h], indexing='ij')
    shifts  = torch.stack([shift_w, shift_h], dim=-1)
    trans_anchors = einops.rearrange(anchors, 'a c -> a 1 1 c')
    trans_shifts  = einops.rearrange(shifts,  'h w c -> 1 h w c')
    all_anchors   = trans_anchors + trans_shifts
    return all_anchors
class PostProcess(nn.Module):
    # ... (Sama seperti asli)
    def __init__(self, anchor_stride, image_size):
        super(PostProcess, self).__init__()
        self.num_anchors = (16 // anchor_stride) ** 2
        anchors = generate_anchors(anchor_stride)
        feat_shape  = (image_size[0] // 16, image_size[1] // 16)
        all_anchors = shift(feat_shape, 16, anchors)
        all_anchors = all_anchors.float().unsqueeze(0)
        self.upscale_factor = int(np.sqrt(self.num_anchors))
        anchors_x   = F.pixel_shuffle(all_anchors[...,0], upscale_factor=self.upscale_factor)
        anchors_y   = F.pixel_shuffle(all_anchors[...,1], upscale_factor=self.upscale_factor)
        all_anchors = torch.stack([anchors_x, anchors_y], dim=-1).squeeze(1)
        self.register_buffer('all_anchors', all_anchors)
        grid_x = (all_anchors[...,0] - image_size[0]/2) / (image_size[0]/2)
        grid_y = (all_anchors[...,1] - image_size[1]/2) / (image_size[1]/2)
        grid   = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer('grid', grid)
    def forward(self, offsets, kcm):
        offsets = einops.rearrange(offsets, 'b (n c) h w -> b n h w c', n=self.num_anchors, c=4)
        coords  = [F.pixel_shuffle(offsets[...,i], upscale_factor=self.upscale_factor) for i in range(4)]
        offsets = torch.stack(coords, dim=-1).squeeze(1)
        regression = torch.zeros_like(offsets)
        regression[...,0::2] = offsets[..., 0::2] + self.all_anchors[...,0:1]
        regression[...,1::2] = offsets[..., 1::2] + self.all_anchors[...,1:2]
        trans_grid  = einops.repeat(self.grid, '1 h w c -> b h w c', b=offsets.shape[0])
        sample_kcm  = F.grid_sample(kcm, trans_grid, mode='bilinear', align_corners=True)
        reg_weight  = F.softmax(sample_kcm.flatten(1), dim=1).unsqueeze(-1)
        regression  = einops.rearrange(regression, 'b h w c -> b (h w) c')
        weighted_reg = torch.sum(reg_weight * regression, dim=1)
        return weighted_reg
class ComClassifier(nn.Module):
    # ... (Sama seperti asli, TAPI kita akan membuat model CACNet baru yang lebih fleksibel)
    def __init__(self, loadweights=True):
        super(ComClassifier, self).__init__()
        self.backbone   = vgg_base(loadweights=loadweights)
        self.composition_module = CompositionModel()

    def forward(self, x, only_classify=False):
        f2,f3,f4 = self.backbone(x)
        logits,kcm = self.composition_module(f2,f3,f4)
        return logits,kcm
        
# ==============================================================================
# 3. KELAS UTAMA CACNet - INI YANG KITA MODIFIKASI SECARA SIGNIFIKAN
# ==============================================================================
class CACNet(nn.Module):
    def __init__(self, backbone_type='mobilenetv2', num_classes=3, loadweights=True):
        super(CACNet, self).__init__()
        anchor_stride = 8
        image_size = cfg.image_size
        
        print(f"Menginisialisasi CACNet dengan backbone: {backbone_type}, kelas: {num_classes}")

        if backbone_type == 'mobilenetv2':
            self.backbone = MobileNetV2_base(loadweights=loadweights)
            # Definisikan lapisan adaptasi untuk menyesuaikan jumlah channel
            # MobileNetV2 outputs: f2 (24), f3 (32), f4 (96)
            # CompositionModel expects: f2 (128), f3 (256), f4 (512)
            self.adapter_f2 = nn.Conv2d(24, 128, kernel_size=1)
            self.adapter_f3 = nn.Conv2d(32, 256, kernel_size=1)
            self.adapter_f4 = nn.Conv2d(96, 512, kernel_size=1)
        elif backbone_type == 'vgg16':
            self.backbone  = vgg_base(loadweights=loadweights)
            # Untuk VGG16, tidak perlu adaptasi, kita gunakan lapisan identitas
            self.adapter_f2 = nn.Identity()
            self.adapter_f3 = nn.Identity()
            self.adapter_f4 = nn.Identity()
        else:
            raise NotImplementedError(f"Backbone '{backbone_type}' belum diimplementasikan.")

        # Modul lainnya tetap sama
        self.composition_module = CompositionModel(num_classes=num_classes)
        self.cropping_module = CroppingModel(anchor_stride)
        self.post_process = PostProcess(anchor_stride, image_size)

    def forward(self, im, only_classify=False):
        # 1. Dapatkan feature map dari backbone
        f2_raw, f3_raw, f4_raw = self.backbone(im)
        
        # 2. Lewatkan feature map melalui lapisan adaptasi
        f2 = self.adapter_f2(f2_raw)
        f3 = self.adapter_f3(f3_raw)
        f4 = self.adapter_f4(f4_raw)
        
        # 3. Sisa dari alur kerja tetap sama persis
        logits, kcm = self.composition_module(f2, f3, f4)
        if only_classify:
            return logits, kcm
        else:
            # PENTING: Cropping module menggunakan f4 *sebelum* adaptasi jika backbone VGG,
            # tapi karena channel outputnya berbeda, kita akan gunakan f4 yang sudah diadaptasi (512 channels)
            # agar konsisten.
            offsets = self.cropping_module(f4)
            box = self.post_process(offsets, kcm)
            return logits, kcm, box
