import torch
import torch.nn as nn
import torch.nn.functional as F

class EDSRResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res

class MultiTaskEDSR(nn.Module):
    def __init__(self, scale=2, num_filters=64, num_res_blocks=4, channels=1):
        super().__init__()
        self.scale = scale
        
        # --- Shared Encoder ---
        self.head = nn.Conv2d(channels, num_filters, kernel_size=3, padding=1)
        self.shared_blocks = nn.Sequential(*[EDSRResBlock(num_filters) for _ in range(num_res_blocks)])
        self.tail = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        
        # --- Deblur Head ---
        self.deblur_conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.deblur_relu = nn.ReLU(inplace=True)
        self.deblur_res = EDSRResBlock(num_filters)
        self.deblur_conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.deblur_up = nn.PixelShuffle(scale)
        self.deblur_up_conv = nn.Conv2d(num_filters, num_filters * (scale ** 2), kernel_size=3, padding=1)
        
        # --- Denoise Head (Log Domain preprocessing happens in forward) ---
        self.denoise_in_conv = nn.Conv2d(channels, num_filters, kernel_size=3, padding=1)
        self.denoise_conv1 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, padding=1)
        self.denoise_relu = nn.ReLU(inplace=True)
        self.denoise_res = EDSRResBlock(num_filters)
        self.denoise_conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.denoise_up = nn.PixelShuffle(scale)
        self.denoise_up_conv = nn.Conv2d(num_filters, num_filters * (scale ** 2), kernel_size=3, padding=1)
        
        # --- SR Head ---
        self.sr_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.sr_relu = nn.ReLU(inplace=True)
        self.sr_up_conv = nn.Conv2d(num_filters, num_filters * (scale ** 2), kernel_size=3, padding=1)
        self.sr_up = nn.PixelShuffle(scale)
        
        # --- Feature Fusion (Simplified for Speed) ---
        self.fusion_conv = nn.Conv2d(num_filters * 3, num_filters, kernel_size=1)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        # --- Output ---
        self.output_conv = nn.Conv2d(num_filters, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # --- Shared Encoder ---
        shared_feat = self.head(x)
        res = self.shared_blocks(shared_feat)
        res = self.tail(res)
        shared_feat = shared_feat + res
        
        # --- Deblur Head ---
        deblur = self.deblur_conv1(shared_feat)
        deblur = self.deblur_relu(deblur)
        deblur = self.deblur_res(deblur)
        deblur = self.deblur_conv2(deblur)
        deblur = self.deblur_up(self.deblur_up_conv(deblur))
        
        # --- Denoise Head ---
        eps = 1e-6
        log_input = torch.log(torch.clamp(x, min=eps, max=1.0))
        log_feat = self.denoise_in_conv(log_input)
        denoise_concat = torch.cat([shared_feat, log_feat], dim=1)
        denoise = self.denoise_conv1(denoise_concat)
        denoise = self.denoise_relu(denoise)
        denoise = self.denoise_res(denoise)
        denoise = self.denoise_conv2(denoise)
        denoise = self.denoise_up(self.denoise_up_conv(denoise))
        
        # --- SR Head ---
        sr = self.sr_conv(shared_feat)
        sr = self.sr_relu(sr)
        sr = self.sr_up(self.sr_up_conv(sr))
        
        # --- Fusion ---
        fused = torch.cat([deblur, denoise, sr], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.fusion_relu(fused)
        
        # --- Output ---
        residual = self.output_conv(fused)
        up_input = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        restored = up_input + residual
        return torch.sigmoid(restored)
