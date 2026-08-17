import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.config import Config
from src.model.edsr import MultiTaskEDSR

def evaluate():
    device = torch.device(Config.DEVICE)
    print(f"Loading EDSR Model on {device}...")
    
    # Initialize the architecture
    model = MultiTaskEDSR(
        scale=Config.SCALE,
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        channels=Config.CHANNELS
    ).to(device)
    
    # Load the Epoch 50 weights from Claude rewrite
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True))
    model.eval()
    
    gt_files = sorted(glob.glob(os.path.join(Config.TRAIN_GT_DIR, '*.npy')))
    lr_files = sorted(glob.glob(os.path.join(Config.TRAIN_NOISY_DIR, '*.npy')))
    
    # Evaluate on the first 100 images
    num_eval = min(100, len(gt_files))
    
    total_psnr = 0.0
    total_ssim = 0.0
    
    print(f"Running inference on {num_eval} images...")
    
    with torch.no_grad():
        for i in range(num_eval):
            gt = np.load(gt_files[i])
            lr = np.load(lr_files[i])
            
            # Format PyTorch Tensors
            lr_t = torch.from_numpy(lr).float()
            if len(lr_t.shape) == 2: lr_t = lr_t.unsqueeze(0).unsqueeze(0)
            elif len(lr_t.shape) == 3: lr_t = lr_t.permute(2, 0, 1).unsqueeze(0)
            lr_t = lr_t.to(device)
            # SENIOR ADVICE: Test-Time Augmentation (TTA)
            
            # Forward Pass 1: Original
            pred_t1 = model(lr_t)
            
            # Forward Pass 2: Flipped horizontally
            lr_t_flip = torch.flip(lr_t, [3])
            pred_t2_flip = model(lr_t_flip)
            pred_t2 = torch.flip(pred_t2_flip, [3])
            
            # Ensemble (Average) the predictions
            pred_t = (pred_t1 + pred_t2) / 2.0
            
            # Convert back to numpy for official metric calculation
            pred = pred_t.squeeze().cpu().numpy()
            
            # Ensure shape matches ground truth for exact comparison
            if pred.shape != gt.shape:
                if len(gt.shape) == 3:
                    pred = np.expand_dims(pred, axis=-1)
            
            # Dynamically calculate the actual data range (e.g. 1.5 - 0.0)
            drange = float(gt.max() - gt.min())
            
            # Calculate Metrics using the TRUE data range
            cur_psnr = psnr(gt, pred, data_range=drange)
            cur_ssim = ssim(gt, pred, data_range=drange, channel_axis=-1 if len(gt.shape) == 3 else None)
            
            total_psnr += cur_psnr
            total_ssim += cur_ssim
            
    avg_psnr = total_psnr / num_eval
    avg_ssim = total_ssim / num_eval
    
    print("\n" + "="*40)
    print("FINAL EVALUATION SCORES")
    print("="*40)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f} ({(avg_ssim*100):.2f}%)")
    print("="*40)

if __name__ == "__main__":
    evaluate()
