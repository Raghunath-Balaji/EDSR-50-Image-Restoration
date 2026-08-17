import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.config import Config
from src.data.dataset import get_dataloader
from src.model.edsr import MultiTaskEDSR
from src.training.losses import get_combined_loss

def train():
    device = torch.device(Config.DEVICE)
    print(f"Initializing Training on: {device.type.upper()}")

    # 1. Setup DataLoader
    print("Loading datasets...")
    train_loader = get_dataloader(
        gt_dir=Config.TRAIN_GT_DIR,
        lr_dir=Config.TRAIN_NOISY_DIR,
        batch_size=Config.BATCH_SIZE,
        patch_size=Config.PATCH_SIZE,
        scale=Config.SCALE,
        augment_data=True,
        num_workers=0 # Keep 0 for Windows native stability
    )
    print(f"Dataset ready. Steps per epoch: {len(train_loader)}")

    # 2. Initialize Model
    model = MultiTaskEDSR(
        scale=Config.SCALE,
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        channels=Config.CHANNELS
    ).to(device)

    print("Initializing EDSR (Claude Rewrite)...")

    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = get_combined_loss(Config.LOSS_ALPHA, Config.LOSS_BETA, Config.LOSS_GAMMA)
    
    # Mixed Precision Scaler for RTX Tensor Cores
    scaler = GradScaler()

    # 4. Training Loop
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\nStarting EDSR Training (Mixed Precision Enabled)...")
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        # tqdm progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS}")
        
        for step, (lr_img, gt_img) in enumerate(train_loader):
            lr_img = lr_img.to(device)
            gt_img = gt_img.to(device)
            
            optimizer.zero_grad()
            
            # --- Mixed Precision Forward Pass ---
            with autocast():
                preds = model(lr_img)
                loss = criterion(gt_img, preds)
                
            # --- Scaled Backward Pass ---
            scaler.scale(loss).backward()
            
            # SENIOR ADVICE: Gradient Clipping to prevent SSIM explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            progress_bar.update(1)
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Completed. Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint (Using claude suffix to protect golden weights)
        if epoch % 5 == 0 or epoch == Config.EPOCHS:
            ckpt_path = f"checkpoints/edsr_claude_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    train()
