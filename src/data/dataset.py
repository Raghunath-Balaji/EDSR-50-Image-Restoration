import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class PatchDataset(Dataset):
    def __init__(self, gt_dir, lr_dir, patch_size=64, scale=2, augment_data=True):
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.npy')))
        self.patch_size = patch_size
        self.scale = scale
        self.augment_data = augment_data
        
        if not self.gt_files or not self.lr_files:
            raise ValueError(f"No .npy files found in {gt_dir} or {lr_dir}")

    def __len__(self):
        return len(self.gt_files)
        
    def __getitem__(self, idx):
        gt = np.load(self.gt_files[idx])
        lr = np.load(self.lr_files[idx])
        
        # Ensure 2D images have a channel dimension (C, H, W) for PyTorch
        if len(gt.shape) == 2: gt = np.expand_dims(gt, axis=0)
        elif len(gt.shape) == 3: gt = gt.transpose(2, 0, 1) # HWC to CHW
        if len(lr.shape) == 2: lr = np.expand_dims(lr, axis=0)
        elif len(lr.shape) == 3: lr = lr.transpose(2, 0, 1)
            
        # CRITICAL SPEEDUP: Random Crop (Patch Extraction)
        _, h_lr, w_lr = lr.shape
        if h_lr > self.patch_size and w_lr > self.patch_size:
            x = random.randint(0, w_lr - self.patch_size)
            y = random.randint(0, h_lr - self.patch_size)
            
            lr = lr[:, y:y+self.patch_size, x:x+self.patch_size]
            
            # Ground truth is scaled
            gt_patch_size = self.patch_size * self.scale
            gt_x = x * self.scale
            gt_y = y * self.scale
            gt = gt[:, gt_y:gt_y+gt_patch_size, gt_x:gt_x+gt_patch_size]
            
        # Augmentation
        if self.augment_data:
            if random.random() > 0.5: # Horizontal flip
                lr = np.flip(lr, axis=2).copy()
                gt = np.flip(gt, axis=2).copy()
            if random.random() > 0.5: # Vertical flip
                lr = np.flip(lr, axis=1).copy()
                gt = np.flip(gt, axis=1).copy()
            rot = random.randint(0, 3) # Random rotation
            if rot > 0:
                lr = np.rot90(lr, k=rot, axes=(1, 2)).copy()
                gt = np.rot90(gt, k=rot, axes=(1, 2)).copy()
                
        # Return as PyTorch tensors
        return torch.from_numpy(lr).float(), torch.from_numpy(gt).float()

def get_dataloader(gt_dir, lr_dir, batch_size=8, patch_size=64, scale=2, augment_data=True, num_workers=0):
    dataset = PatchDataset(gt_dir, lr_dir, patch_size, scale, augment_data)
    # Windows native doesn't like num_workers > 0 easily, so defaulting to 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
