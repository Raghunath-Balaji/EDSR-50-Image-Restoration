import torch

class Config:
    # Model Hyperparameters
    SCALE = 2
    NUM_FILTERS = 64
    NUM_RES_BLOCKS = 16  # Standard EDSR capacity for high accuracy
    CHANNELS = 1

    # Loss Weights
    LOSS_ALPHA = 0.5  # Charbonnier
    LOSS_BETA = 0.3   # SSIM
    LOSS_GAMMA = 0.1  # Sobel Edge

    # Training Config
    BATCH_SIZE = 16 
    EPOCHS = 50 
    LEARNING_RATE = 1e-4 
    PATCH_SIZE = 128 # 128x128 patches to capture structural context for SSIM

    # Data paths (Pointing back to the root dataset folder so we don't duplicate gigabytes of images)
    TRAIN_GT_DIR = '../train/GT'
    TRAIN_NOISY_DIR = '../train/NoisyLR'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
