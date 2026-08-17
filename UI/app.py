import gradio as gr
import torch
import numpy as np
from PIL import Image
import sys
import os

# Ensure we can import from the parent directory (EDSR50)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.model.edsr import MultiTaskEDSR
from src.config import Config

# Initialize Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading UI with model on {device}...")

# Load Model Architecture
model = MultiTaskEDSR(
    scale=Config.SCALE,
    num_filters=Config.NUM_FILTERS,
    num_res_blocks=Config.NUM_RES_BLOCKS,
    channels=Config.CHANNELS
).to(device)

# Load Golden Weights
model_path = os.path.join(parent_dir, "checkpoints", "best_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

def restore_image(input_image):
    if input_image is None:
        return None
        
    # Convert incoming PIL Image to Grayscale Numpy Array
    img = input_image.convert("L")
    img_arr = np.array(img).astype(np.float32) / 255.0
    
    # Format for Model (B, C, H, W)
    img_t = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run Inference
    with torch.no_grad():
        pred_t = model(img_t)
        
    # Convert back to PIL Image
    pred_arr = pred_t.squeeze().cpu().numpy()
    pred_arr = np.clip(pred_arr, 0.0, 1.0) * 255.0
    return Image.fromarray(pred_arr.astype(np.uint8), mode="L")

# Create Gradio Web Interface
interface = gr.Interface(
    fn=restore_image,
    inputs=gr.Image(type="pil", label="Upload Noisy LR Image (256x256)"),
    outputs=gr.Image(type="pil", label="Restored HR Image (512x512)"),
    title="EDSR Speckle Noise Super-Resolution",
    description="Upload a 256x256 image corrupted by multiplicative speckle noise. Our decoupled EDSR model will mathematically invert the noise domain and safely upscale the physical structure to a clean 512x512."
)

if __name__ == "__main__":
    interface.launch()
