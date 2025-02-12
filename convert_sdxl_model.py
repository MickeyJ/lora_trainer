from diffusers import StableDiffusionXLPipeline
import os

# Input model path (use raw string for Windows path)
model_path = r"C:\Users\18057\Documents\ComfyUI_windows_portable\ComfyUI\models\checkpoints\SDXL\cyberrealisticXL_v4.safetensors"

# Output directory (no file extension needed - it's a directory)
output_dir = r"C:\Users\18057\Documents\training_models\cyberrealisticXL_v4"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert the model
print(f"Converting model from {model_path}")
pipeline = StableDiffusionXLPipeline.from_single_file(model_path, use_safetensors=True)

# Save in diffusers format
print(f"Saving model to {output_dir}")
pipeline.save_pretrained(output_dir)

print("Conversion completed!")
