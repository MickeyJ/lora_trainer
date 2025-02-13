import os
import shutil
from PIL import Image
import torch


def setup_test_data(test_dir="test_data"):
    """Create a temporary directory with test images and prompts"""
    os.makedirs(test_dir, exist_ok=True)

    # Create test images at SDXL resolution (1024x1024)
    for i in range(3):
        image = Image.new("RGB", (512, 512), color=f"rgb({i*50}, {i*50}, {i*50})")
        image.save(os.path.join(test_dir, f"test_image_{i}.png"))
        with open(os.path.join(test_dir, f"test_image_{i}.txt"), "w") as f:
            f.write(f"test prompt for image {i}")


def test_train_lora_sdxl():
    """Test the SDXL LoRA training pipeline"""
    from train_lora_sdxl import train_lora_sdxl

    test_data_dir = "test_data"
    test_output_dir = "test_output"

    try:
        # Create test data first
        setup_test_data(test_data_dir)

        train_lora_sdxl(
            pretrained_model_path="stabilityai/stable-diffusion-xl-base-1.0",
            train_data_dir=test_data_dir,
            output_dir=test_output_dir,
        )

    finally:
        # Clean up test directories
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)


if __name__ == "__main__":
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Run the test
    test_train_lora_sdxl()
