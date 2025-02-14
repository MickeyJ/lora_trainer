import os
import shutil
from PIL import Image
import torch
import tempfile
import numpy as np


def setup_test_data(test_dir="test_data", use_random_images=False, image_count=10):
    """Create a temporary directory with test images and prompts"""
    os.makedirs(test_dir, exist_ok=True)

    if use_random_images:
        # Create test images to match recommended dataset size
        for i in range(image_count):
            img = Image.fromarray(
                np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            )
            img.save(os.path.join(test_dir, f"test_image_{i}.jpg"))
            with open(os.path.join(test_dir, f"test_image_{i}.txt"), "w") as f:
                f.write("A photo of sks person, high quality portrait")
    else:
        # Use existing test images
        if not os.path.exists("test_image_data"):
            raise ValueError("test_image_data directory not found")


def cleanup_test_data(test_dir="test_output"):
    """Clean up test directories after running"""
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def test_train_lora_sdxl():
    """Test the SDXL LoRA training pipeline"""
    from train_lora_sdxl import train_lora_sdxl

    try:
        # Create test data that matches recommended real training
        setup_test_data(use_random_images=True, image_count=15)  # Will create 15 test images

        train_lora_sdxl(
            train_data_dir="test_data",  # Use the random test images
            output_dir="test_output",
            num_epochs=100,  # More epochs for fewer images
            batch_size=1,
            image_size=512,
            learning_rate=1e-5,  # Lower for stability
            gradient_accumulation_steps=4,
        )
    finally:
        cleanup_test_data("test_data")  # Clean up test input
        cleanup_test_data("test_output")  # Clean up test output


if __name__ == "__main__":
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Run the test
    test_train_lora_sdxl()
