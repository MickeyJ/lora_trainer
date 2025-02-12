import os
import shutil
from PIL import Image
import torch


def setup_test_data(test_dir="test_data"):
    """Create a temporary directory with test images and prompts"""
    os.makedirs(test_dir, exist_ok=True)

    # Create test images at SDXL resolution (1024x1024)
    for i in range(3):
        image = Image.new("RGB", (1024, 1024), color=f"rgb({i*50}, {i*50}, {i*50})")
        image.save(os.path.join(test_dir, f"test_image_{i}.png"))
        with open(os.path.join(test_dir, f"test_image_{i}.txt"), "w") as f:
            f.write(f"test prompt for image {i}")


def test_train_lora_sdxl():
    """Run a minimal test of the SDXL LoRA training pipeline"""
    from train_lora_sdxl import train_lora_sdxl

    test_data_dir = "test_data"
    test_output_dir = "test_output"

    try:
        setup_test_data(test_data_dir)

        train_lora_sdxl(
            pretrained_model_path="C:/Users/18057/Documents/training_models/cyberrealisticXL_v4",
            train_data_dir=test_data_dir,
            output_dir=test_output_dir,
            rank=4,
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            mixed_precision="bf16",  # Use bf16 for better memory efficiency
            save_steps=100,
            debug=True,
        )

        # Verify outputs
        assert os.path.exists(test_output_dir), "Output directory was not created"
        assert os.path.exists(
            os.path.join(test_output_dir, "best_model")
        ), "LoRA weights were not saved"
        assert os.path.exists(
            os.path.join(test_output_dir, "best_model", "adapter_model.safetensors")
        ), "LoRA weights file was not saved"
        # Checkpoint verification (optional)
        assert os.path.exists(
            os.path.join(test_output_dir, "checkpoint-1.pt")
        ), "Checkpoint was not saved"

        print("Test completed successfully!")

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
