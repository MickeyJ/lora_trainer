import os
import shutil
from PIL import Image
import torch


def setup_test_data(test_dir="test_data"):
    """Create a temporary directory with test images and prompts"""
    os.makedirs(test_dir, exist_ok=True)

    # Create larger test images (512x512) to match training expectations
    for i in range(3):
        image = Image.new("RGB", (512, 512), color=f"rgb({i*50}, {i*50}, {i*50})")
        image.save(os.path.join(test_dir, f"test_image_{i}.png"))
        with open(os.path.join(test_dir, f"test_image_{i}.txt"), "w") as f:
            f.write(f"test prompt for image {i}")


def cleanup_test_data(test_dir="test_data"):
    """Remove temporary test directory and its contents"""
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def test_train_lora():
    """Run a minimal test of the LoRA training pipeline"""
    from train_lora import train_lora

    test_data_dir = "test_data"
    test_output_dir = "test_output"

    try:
        setup_test_data(test_data_dir)

        train_lora(
            image_dir=test_data_dir,
            output_dir=test_output_dir,
            instance_prompt="test person photo",
            num_epochs=2,
            batch_size=1,
            learning_rate=1e-4,
            rank=4,
            validation_split=0.1,
            checkpoint_freq=1,
        )

        # Verify outputs
        assert os.path.exists(test_output_dir), "Output directory was not created"
        assert os.path.exists(
            os.path.join(test_output_dir, "checkpoint-1.pt")
        ), "Checkpoint was not saved"
        assert os.path.exists(
            os.path.join(test_output_dir, "best_model")
        ), "Best model was not saved"

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
    test_train_lora()
