import os
import shutil
from PIL import Image
import torch


def setup_test_data(test_dir="test_data"):
    """Create a temporary directory with test images and prompts"""

    # Create test directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)

    # Create a few test images and corresponding text files
    for i in range(3):
        # Create a small test image (64x64 RGB)
        image = Image.new("RGB", (64, 64), color=f"rgb({i*50}, {i*50}, {i*50})")
        image.save(os.path.join(test_dir, f"test_image_{i}.png"))

        # Create corresponding text file
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
        # Setup test environment
        setup_test_data(test_data_dir)

        # Run training with minimal parameters
        train_lora(
            image_dir=test_data_dir,
            output_dir=test_output_dir,
            instance_prompt="test person photo",
            num_epochs=2,  # Minimal epochs for testing
            batch_size=1,
            learning_rate=1e-4,
            rank=4,
            validation_split=0.1,
            checkpoint_freq=1,
        )

        # Basic checks
        assert os.path.exists(test_output_dir), "Output directory was not created"
        assert os.path.exists(
            os.path.join(test_output_dir, "checkpoint-1.pt")
        ), "Checkpoint was not saved"

        print("Test completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise

    finally:
        # Cleanup
        cleanup_test_data(test_data_dir)
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir)


if __name__ == "__main__":
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Run the test
    test_train_lora()
