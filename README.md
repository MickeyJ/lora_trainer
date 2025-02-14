# LoRA Training Script (WIP)

This repository contains scripts for training LoRA (Low-Rank Adaptation) model for SDXL.

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate conda environment:
```bash
conda create -n lora-training python=3.10.11 pip
conda activate lora-training
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training SDXL LoRA

To train a LoRA model for Stable Diffusion XL:
```bash
python train_lora_sdxl.py \
    --train_data_dir <path-to-images> \
    --output_dir <output-directory> \
    --pretrained_model_path "stabilityai/stable-diffusion-xl-base-1.0"
```

### Common Parameters

- `dataset`: Path to directory containing training images (and optional .txt files with prompts)
- `output_dir`: Directory where the trained LoRA weights will be saved
- `image_size`: Size to resize images during training (default: 512)
- `gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 8)
- `num_epochs`: Number of training epochs (default: 5, more for complex concepts)
- `learning_rate`: Learning rate for training (default: 1e-4)
- `rank`: Rank of LoRA update matrices (default: 4)
- `validation_split`: Fraction of data to use for validation (default: 0.1)
- `checkpoint_freq`: Save checkpoint every N epochs (default: 10)

## Dataset Format

The training scripts expect a directory containing:
- Image files (PNG, JPG, or JPEG)
- Optional text files with the same name as the images but .txt extension
  - If a text file exists, its content will be used as the prompt for that image
  - If no text file exists, the `instance_prompt` will be used

Example:
```
dataset/
├── image1.jpg
├── image1.txt
├── image2.png
└── image3.jpg
```

## Requirements

- Python 3.10.11
- CUDA-capable GPU (recommended)
- See requirements.txt for full package list

## Image Sizes and Memory Requirements

- Input images can be any size - they will be automatically resized
- Default training size is 512x512 (good for 8GB VRAM)
- Can be increased with --image_size parameter:
  - 512x512: ~8GB VRAM
  - 768x768: ~12GB VRAM
  - 1024x1024: ~16GB VRAM
- Aspect ratio is preserved during resizing

## Testing

To run the test suite:
```bash
# Test SD 1.5 training
python test_train_lora.py

# Test SDXL training
python test_train_lora_sdxl.py
```

## Model Download

The script will automatically download the SDXL base model (~12GB) from Hugging Face on first run. Make sure you have:
- At least 20GB of free disk space
- A stable internet connection for the initial download
