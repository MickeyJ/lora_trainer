# LoRA Training Scripts (WIP)

This repository contains scripts for training LoRA (Low-Rank Adaptation) models for both Stable Diffusion 1.5 and SDXL.

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate conda environment:
```bash
conda create -n lora-training python=3.10 pip
conda activate lora-training
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training SD 1.5 LoRA

To train a LoRA model for Stable Diffusion 1.5:
```bash
python train_lora.py \
    --dataset <path-to-images> \
    --output_dir <output-directory> \
    --base_model "runwayml/stable-diffusion-v1-5"
```

### Training SDXL LoRA

To train a LoRA model for Stable Diffusion XL:
```bash
python train_lora_sdxl.py \
    --dataset <path-to-images> \
    --output_dir <output-directory> \
    --base_model "stabilityai/stable-diffusion-xl-base-1.0"
```

### Common Parameters

- `dataset`: Path to directory containing training images (and optional .txt files with prompts)
- `output_dir`: Directory where the trained LoRA weights will be saved
- `base_model`: The name or path of the base model to adapt
- `instance_prompt`: Default text prompt for training (default: "photo of sks person")
- `batch_size`: Images per batch (default: 1)
- `num_epochs`: Number of training epochs (default: 100)
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

- Python 3.10
- CUDA-capable GPU (recommended)
- See requirements.txt for full package list

## Image Sizes

- For SD 1.5: Images are automatically resized to 512x512
- For SDXL: Images are automatically resized to 1024x1024

## Testing

To run the test suite:
```bash
# Test SD 1.5 training
python test_train_lora.py

# Test SDXL training
python test_train_lora_sdxl.py
```
