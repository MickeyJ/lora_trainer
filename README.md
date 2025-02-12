# LoRA Training Script (WIP)

This repository contains a script for training LoRA (Low-Rank Adaptation) models.

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```


2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train a LoRA model, use the `train_lora.py` script:
```bash
python train_lora.py --base_model <base-model-name> \
--dataset <dataset-path> \
--output_dir <output-directory>
```

### Key Parameters

- `base_model`: The name or path of the base model to adapt (e.g., "facebook/opt-1.3b")
- `dataset`: Path to your training dataset
- `output_dir`: Directory where the trained LoRA weights will be saved
