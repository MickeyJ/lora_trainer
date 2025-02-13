import os
import shutil

# More aggressive CUDA memory settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:32,"  # Keep our existing setting
    "expandable_segments:True,"  # Allow memory segments to expand
    "garbage_collection_threshold:0.6,"  # More aggressive than before
    "roundup_power2:True"  # Might help with fragmentation
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable symlinks warning
# Add their recommended number of CPU threads
os.environ["OMP_NUM_THREADS"] = "8"  # They suggest 8 CPU threads

import torch
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)
from peft import LoraConfig, get_peft_model
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.cuda
import gc
from bitsandbytes.optim import AdamW8bit

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_lora_sdxl(
    pretrained_model_path: str,
    train_data_dir: str,
    output_dir: str,
    num_epochs: int = 5,  # They recommend 5 epochs
    batch_size: int = 1,  # They suggest reducing to 1 if having memory issues
    gradient_accumulation_steps: int = 8,  # Keep our increased value
    learning_rate: float = 3e-4,  # They use 0.0003
):
    """Train a LoRA adapter for SDXL"""

    logger.info("Starting SDXL LoRA training...")

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16",
    )

    # Add gradient clipping after initialization
    accelerator.clip_grad_norm_ = 1.0  # Enable gradient clipping

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory created at {output_dir}")

    # Load the tokenizers and models
    logger.info("Loading tokenizers...")
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer_2",
    )

    logger.info("Loading models...")
    # Load models without device mapping or offloading
    text_encoder_one = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        use_safetensors=True,  # Add this to ensure proper loading
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    # Load VAE in bfloat16 instead of float32
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,  # Changed from float32
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )

    # Let accelerator handle all models except VAE
    unet, text_encoder_one, text_encoder_two = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two
    )
    # Handle VAE separately
    vae = vae.to(accelerator.device)

    # Enable memory efficient attention
    if hasattr(unet.config, "use_memory_efficient_attention") and hasattr(
        unet.config, "use_sdpa"
    ):
        unet.config.use_memory_efficient_attention = True
        unet.config.use_sdpa = True

    # Freeze vae and text encoders
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # After loading UNet but before applying LoRA
    unet.config.sample_size = 32  # Change back to 32 from 16

    # After loading UNet
    try:
        import xformers

        unet.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xformers memory efficient attention")
    except ImportError:
        logger.info("xformers not available")

    logger.info("Models loaded successfully")

    # Set up LoRA config
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=[
            "to_k",
            "to_q",
            "to_v",  # They include this but with lower rank
        ],
        bias="none",
        lora_dropout=0.0,
        rank_pattern={"to_v": 2},  # Lower rank for value projection
    )

    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    logger.info("LoRA applied to UNet")

    # After LoRA setup
    logger.info("Creating dataset...")
    dataset = SDXLDataset(
        train_data_dir,
        tokenizer_one,
        tokenizer_two,
        size=128,  # Reduced from 512
        cache_latents=True,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    logger.info(f"Dataset created with {len(dataset)} images")

    # Set up optimizer
    logger.info("Setting up optimizer...")
    optimizer = AdamW8bit(
        unet.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01
    )

    # Prepare everything with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Enable gradient checkpointing
    unet.enable_gradient_checkpointing()
    text_encoder_one._set_gradient_checkpointing(True)
    text_encoder_two._set_gradient_checkpointing(True)

    # Get weight dtype for reference only
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info(f"Using mixed precision: {weight_dtype}")
    logger.info("Training setup complete")

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")

    # More aggressive memory cleanup
    def cleanup():
        if torch.cuda.is_available():
            # More aggressive cleanup
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if hasattr(torch.cuda, "memory_stats"):
                torch.cuda.reset_peak_memory_stats()
            gc.collect()

    # Split VAE encoding into chunks
    def encode_images_in_chunks(images, chunk_size=1):
        all_latents = []
        for i in range(0, images.shape[0], chunk_size):
            chunk = images[i : i + chunk_size]
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                chunk_latents = vae.encode(
                    chunk.to(accelerator.device, dtype=torch.bfloat16)
                ).latent_dist.sample()
                chunk_latents = chunk_latents * 0.18215
            all_latents.append(chunk_latents)
            del chunk
            cleanup()
        return torch.cat(all_latents)

    for epoch in range(num_epochs):
        total_loss = 0
        num_steps = 0
        logger.info(f"\nStarting epoch {epoch+1}/{num_epochs}")

        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Process in smaller chunks
            with accelerator.accumulate(unet):  # This helps with memory
                with torch.cuda.amp.autocast():
                    # Process VAE in chunks
                    latents = encode_images_in_chunks(batch["pixel_values"])
                    del batch["pixel_values"]
                    cleanup()

                    # Get the text embeddings
                    with torch.no_grad():
                        # Get full hidden states
                        text_outputs = text_encoder_one(
                            batch["input_ids_one"].to(accelerator.device),
                            output_hidden_states=True,
                        )
                        prompt_embeds = text_outputs.hidden_states[
                            -1
                        ]  # Use full hidden states

                        # Project to correct dimension if needed
                        if prompt_embeds.shape[-1] != 2048:
                            prompt_embeds = torch.nn.functional.pad(
                                prompt_embeds, (0, 2048 - prompt_embeds.shape[-1])
                            )

                        pooled_prompt_embeds = text_encoder_two(
                            batch["input_ids_two"].to(accelerator.device)
                        ).text_embeds
                        logger.debug(
                            f"Pooled embeds shape: {pooled_prompt_embeds.shape}"
                        )

                    # Add dimension checks
                    # logger.info(f"Prompt embeds shape: {prompt_embeds.shape}")
                    # logger.info(f"Pooled embeds shape: {pooled_prompt_embeds.shape}")
                    # logger.info(f"Latents shape: {latents.shape}")

                    # Training step
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, 1000, (latents.shape[0],), device=latents.device
                    )
                    noisy_latents = noise + timesteps.reshape(-1, 1, 1, 1) * latents

                    # Predict noise
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": torch.zeros((batch_size, 6)).to(
                                accelerator.device
                            ),
                        },
                    ).sample

                    loss = F.mse_loss(noise_pred, noise, reduction="mean")
                    total_loss += loss.item()
                    num_steps += 1

                    # Optimization step
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                # Free memory more aggressively
                del noise_pred
                del noise
                torch.cuda.empty_cache()

            if step % 2 == 0:  # More frequent cleanup
                cleanup()

            if step % 5 == 0:
                avg_loss = total_loss / num_steps if num_steps > 0 else 0
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} "
                    f"| Step {step}/{len(train_dataloader)} "
                    f"| Current Loss: {loss.item():.4f} "
                    f"| Avg Loss: {avg_loss:.4f}"
                )

        # End of epoch logging
        avg_epoch_loss = total_loss / num_steps if num_steps > 0 else 0
        logger.info(f"Epoch {epoch+1} completed | Average Loss: {avg_epoch_loss:.4f}")

    # Clean up offload directory
    if os.path.exists("offload"):
        shutil.rmtree("offload")
    logger.info("Cleaned up temporary files")

    logger.info("Training complete!")


class SDXLDataset(Dataset):
    def __init__(
        self, data_dir, tokenizer_one, tokenizer_two, size=128, cache_latents=True
    ):
        self.data_dir = data_dir
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.cache_latents = cache_latents
        self.cached_latents = {}

        # Get image paths
        self.image_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        # Image transforms
        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        # Get prompt from .txt file or use default
        prompt_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                prompt = f.read().strip()
        else:
            prompt = "a photo"

        # Tokenize prompt with both tokenizers
        tokenized_one = self.tokenizer_one(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_two = self.tokenizer_two(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "input_ids_one": tokenized_one.input_ids[0],
            "input_ids_two": tokenized_two.input_ids[0],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    train_lora_sdxl(
        pretrained_model_path=args.pretrained_model_path,
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
    )
