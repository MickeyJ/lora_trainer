import os
import shutil

# More conservative CUDA memory settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:21,"  # Minimum value that works
    "expandable_segments:True,"
    "garbage_collection_threshold:0.6"
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable symlinks warning
# Add their recommended number of CPU threads
os.environ["OMP_NUM_THREADS"] = "8"  # They suggest 8 CPU threads
os.environ["XFORMERS_MORE_DETAILS"] = "1"  # They suggest 8 CPU threads

import torch
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.cuda
import gc
from bitsandbytes.optim import AdamW8bit
from tqdm.auto import tqdm
from diffusers.models.attention_processor import XFormersAttnProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_lora_sdxl(
    train_data_dir: str,
    output_dir: str,
    pretrained_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
    num_epochs: int = 5,  # They recommend 5 epochs
    image_size: int = 512,  # Add this parameter
    batch_size: int = 1,  # They suggest reducing to 1 if having memory issues
    gradient_accumulation_steps: int = 8,  # Keep our increased value
    learning_rate: float = 1e-4,  # Reduced from 3e-4 to 1e-4
):
    """Train a LoRA adapter for SDXL"""

    logger.info("Starting SDXL LoRA training...")

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16",
    )

    # Reduce max gradient norm for more stability
    accelerator.clip_grad_norm_ = 0.5  # Changed from 1.0 to 0.5

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
    # Set up quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load models with quantization config
    text_encoder_one = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
        quantization_config=quantization_config,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder_2",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    # Keep VAE in bfloat16 since it's sensitive to precision
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

    # Only prepare non-8bit models with accelerator
    vae = accelerator.prepare(vae)

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

    # After loading VAE
    vae = vae.to("cpu")  # Keep VAE in CPU

    # After loading UNet
    try:
        import xformers

        class CustomAttnProcessor(XFormersAttnProcessor):
            def __call__(
                self,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
            ):
                batch_size, sequence_length, _ = hidden_states.shape
                attention_mask = attn.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )

                query = attn.to_q(hidden_states)
                key = attn.to_k(
                    encoder_hidden_states
                    if encoder_hidden_states is not None
                    else hidden_states
                )
                value = attn.to_v(
                    encoder_hidden_states
                    if encoder_hidden_states is not None
                    else hidden_states
                )

                # Convert all to same dtype
                dtype = torch.float16
                query = query.to(dtype)
                key = key.to(dtype)
                value = value.to(dtype)

                hidden_states = xformers.ops.memory_efficient_attention(
                    query, key, value, attention_mask
                )
                # Apply the output projections and return a single tensor
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states

        unet.set_attn_processor(CustomAttnProcessor())
        logger.info(f"Successfully enabled xformers version {xformers.__version__}")
        logger.info("Memory efficient attention active - expect 20-30% less VRAM usage")
    except (ImportError, Exception) as e:
        logger.warning(f"xformers not available: {e}")
        logger.info("Falling back to default memory efficient attention")
        if hasattr(unet.config, "use_memory_efficient_attention"):
            unet.config.use_memory_efficient_attention = True
            unet.config.use_sdpa = True
            logger.info("Using PyTorch's native memory efficient attention")

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
        size=image_size,
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

    # Enable gradient checkpointing with custom config
    unet.enable_gradient_checkpointing()
    unet.gradient_checkpointing_kwargs = {
        "use_reentrant": False,  # More memory efficient
        "checkpoint_rng_state": False,  # Skip RNG state checkpointing
    }

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
            # Force garbage collection
            gc.collect()
            # Clear gradient checkpoints
            if hasattr(torch, "clear_checkpoints"):
                torch.clear_checkpoints()

    # Split VAE encoding into chunks
    def encode_images_in_chunks(images, chunk_size=1):
        all_latents = []
        for i in range(0, images.shape[0], chunk_size):
            chunk = images[i : i + chunk_size]
            # Move chunk to CPU, encode, then move back
            with torch.no_grad():
                chunk_cpu = chunk.to("cpu")
                chunk_latents = vae.encode(chunk_cpu).latent_dist.sample()
                chunk_latents = chunk_latents.to(accelerator.device) * 0.18215
            all_latents.append(chunk_latents)
            del chunk, chunk_latents, chunk_cpu
            cleanup()
        return torch.cat(all_latents)

    def log_memory_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(
                f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved"
            )

    for epoch in range(num_epochs):
        cleanup()  # Clean before each epoch
        progress_bar = tqdm(
            total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}"
        )
        total_loss = 0
        num_steps = 0
        logger.info(f"\nStarting epoch {epoch+1}/{num_epochs}")

        unet.train()
        for step, batch in enumerate(train_dataloader):
            cleanup()  # Clean before each batch
            with accelerator.accumulate(unet):
                with torch.cuda.amp.autocast():
                    # Process VAE in chunks
                    latents = encode_images_in_chunks(batch["pixel_values"])
                    del batch["pixel_values"]
                    cleanup()

                    # Get noise - ensure same device as latents
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, 1000, (latents.shape[0],), device=latents.device
                    )
                    noisy_latents = (
                        noise
                        + timesteps.reshape(-1, 1, 1, 1).to(latents.device) * latents
                    )
                    del latents
                    cleanup()

                    # Get text embeddings
                    with torch.no_grad():
                        text_outputs = text_encoder_one(
                            batch["input_ids_one"].to(accelerator.device),
                            output_hidden_states=True,
                        )
                        prompt_embeds = text_outputs.hidden_states[-1]

                        # Project to correct dimension if needed
                        if prompt_embeds.shape[-1] != 2048:
                            prompt_embeds = torch.nn.functional.pad(
                                prompt_embeds, (0, 2048 - prompt_embeds.shape[-1])
                            )

                        del text_outputs
                        cleanup()

                        pooled_prompt_embeds = text_encoder_two(
                            batch["input_ids_two"].to(accelerator.device)
                        ).text_embeds

                        # Ensure correct shape
                        if pooled_prompt_embeds.shape[-1] != 1280:
                            pooled_prompt_embeds = torch.nn.functional.pad(
                                pooled_prompt_embeds,
                                (0, 1280 - pooled_prompt_embeds.shape[-1]),
                            )

                        cleanup()

                    # Forward pass
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": torch.zeros(
                                (batch_size, 6), device=accelerator.device
                            ),
                        },
                    ).sample
                    del noisy_latents, prompt_embeds, pooled_prompt_embeds
                    cleanup()

                    loss = F.mse_loss(noise_pred, noise, reduction="mean")
                    total_loss += loss.item()
                    num_steps += 1

                    # Optimization step
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    cleanup()  # Clean after optimization

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

            progress_bar.update(1)

            # Add to training loop at key points
            log_memory_usage()

        # End of epoch logging
        avg_epoch_loss = total_loss / num_steps if num_steps > 0 else 0
        logger.info(f"Epoch {epoch+1} completed | Average Loss: {avg_epoch_loss:.4f}")

    # Clean up offload directory
    if os.path.exists("offload"):
        shutil.rmtree("offload")
    logger.info("Cleaned up temporary files")

    # After training loop completes, before "Training complete!"
    # Save the LoRA weights
    logger.info("Saving LoRA weights...")
    unet.save_pretrained(output_dir)

    # Save the config file
    lora_config.save_pretrained(output_dir)

    # Optional: Save a model card with training details
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(
            f"""
# SDXL LoRA Model
- Base model: {pretrained_model_path}
- Training epochs: {num_epochs}
- Learning rate: {learning_rate}
- Batch size: {batch_size}
- Gradient accumulation steps: {gradient_accumulation_steps}
        """
        )

    logger.info("Training complete!")


class SDXLDataset(Dataset):
    def __init__(
        self, data_dir, tokenizer_one, tokenizer_two, size=512, cache_latents=True
    ):
        self.data_dir = data_dir
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.cache_latents = cache_latents
        self.cached_latents = {}
        self.text_embeddings_cache = {}
        # Move text encoders to CPU
        self.text_encoder_one = text_encoder_one.to("cpu")
        self.text_encoder_two = text_encoder_two.to("cpu")

        # Get image paths
        self.image_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        # Image transforms
        self.transforms = transforms.Compose(
            [
                # First resize the smaller edge to target size while maintaining aspect ratio
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                # Then pad to square with a fixed padding function
                transforms.Lambda(
                    lambda x: transforms.functional.pad(
                        x,
                        padding=[
                            0,  # left
                            0,  # top
                            max(0, size - x.size[-1]),  # right
                            max(0, size - x.size[-2]),  # bottom
                        ],
                        fill=0,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Clear any existing cached data
        if hasattr(self, "cached_latents"):
            del self.cached_latents
        self.cached_latents = {}
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Clear cache periodically
        if idx % 4 == 0 and hasattr(self, "cached_latents"):
            self.cached_latents.clear()
            torch.cuda.empty_cache()

        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transforms(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

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

    def get_text_embeddings(self, prompt):
        if prompt in self.text_embeddings_cache:
            return self.text_embeddings_cache[prompt]

        # Process on CPU
        with torch.no_grad():
            text_inputs_one = self.tokenizer_one(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_one.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to("cpu")

            text_inputs_two = self.tokenizer_two(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to("cpu")

            prompt_embeds = self.text_encoder_one(
                text_inputs_one.input_ids,
                output_hidden_states=True,
            ).hidden_states[-1]

            pooled_prompt_embeds = self.text_encoder_two(
                text_inputs_two.input_ids
            ).text_embeds

            # Cache results
            self.text_embeddings_cache[prompt] = {
                "prompt_embeds": prompt_embeds.to(accelerator.device),
                "pooled_prompt_embeds": pooled_prompt_embeds.to(accelerator.device),
            }

        return self.text_embeddings_cache[prompt]


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
