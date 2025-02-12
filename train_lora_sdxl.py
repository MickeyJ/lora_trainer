# Core Python libraries
import os
import torch
import argparse
import logging
from datetime import datetime
from typing import Optional, List
import gc
from torch.cuda import empty_cache
import psutil
import torch.backends.cuda
import torch.backends.cudnn

# PyTorch data handling and processing
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# Stable Diffusion XL components
from diffusers import (
    StableDiffusionXLPipeline,  # Full SDXL pipeline (not used in training but listed for reference)
    AutoencoderKL,  # VAE for encoding images to/from latent space
    UNet2DConditionModel,  # The main model we're adapting with LoRA
)
from transformers import (
    CLIPTextModel,  # First text encoder (base CLIP model)
    CLIPTokenizer,  # Tokenizers for converting text to tokens
    CLIPTextModelWithProjection,  # Second text encoder (refined CLIP model, SDXL-specific)
)

# Training utilities
from accelerate import Accelerator
from tqdm.auto import tqdm  # Progress bars
import bitsandbytes as bnb  # 8-bit optimizers for memory efficiency
from peft import LoraConfig, get_peft_model  # LoRA adaptation tools

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler("training.log"),  # Save to file
    ],
)
logger = logging.getLogger(__name__)


class SDXLDataset(Dataset):
    """Dataset class for SDXL training

    This class handles loading and preprocessing of images and their corresponding
    text prompts for SDXL training. It supports both direct images and text files
    containing per-image prompts.
    """

    def __init__(
        self,
        image_dir=None,
        instance_prompt=None,
        tokenizer=None,
        tokenizer_2=None,  # SDXL uses two tokenizers
        size=1024,  # SDXL typically uses 1024x1024 images
    ):
        """Initialize the dataset

        Args:
            image_dir (str): Directory containing training images
            instance_prompt (str): Default prompt to use if no text file exists
            tokenizer (CLIPTokenizer): First CLIP tokenizer
            tokenizer_2 (CLIPTokenizer): Second CLIP tokenizer (SDXL-specific)
            size (int): Target image size (default: 1024 for SDXL)
        """
        logger.info(f"Initializing SDXL dataset from {image_dir}")

        self.image_dir = image_dir
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size

        # Get all valid image paths
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        logger.info(f"Found {len(self.image_paths)} images")

        # Get corresponding text files (same name but .txt extension)
        self.text_paths = [p.rsplit(".", 1)[0] + ".txt" for p in self.image_paths]

        # Define image preprocessing pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),  # Scale to target size
                transforms.CenterCrop(size),  # Crop to square
                transforms.ToTensor(),  # Convert to tensor (0-1 range)
                transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1] range
            ]
        )

    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load and process a single image-text pair

        Args:
            idx (int): Index of the item to load

        Returns:
            dict: Contains processed image tensor and tokenized text for both encoders
        """
        # Load and preprocess image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Try to load specific prompt from text file, fall back to instance_prompt
        text_path = self.text_paths[idx]
        if os.path.exists(text_path):
            with open(text_path, "r") as f:
                prompt = f.read().strip()
        else:
            prompt = self.instance_prompt

        # Get tokens from both tokenizers (SDXL-specific)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,  # Explicitly set max length for both encoders
            truncation=True,
            return_tensors="pt",
        )

        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,  # Match the length of the first encoder
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,  # [3, size, size]
            "input_ids": text_inputs.input_ids[0],  # [max_length]
            "input_ids_2": text_inputs_2.input_ids[0],  # [max_length]
        }


def resume_from_checkpoint(checkpoint_path, unet, optimizer):
    """Load training state from a checkpoint file

    Args:
        checkpoint_path (str): Path to the checkpoint file
        unet (UNet2DConditionModel): The UNet model to load weights into
        optimizer: The optimizer to load state into

    Returns:
        tuple: (start_epoch, best_val_loss)
    """
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    unet.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return start_epoch, best_val_loss


def get_available_modules(model):
    """Get all available module names in the model

    Args:
        model: The PyTorch model to inspect

    Returns:
        dict: Module names mapped to their types
    """
    modules_dict = {}

    def _get_modules(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            modules_dict[full_name] = type(child).__name__
            _get_modules(child, full_name)

    _get_modules(model)
    return modules_dict


def validate_target_modules(model, target_modules):
    """Validate that target modules exist in the model

    Args:
        model: The PyTorch model to check
        target_modules: List of module names to validate

    Returns:
        tuple: (bool, list) - (is_valid, list of missing modules)
    """
    available_modules = get_available_modules(model)
    logger.debug("Available modules:")
    for name, type_name in available_modules.items():
        logger.debug(f"  {name}: {type_name}")

    missing_modules = [
        module
        for module in target_modules
        if not any(module in full_name for full_name in available_modules.keys())
    ]

    if missing_modules:
        logger.error("The following target modules were not found in the model:")
        for module in missing_modules:
            logger.error(f"  - {module}")
        logger.error("\nAvailable modules that might be suitable targets:")
        for name, type_name in available_modules.items():
            if "Linear" in type_name or any(
                x in name.lower() for x in ["attn", "proj", "to_"]
            ):
                logger.error(f"  - {name} ({type_name})")

    return len(missing_modules) == 0, missing_modules


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        logger.info(
            f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, "
            f"{torch.cuda.memory_reserved()/1024**3:.2f}GB reserved"
        )


def clear_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        empty_cache()
    gc.collect()


def adjust_tensor_shapes(encoder_hidden_states_1, encoder_hidden_states_2, logger=None):
    """Adjust tensor shapes to ensure they can be concatenated properly.

    Args:
        encoder_hidden_states_1: Output from first text encoder
        encoder_hidden_states_2: Output from second text encoder
        logger: Optional logger for debugging shapes

    Returns:
        tuple: (adjusted_states_1, adjusted_states_2)
    """
    if logger:
        logger.debug(f"Encoder 1 output shape: {encoder_hidden_states_1.shape}")
        logger.debug(f"Encoder 2 output shape: {encoder_hidden_states_2.shape}")

    # Ensure both tensors are 3D [batch, sequence, hidden]
    if encoder_hidden_states_1.ndim == 2:
        encoder_hidden_states_1 = encoder_hidden_states_1.unsqueeze(0)
    if encoder_hidden_states_2.ndim == 2:
        encoder_hidden_states_2 = encoder_hidden_states_2.unsqueeze(0)

    # Get shapes
    b1, s1, h1 = encoder_hidden_states_1.shape
    b2, s2, h2 = encoder_hidden_states_2.shape

    # Match sequence lengths if needed
    if s1 != s2:
        if s1 == 1:
            # Repeat encoder_1 output to match encoder_2
            encoder_hidden_states_1 = encoder_hidden_states_1.repeat(1, s2, 1)
        elif s2 == 1:
            # Repeat encoder_2 output to match encoder_1
            encoder_hidden_states_2 = encoder_hidden_states_2.repeat(1, s1, 1)
        else:
            # If neither is length 1, pad the shorter one
            target_len = max(s1, s2)
            if s1 < target_len:
                pad_len = target_len - s1
                encoder_hidden_states_1 = torch.nn.functional.pad(
                    encoder_hidden_states_1, (0, 0, 0, pad_len)
                )
            if s2 < target_len:
                pad_len = target_len - s2
                encoder_hidden_states_2 = torch.nn.functional.pad(
                    encoder_hidden_states_2, (0, 0, 0, pad_len)
                )

    if logger:
        logger.debug(f"Adjusted encoder 1 shape: {encoder_hidden_states_1.shape}")
        logger.debug(f"Adjusted encoder 2 shape: {encoder_hidden_states_2.shape}")

    return encoder_hidden_states_1, encoder_hidden_states_2


def setup_memory_controls():
    """Setup CUDA memory controls based on recommendations"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    # Memory settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:32,"  # Reduced from 128
        "expandable_segments:True,"
        "garbage_collection_threshold:0.6,"  # More aggressive GC
        "roundup_power2:True"  # Add power of 2 rounding
    )


def enable_unet_optimizations(unet):
    """Enable all UNet memory optimizations"""
    logger.info("Enabling UNet memory optimizations")

    # Enable gradient checkpointing
    unet.enable_gradient_checkpointing()

    # Enable memory efficient attention
    if hasattr(unet.config, "use_memory_efficient_attention"):
        unet.config.use_memory_efficient_attention = True
        unet.config.use_sdpa = False

    # Enable split attention
    if hasattr(unet, "set_attention_slice_size"):
        unet.set_attention_slice_size(1)

    # Set to half precision
    unet = unet.half()

    return unet


def train_lora_sdxl(
    pretrained_model_path: str,
    train_data_dir: str,
    output_dir: str,
    target_modules: Optional[List[str]] = None,
    rank: int = 4,
    num_epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    mixed_precision: str = "bf16",  # Changed to bf16 for better memory efficiency
    save_steps: int = 100,
    debug: bool = False,
):
    """Train a LoRA adapter for SDXL"""

    # Default target modules for SDXL
    if target_modules is None:
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "time_emb_proj",
            "conv1",
            "conv2",
        ]

    try:
        # Setup memory optimizations first
        setup_memory_controls()

        # Load models with memory optimization
        logger.info("Loading model components with memory optimization...")
        clear_memory()

        tokenizer_1 = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_2"
        )

        clear_memory()
        text_encoder_1 = CLIPTextModel.from_pretrained(
            pretrained_model_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        ).eval()

        clear_memory()
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_path, subfolder="text_encoder_2", torch_dtype=torch.float16
        ).eval()

        clear_memory()
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, subfolder="vae", torch_dtype=torch.float16
        ).eval()

        clear_memory()
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_path, subfolder="unet", torch_dtype=torch.float16
        )

        # Move models to CPU initially
        text_encoder_1.to("cpu")
        text_encoder_2.to("cpu")
        vae.to("cpu")

        print_gpu_memory()

        logger.info("Initializing SDXL LoRA training")
        if not os.path.exists(train_data_dir):
            raise ValueError(f"Image directory not found: {train_data_dir}")

        logger.info(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Verify CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available! Training will be very slow on CPU.")
            if os.name == "nt":  # Windows
                logger.warning(
                    "On Windows, ensure you have installed bitsandbytes-windows"
                )

        # Apply VAE optimizations
        logger.info("Applying VAE optimizations")
        vae.requires_grad_(False)
        vae.half()
        vae.enable_slicing()
        vae.enable_tiling()

        # Apply all UNet optimizations (including attention slicing)
        unet = enable_unet_optimizations(unet)

        # After VAE and UNet optimizations but before LoRA config
        # Configure accelerator with simpler memory optimizations
        logger.info("Configuring accelerator")
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )

        # After UNet optimizations but before dataset creation
        # Configure LoRA for SDXL
        logger.info("Configuring LoRA")
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=target_modules,
            lora_dropout=0.1,
        )

        try:
            logger.info("Validating LoRA target modules")
            valid_modules, missing = validate_target_modules(
                unet, lora_config.target_modules
            )
            if not valid_modules:
                raise ValueError(
                    f"Invalid target modules. Please check the logs for available modules."
                )

            logger.info("Applying LoRA to UNet")
            unet = get_peft_model(unet, lora_config)
        except Exception as e:
            logger.error(f"LoRA configuration failed: {str(e)}")
            raise

        # Reduce image resolution if needed
        dataset = SDXLDataset(
            image_dir=train_data_dir,
            instance_prompt=None,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            size=768,  # Reduced from 1024 to save memory
        )

        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        logger.info(f"Dataset split: {train_size} training, {val_size} validation")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,  # Disable pin memory
            num_workers=0,  # Disable multiprocessing
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=0,
        )

        # Initialize optimizer
        logger.info("Initializing 8-bit AdamW optimizer")
        optimizer = bnb.optim.AdamW8bit(
            unet.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )

        # Prepare for distributed training
        logger.info("Preparing for distributed training")
        unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader, val_dataloader
        )

        # Move models to device after optimization setup
        device = accelerator.device
        logger.info(f"Using device: {device}")

        text_encoder_1.to(device)
        text_encoder_2.to(device)
        vae.to(device)

        # Update training loop to ensure tensors are on correct device
        best_val_loss = float("inf")
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            print_gpu_memory()

            # Training phase
            unet.train()
            train_loss = 0
            train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
            batch_count = 0

            for batch in train_progress:
                batch_count += 1
                with accelerator.accumulate(unet):
                    # Process images
                    logger.debug(
                        f"Processing batch {batch_count}/{len(train_dataloader)}"
                    )

                    # Move input tensors to correct device
                    input_ids = batch["input_ids"].to(device)
                    input_ids_2 = batch["input_ids_2"].to(device)
                    pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)

                    # Get embeddings from both text encoders (SDXL-specific)
                    logger.debug("Generating text embeddings")
                    encoder_hidden_states_1 = text_encoder_1(input_ids)[0]
                    encoder_hidden_states_2 = text_encoder_2(input_ids_2)[0]

                    # Adjust tensor shapes
                    encoder_hidden_states_1, encoder_hidden_states_2 = (
                        adjust_tensor_shapes(
                            encoder_hidden_states_1, encoder_hidden_states_2, logger
                        )
                    )

                    # Combine the embeddings (SDXL-specific)
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states_1, encoder_hidden_states_2], dim=-1
                    )

                    # Convert images to latent space
                    logger.debug("Converting images to latent space")
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215  # Scale factor for stability

                    # Add noise to latents
                    logger.debug("Adding noise to latents")
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, 1000, (latents.shape[0],))
                    noisy_latents = noise + timesteps.reshape(-1, 1, 1, 1) * latents

                    # Predict noise and calculate loss
                    logger.debug("Predicting noise and calculating loss")
                    noise_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)

                    # Backpropagate and optimize
                    logger.debug("Performing backpropagation")
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                    train_loss += loss.item()
                    current_loss = loss.item()
                    train_progress.set_postfix({"loss": current_loss})
                    logger.debug(f"Batch {batch_count} loss: {current_loss:.4f}")

                    # Add memory cleanup after heavy operations
                    if (
                        torch.cuda.memory_allocated()
                        / torch.cuda.max_memory_allocated()
                        > 0.85
                    ):
                        clear_memory()
                        print_gpu_memory()

            avg_train_loss = train_loss / len(train_dataloader)
            logger.info(
                f"Epoch {epoch+1} training completed. Average loss: {avg_train_loss:.4f}"
            )

            # Clear memory between epochs
            clear_memory()
            print_gpu_memory()

            # Validation phase
            logger.info("Starting validation phase")
            print_gpu_memory()

            unet.eval()
            val_loss = 0
            val_progress = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")
            batch_count = 0

            with torch.no_grad():
                for batch in val_progress:
                    batch_count += 1
                    logger.debug(
                        f"Validating batch {batch_count}/{len(val_dataloader)}"
                    )
                    input_ids = batch["input_ids"].to(device)
                    input_ids_2 = batch["input_ids_2"].to(device)
                    pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)

                    # Get embeddings from both encoders
                    encoder_hidden_states_1 = text_encoder_1(input_ids)[0]
                    encoder_hidden_states_2 = text_encoder_2(input_ids_2)[0]

                    # Adjust tensor shapes
                    encoder_hidden_states_1, encoder_hidden_states_2 = (
                        adjust_tensor_shapes(
                            encoder_hidden_states_1, encoder_hidden_states_2, logger
                        )
                    )

                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states_1, encoder_hidden_states_2], dim=-1
                    )

                    # Process through VAE and add noise
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, 1000, (latents.shape[0],))
                    noisy_latents = noise + timesteps.reshape(-1, 1, 1, 1) * latents

                    # Predict noise
                    noise_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    val_loss += loss.item()
                    current_loss = loss.item()
                    val_progress.set_postfix({"loss": current_loss})
                    logger.debug(
                        f"Validation batch {batch_count} loss: {current_loss:.4f}"
                    )

            avg_val_loss = val_loss / len(val_dataloader)
            logger.info(f"Validation completed. Average loss: {avg_val_loss:.4f}")

            # Log epoch summary
            logger.info(f"Epoch {epoch+1} Summary:")
            logger.info(f"Training Loss: {avg_train_loss:.4f}")
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"New best model! (Val Loss: {avg_val_loss:.4f})")
                accelerator.wait_for_everyone()
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(
                    os.path.join(output_dir, "best_model"),
                    safe_serialization=True,  # Use safetensors format
                )

            # Save checkpoint
            if (epoch + 1) % save_steps == 0:
                logger.info(f"Saving checkpoint for epoch {epoch+1}")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": accelerator.unwrap_model(unet).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                }
                torch.save(
                    checkpoint, os.path.join(output_dir, f"checkpoint-{epoch+1}.pt")
                )

        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final model saved in: {os.path.join(output_dir, 'best_model')}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print_gpu_memory()
        raise

    finally:
        # Move models back to CPU before cleanup
        text_encoder_1.to("cpu")
        text_encoder_2.to("cpu")
        vae.to("cpu")
        clear_memory()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train a LoRA model for Stable Diffusion XL fine-tuning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to directory containing training images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save model checkpoints and final weights",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="photo of sks person",
        help="Text prompt for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Images per batch",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Rank of LoRA update matrices",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_lora_sdxl(
        pretrained_model_path=args.base_model,
        train_data_dir=args.dataset,
        output_dir=args.output_dir,
        rank=args.rank,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.checkpoint_freq,
    )
