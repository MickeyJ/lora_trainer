from train_lora_sdxl import train_lora_sdxl

train_lora_sdxl(
    train_data_dir=r"input",
    output_dir="output",
    num_epochs=100,  # More epochs for fewer images
    batch_size=1,
    image_size=384,
    learning_rate=1e-6,  # Lower for stability
    gradient_accumulation_steps=8,
)
