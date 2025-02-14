import os
from PIL import Image


def resize_images(input_dir, target_size=1024):
    """
    Resize all images in the directory to have a maximum dimension of target_size
    while maintaining aspect ratio. Convert PNGs to JPGs.
    """
    # Get all image files
    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for filename in image_files:
        filepath = os.path.join(input_dir, filename)
        with Image.open(filepath) as img:
            # Convert to RGB (in case of PNG with alpha channel)
            if img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            ):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                img = bg

            # Calculate new dimensions maintaining aspect ratio
            width, height = img.size
            if width > height:
                new_width = target_size
                new_height = int(height * (target_size / width))
            else:
                new_height = target_size
                new_width = int(width * (target_size / height))

            # Resize image
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save as JPG, replacing original file
            new_filepath = os.path.splitext(filepath)[0] + ".jpg"
            resized.save(new_filepath, "JPEG", quality=95)

            # Remove original PNG if it was converted
            if filepath.lower().endswith(".png") and filepath != new_filepath:
                os.remove(filepath)

            print(f"Processed {filename}: {width}x{height} -> {new_width}x{new_height}")


if __name__ == "__main__":
    input_dir = r"extra_image_data"
    target_size = 512  # Reduced from 1024

    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found!")
    else:
        resize_images(input_dir, target_size)
        print("Processing complete!")
