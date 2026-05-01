from PIL import Image, ImageOps

from src.config import MAX_IMAGE_EDGE


def load_image(image_source) -> Image.Image:
    """Open, orient, and normalize a user-provided image for model inference."""
    image = Image.open(image_source)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Very large uploads slow down inference without adding much caption quality.
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    return image
