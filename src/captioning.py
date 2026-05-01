import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.config import CAPTION_MODEL_NAME, CAPTION_PROMPT, GENERATION_SETTINGS


class ImageCaptioner:
    """Small wrapper around BLIP image captioning inference."""

    def __init__(
        self,
        processor: BlipProcessor,
        model: BlipForConditionalGeneration,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.processor = processor
        self.model = model
        self.device = device
        self.dtype = dtype

    def generate_caption(self, image) -> str:
        processor_args = {"images": image, "return_tensors": "pt"}
        if CAPTION_PROMPT:
            processor_args["text"] = CAPTION_PROMPT

        inputs = self.processor(**processor_args)
        inputs = {
            name: self._move_input_tensor(value)
            for name, value in inputs.items()
        }

        with torch.inference_mode():
            output = self.model.generate(**inputs, **GENERATION_SETTINGS)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return self._clean_caption(caption)

    def _move_input_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(value):
            return value.to(device=self.device, dtype=self.dtype)
        return value.to(self.device)

    def _clean_caption(self, caption: str) -> str:
        caption = caption.strip()
        if CAPTION_PROMPT and caption.lower().startswith(CAPTION_PROMPT.lower()):
            caption = caption[len(CAPTION_PROMPT):].strip(" ,.-")
        return caption


def get_inference_device() -> torch.device:
    """Prefer GPU when available, otherwise use CPU for broad deployment support."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_dtype(device: torch.device):
    """Use lower precision on CUDA to reduce memory use during inference."""
    return torch.float16 if device.type == "cuda" else torch.float32


def load_processor():
    try:
        return BlipProcessor.from_pretrained(
            CAPTION_MODEL_NAME,
            use_fast=False,
            local_files_only=True,
        )
    except OSError:
        return BlipProcessor.from_pretrained(CAPTION_MODEL_NAME, use_fast=False)


def load_model(model_kwargs):
    try:
        return BlipForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_NAME,
            local_files_only=True,
            **model_kwargs,
        )
    except OSError:
        return BlipForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_NAME,
            **model_kwargs,
        )


def load_captioner() -> ImageCaptioner:
    """Load the BLIP processor and model for image captioning."""
    device = get_inference_device()
    dtype = get_model_dtype(device)

    model_kwargs = {}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = dtype

    processor = load_processor()
    model = load_model(model_kwargs).to(device)
    model.eval()
    return ImageCaptioner(processor=processor, model=model, device=device, dtype=dtype)
