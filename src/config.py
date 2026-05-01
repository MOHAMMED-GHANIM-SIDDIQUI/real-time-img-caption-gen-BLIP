CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# BLIP may echo conditional prompts in the final caption, so the default uses
# unconditional captioning and leaves the visible output model-generated only.
CAPTION_PROMPT = ""

MAX_IMAGE_EDGE = 1024
VIDEO_SAMPLE_FRAMES = 6
MAX_VIDEO_SIZE_MB = 100

GENERATION_SETTINGS = {
    "max_new_tokens": 60,
    "num_beams": 6,
    "repetition_penalty": 1.2,
}

AUDIO_LANGUAGE = "en"
