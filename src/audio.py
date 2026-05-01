from io import BytesIO

from gtts import gTTS

from src.config import AUDIO_LANGUAGE


def create_caption_audio(text: str) -> bytes:
    """Convert caption text into MP3 bytes without writing a shared file."""
    audio_buffer = BytesIO()
    tts = gTTS(text=text, lang=AUDIO_LANGUAGE)
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()
