import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import torch
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

class ImageCaption:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate(self, img):
        if isinstance(img, str):
            img = Image.open(img)

        torch_device = torch.device("cpu")

        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(torch_device)
        model.eval()

        inputs = self.processor(images=img, return_tensors='pt')
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return img, caption

# Function to convert text to speech
def speak_text(text, filename="caption_audio.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

# Initialize
ic = ImageCaption()

# --- Streamlit UI ---
st.set_page_config(page_title="Image Captioning with Voice", layout="centered")
st.title("📸 Real-Time Image Captioning with Voice")
st.markdown("Upload or Capture an Image, and let AI *see* and *speak* for you.")

col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

with col2:
    captured_image = st.camera_input("📷 Capture Image from Camera")

image = None
if uploaded_image is not None:
    image = Image.open(uploaded_image)
elif captured_image is not None:
    image = Image.open(captured_image)

if image is not None:
    st.image(image, caption="🖼️ Your Image", use_column_width=True)
    st.markdown("### 🧠 Generating Caption...")

    try:
        result_img, caption = ic.generate(image)
        st.image(result_img, caption="🖼️ Image (Processed)", use_column_width=True)
        st.success("✅ Caption generated successfully!")
        st.write("**📝 Caption:**", caption)

        # Convert to speech
        audio_path = speak_text(caption)
        st.audio(audio_path, format="audio/mp3")
        st.info("🔊 Listen to the caption above!")

    except Exception as e:
        st.error(f"Error generating caption or audio: {str(e)}")

    st.markdown("<br><center>🔧 Created by: MOHAMMED GHANIM SIDDIQUI</center>", unsafe_allow_html=True)
    st.markdown("---")
