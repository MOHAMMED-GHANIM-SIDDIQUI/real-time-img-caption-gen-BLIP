import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import logging

# Optional logging for debug (can remove later)
logging.basicConfig(level=logging.INFO)

class ImageCaption:
    def __init__(self):
        # Only load processor once
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate(self, img):
        if isinstance(img, str):
            img = Image.open(img)

        # Force everything to CPU
        torch_device = torch.device("cpu")

        # Reload model on CPU every time (important for Streamlit Cloud)
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(torch_device)
        model.eval()

        # Process and move inputs to CPU
        inputs = self.processor(images=img, return_tensors='pt')
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}

        # Generate caption
        with torch.no_grad():
            output = model.generate(**inputs)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return img, caption

# Initialize
ic = ImageCaption()

# --- Streamlit UI ---
st.title("📸 Real-Time Image Captioning with BLIP")
st.markdown("Upload or Capture an Image, and let AI describe it for you.")

col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

with col2:
    captured_image = st.camera_input("📷 Capture Image from Camera")

# Determine the source of the image
image = None
if uploaded_image is not None:
    image = Image.open(uploaded_image)
elif captured_image is not None:
    image = Image.open(captured_image)

# If an image is provided, run caption generation
if image is not None:
    st.image(image, caption="🖼️ Uploaded or Captured Image", use_column_width=True)
    st.markdown("### 🧠 Generating Caption...")

    try:
        result_img, caption = ic.generate(image)
        st.image(result_img, caption="🖼️ Image (No Watermark)", use_column_width=True)
        st.write("**📝 Generated Caption:**", caption)
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")

    st.markdown("<br><br><center>🔧 Created by: MOHAMMED GHANIM SIDDIQUI</center>", unsafe_allow_html=True)
    st.markdown("---")
