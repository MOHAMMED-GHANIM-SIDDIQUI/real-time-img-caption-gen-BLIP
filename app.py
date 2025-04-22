
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaption:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate(self, img):
        if isinstance(img, str):
            img = Image.open(img)

        inputs = self.processor(images=img, return_tensors='pt')
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        return img, caption

# Initialize the ImageCaption class
ic = ImageCaption()

# Streamlit UI
st.title("BLIP Image Captioning")

st.markdown("## Upload or Capture an Image to Generate a Caption")

# Add a camera capture functionality
col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

with col2:
    # Use Streamlit's camera component to capture an image
    captured_image = st.camera_input("Capture Image from Camera")

# Logic for handling image upload or capture
if uploaded_image is not None:
    image = Image.open(uploaded_image)
elif captured_image is not None:
    image = Image.open(captured_image)

# Check if an image is available
if uploaded_image is not None or captured_image is not None:
    st.image(image, caption="Uploaded or Captured Image", use_column_width=True)
    st.markdown("### Generating Caption...")

    # Generate caption
    result_img, caption = ic.generate(image)
    
    # Display the generated caption and image without watermark on the image itself
    st.image(result_img, caption="Image without Watermark", use_column_width=True)
    st.write("**Generated Caption:**", caption)

    # Add watermark below the image in the UI
    st.markdown("<br><br><center>Created by: MOHAMMED GHANIM SIDDIQUI</center>", unsafe_allow_html=True)

    # Add a separator
    st.markdown("---")
