import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration

from PIL import Image

# Load BLIP model and processor
@st.cache_resource
def load_model():
    st.write("Loading BLIP model and processor...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load the model
processor, model = load_model()

# Streamlit UI
st.title("BLIP Image Captioning")
st.write("Upload an image to generate a caption.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Open and display the uploaded image
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and generate the caption
    st.write("Generating caption...")
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs)

    # Decode the output to get the caption
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Display the caption
    st.subheader("Generated Caption:")
    st.write(caption)
