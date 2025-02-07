import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import base64

# Function to encode image in base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to your local logo image
logo_path = "C:/Users/KASHYAP/OneDrive/Desktop/CAPit2/pypro/WhatsApp Image 2025-02-07 at 10.08.30_63eb146f.jpg"
logo_base64 = get_base64_image(logo_path)

# Custom CSS for UI and Background
st.markdown(f"""
    <style>
    html, body, [class*="stApp"] {{
        background: rgb(45,19,74);
background: radial-gradient(circle, rgba(45,19,74,1) 50%, rgba(2,0,36,1) 100%);
        color: white;
    }}
    .title-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }}
    .logo {{
        width: 50px;
        height: 50px;
        border-radius: 50%;
    }}
    .title {{
        font-size: 42px;
        font-weight: bold;
        color: #F4A261;
        text-align: center;
    }}
    .subtitle {{
        font-size: 28px;
        font-weight: bold;
        color: #E9C46A;
        text-align: center;
        margin-top: 10px;
    }}
    .image-container {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .generated-caption {{
        font-size: 22px;
        font-weight: bold;
        color: #F94144;
        background-color: #FFFFFF;
            text-transform: uppercase;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }}
    .sidebar {{
        background-color:transparent;
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# Load BLIP model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load the model
processor, model = load_model()

# Sidebar for file upload
with st.sidebar:
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    st.title("📂 Recent Files")

    # Directory to store images
    IMAGE_DIR = "uploaded_images"
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    # Upload Image
    uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    # Save Uploaded Image
    if uploaded_image:
        image_path = os.path.join(IMAGE_DIR, uploaded_image.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        st.success(f"✅ Uploaded {uploaded_image.name}")

    st.markdown('</div>', unsafe_allow_html=True)

# Title with Logo
st.markdown(
    f"""
    <div class="title-container">
        <img src="data:image/jpg;base64,{logo_base64}" class="logo">
        <h1 class="title">CAPiT</h1>
    </div>
    """, unsafe_allow_html=True
)

# Subtitle
st.markdown('<h2 class="subtitle">Explore The Possibilities Of AI Captioning with "CAPiT"</h2>', unsafe_allow_html=True)

# Center-aligned text
st.markdown('<div style="text-align: center;"><p>📷 Upload an image to generate a caption.</p></div>', unsafe_allow_html=True)

# Image Processing
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")

    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(img, caption="📌 Uploaded Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Generate Caption
    st.write("🤖 Generating caption...")
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Display Caption (Centered)
    st.markdown(f'<p class="generated-caption">📝 {caption}</p>', unsafe_allow_html=True)

    # Copy Button
    st.code(caption)  # Shows caption in a copyable format
