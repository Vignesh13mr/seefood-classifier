"""Small Streamlit interface for the trained SeeFood model."""

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.seefood.config import MODEL_PATH
from src.seefood.inference import predict_image

st.set_page_config(page_title="SeeFood", page_icon="🌭", layout="centered")
st.title("🌭 SeeFood: Hotdog or Not Hotdog?")
st.write("Upload an image and the CNN will estimate whether it contains a hotdog.")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Use a JPG, PNG, WebP, or BMP image.",
)

if not MODEL_PATH.is_file():
    st.info("Train the model first. Follow the dataset and training steps in README.md.")

if uploaded is not None:
    try:
        preview = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be read as an image. Please upload a valid image.")
    else:
        st.image(preview, caption="Uploaded image", use_container_width=True)
        if MODEL_PATH.is_file():
            with st.spinner("Analyzing image…"):
                temporary_path = Path(".uploaded_image")
                try:
                    preview.save(temporary_path, format="PNG")
                    prediction = predict_image(temporary_path)
                except Exception as error:
                    st.error(f"Could not classify this image: {error}")
                else:
                    st.subheader(prediction["label"] + "!")
                    st.metric("Model confidence", f"{prediction['confidence']:.1%}")
                    st.caption(
                        "Confidence is the model's probability for its chosen class; "
                        "it is not a guarantee."
                    )
                finally:
                    temporary_path.unlink(missing_ok=True)

