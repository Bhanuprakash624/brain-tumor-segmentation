
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = 256

HF_REPO_ID = "Bhanu624/brain-tumor-segmentation"
MODEL_FILENAME = "brain_tumor_model.keras"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image):

    image = np.array(image)

    # Convert RGB/RGBA to grayscale
    if len(image.shape) == 3:

        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize
    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Convert grayscale to 3 channels
    image = np.stack(
        [image, image, image],
        axis=-1
    )

    # EfficientNet preprocessing is already handled
    # inside the Keras model
    image = image.astype(np.float32)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_mask(model, image):

    prediction = model.predict(
        image,
        verbose=0
    )

    prediction = prediction[0, :, :, 0]

    # Threshold
    mask = (prediction >= 0.5).astype(np.uint8)

    return mask


# --------------------------------------------------
# Overlay
# --------------------------------------------------

def create_overlay(original_image, mask):

    original = np.array(original_image)

    original = cv2.resize(
        original,
        (IMG_SIZE, IMG_SIZE)
    )

    if len(original.shape) == 2:
        original = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2RGB
        )

    overlay = original.copy()

    # Red color for predicted tumor
    overlay[mask == 1] = [255, 0, 0]

    # Blend original + segmentation
    result = cv2.addWeighted(
        original,
        0.65,
        overlay,
        0.35,
        0
    )

    return result


# --------------------------------------------------
# Streamlit Interface
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Segmentation Using Deep Learning")

st.write(
    "EfficientNet-B1 + U-Net++ based MRI brain tumor segmentation"
)

st.info(
    "Upload an MRI brain image to generate a predicted tumor segmentation mask."
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded MRI")

    st.image(
        image,
        caption="Original MRI Image",
        use_container_width=True
    )

    if st.button("🔍 Segment Tumor"):

        with st.spinner("Loading model and analyzing MRI..."):

            model = load_model()

            processed_image = preprocess_image(image)

            mask = predict_mask(
                model,
                processed_image
            )

            overlay = create_overlay(
                image,
                mask
            )

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Predicted Tumor Mask")

            st.image(
                mask * 255,
                caption="Predicted Segmentation",
                use_container_width=True
            )

        with col2:

            st.subheader("Segmentation Overlay")

            st.image(
                overlay,
                caption="Tumor Highlighted in Red",
                use_container_width=True
            )

        # --------------------------------------------------
        # Tumor Area
        # --------------------------------------------------

        tumor_pixels = np.sum(mask == 1)

        total_pixels = mask.shape[0] * mask.shape[1]

        tumor_percentage = (
            tumor_pixels / total_pixels
        ) * 100

        st.success(
            f"Predicted tumor area: {tumor_percentage:.2f}%"
        )

        st.warning(
            "This application is an academic research prototype "
            "and is not intended for clinical diagnosis."
        )
