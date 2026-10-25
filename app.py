```python
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

from PIL import Image
from huggingface_hub import hf_hub_download


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Segmentation Using Deep Learning")

st.write(
    "Upload a brain MRI image and its corresponding "
    "ground-truth tumor mask to evaluate the segmentation."
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_trained_model():

    model_path = hf_hub_download(
        repo_id="Bhanu624/brain-tumor-segmentation",
        filename="brain_tumor_model.keras"
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

IMG_SIZE = 256


def preprocess_mri(uploaded_file):

    image = Image.open(uploaded_file).convert("L")

    original_image = np.array(image)

    resized_image = cv2.resize(
        original_image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Repeat grayscale image into three channels.
    image_3channel = np.stack(
        [resized_image] * 3,
        axis=-1
    )

    # EfficientNet preprocessing used in the project.
    image_input = np.expand_dims(
        image_3channel.astype(np.float32),
        axis=0
    )

    return original_image, resized_image, image_input


# ==========================================
# GROUND-TRUTH MASK PREPROCESSING
# ==========================================

def preprocess_mask(uploaded_file):

    mask_image = Image.open(uploaded_file).convert("L")

    mask = np.array(mask_image)

    mask = cv2.resize(
        mask,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert grayscale mask to binary values.
    mask = (mask > 127).astype(np.uint8)

    return mask


# ==========================================
# CALCULATE EVALUATION METRICS
# ==========================================

def calculate_metrics(y_true, y_pred):

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = np.logical_and(y_true, y_pred).sum()
    tn = np.logical_and(~y_true, ~y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()

    # Dice Similarity Coefficient
    dice_denominator = (2 * tp) + fp + fn

    dice = (
        (2 * tp) / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    # Intersection over Union
    iou_denominator = tp + fp + fn

    iou = (
        tp / iou_denominator
        if iou_denominator > 0
        else 1.0
    )

    # Precision
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    # Recall
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # Pixel Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "DSC": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Pixel Accuracy": accuracy
    }


# ==========================================
# UPLOAD FILES
# ==========================================

col1, col2 = st.columns(2)

with col1:
    mri_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["png", "jpg", "jpeg"]
    )

with col2:
    mask_file = st.file_uploader(
        "Upload Ground-Truth Tumor Mask",
        type=["png", "jpg", "jpeg"]
    )


# ==========================================
# PREDICTION AND EVALUATION
# ==========================================

if mri_file is not None and mask_file is not None:

    try:

        original_image, resized_image, image_input = (
            preprocess_mri(mri_file)
        )

        ground_truth = preprocess_mask(mask_file)

        # Predict tumor probability for every pixel.
        prediction = model.predict(
            image_input,
            verbose=0
        )[0, :, :, 0]

        # Convert probabilities into a binary mask.
        predicted_mask = (prediction >= 0.5).astype(np.uint8)

        # Calculate metrics against the actual mask.
        metrics = calculate_metrics(
            ground_truth,
            predicted_mask
        )

        # Calculate predicted tumor area.
        tumor_pixels = np.sum(predicted_mask)

        total_pixels = predicted_mask.size

        tumor_area = (
            tumor_pixels / total_pixels
        ) * 100

        # ==========================================
        # CREATE OVERLAY
        # ==========================================

        mri_rgb = cv2.cvtColor(
            resized_image,
            cv2.COLOR_GRAY2RGB
        )

        overlay = mri_rgb.copy()

        # Red indicates predicted tumor.
        overlay[predicted_mask == 1] = [255, 0, 0]

        # Blend original MRI and prediction.
        overlay = cv2.addWeighted(
            mri_rgb,
            0.65,
            overlay,
            0.35,
            0
        )

        # ==========================================
        # DISPLAY IMAGES
        # ==========================================

        st.subheader("Segmentation Results")

        img1, img2, img3 = st.columns(3)

        with img1:
            st.image(
                resized_image,
                caption="Original MRI",
                use_container_width=True
            )

        with img2:
            st.image(
                predicted_mask * 255,
                caption="Predicted Tumor Mask",
                use_container_width=True
            )

        with img3:
            st.image(
                overlay,
                caption="Segmentation Overlay",
                use_container_width=True
            )

        # ==========================================
        # GROUND TRUTH
        # ==========================================

        st.subheader("Ground-Truth Mask")

        st.image(
            ground_truth * 255,
            caption="Actual Tumor Mask",
            width=300
        )

        # ==========================================
        # TUMOR AREA
        # ==========================================

        st.subheader("Predicted Tumor Area")

        st.metric(
            label="Tumor Area (% of image pixels)",
            value=f"{tumor_area:.2f}%"
        )

        # ==========================================
        # EVALUATION METRICS
        # ==========================================

        st.subheader("Evaluation Metrics for This MRI")

        st.caption(
            "Metrics are calculated by comparing the predicted "
            "mask with the uploaded ground-truth mask."
        )

        metric1, metric2, metric3, metric4, metric5 = (
            st.columns(5)
        )

        metric1.metric(
            "Dice (DSC)",
            f"{metrics['DSC'] * 100:.2f}%"
        )

        metric2.metric(
            "IoU",
            f"{metrics['IoU'] * 100:.2f}%"
        )

        metric3.metric(
            "Precision",
            f"{metrics['Precision'] * 100:.2f}%"
        )

        metric4.metric(
            "Recall",
            f"{metrics['Recall'] * 100:.2f}%"
        )

        metric5.metric(
            "Pixel Accuracy",
            f"{metrics['Pixel Accuracy'] * 100:.2f}%"
        )

    except Exception as e:
        st.error(f"Error processing the uploaded files: {e}")

elif mri_file is not None or mask_file is not None:

    st.info(
        "Please upload both the MRI image and its "
        "corresponding ground-truth tumor mask."
    )

else:

    st.info(
        "Upload an MRI image and its matching ground-truth "
        "mask to generate predictions and evaluation metrics."
    )
```
