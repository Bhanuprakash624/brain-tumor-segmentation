# Brain Tumor Segmentation Using EfficientNet-B1 and U-Net++

## Project Overview

This project presents an automated brain tumor segmentation system for MRI images using deep learning and transfer learning.

A pre-trained EfficientNet-B1 network is used as the encoder for feature extraction, while a U-Net++-style decoder performs pixel-level tumor segmentation.

## Model

- Encoder: EfficientNet-B1
- Decoder: U-Net++-style decoder
- Transfer Learning: ImageNet pre-trained weights
- Loss Function: Binary Cross-Entropy + Dice Loss
- Input Size: 256 × 256 pixels

## Dataset

The model was developed using a brain tumor MRI dataset containing:

- 3,064 MRI images
- Corresponding tumor segmentation masks
- 233 unique patients

The dataset was divided at the patient level into training, validation and testing sets to prevent patient overlap.

## Test Results

The final model was evaluated on 454 independent test images.

| Metric | Result |
|---|---:|
| Dice Score | 64.49% |
| IoU | 52.73% |
| Precision | 66.91% |
| Recall | 73.79% |
| Accuracy | 99.10% |

## Application

The Streamlit application allows the user to:

1. Upload an MRI brain image.
2. Process the image automatically.
3. Generate a tumor segmentation mask.
4. Display the segmentation overlay.
5. Estimate the predicted tumor area.

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNet-B1
- U-Net++
- OpenCV
- NumPy
- Pillow
- Streamlit

## Disclaimer

This project is an academic research prototype and is not intended for clinical diagnosis or medical decision-making.
