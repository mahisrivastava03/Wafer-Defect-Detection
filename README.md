# Wafer Defect Detection using Deep Learning

## Problem Statement
Automated detection and classification of wafer defects to improve yield in semiconductor manufacturing.

## Dataset
Wafer defect image dataset with 6 classes:
0, 1, 2, 3, 4, good  
Organized into Train, Validation, and Test folders.

> Note: Due to GitHub file size limits, the dataset zip is provided separately in the hackathon submission portal.

## Model
- Architecture: MobileNetV2 (Transfer Learning)
- Framework: TensorFlow + Keras
- Training Platform: Google Colab
- GPU Used: NVIDIA Tesla T4

## Results
Model evaluated using Accuracy, Precision, Recall, and Confusion Matrix.

## Deployment
- Model exported to ONNX format
- Suitable for NXP eIQ deployment

## Files
- wafer_model.onnx – Trained ONNX model
- notebook.ipynb – Training and inference code

