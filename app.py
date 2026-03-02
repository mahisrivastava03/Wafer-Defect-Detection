import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("wafer_model.keras")

st.title("Wafer Defect Detection System")

uploaded_file = st.file_uploader("Upload Wafer Image", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_resized = cv2.resize(img, (224,224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_resized)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    st.image(img, channels="BGR")
    st.success(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
