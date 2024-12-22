import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model('emotion_model.h5')  # Make sure to specify the correct path to your model

# Streamlit frontend
st.title('Model Prediction App')

# File uploader widget
st.write("Upload an image for prediction:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Make prediction and display result
if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (this might vary depending on your model)
    img = img.resize((224, 224))  # Resize to the size your model expects
    img = np.array(img) / 255.0  # Normalize if needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)

    # Assuming the model outputs a single class label
    st.write(f"Prediction: {prediction}")
