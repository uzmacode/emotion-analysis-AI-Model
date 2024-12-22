import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model, tokenizer, and label encoder

import tensorflow as tf

# Load the saved model from .keras file
model = tf.keras.models.load_model('path_to_your_model/emotion_model.h5')

tokenizer = joblib.load('AI_model/tokenizer.pkl')
label_encoder = joblib.load('AI_model/label_encoder.pkl')

# Define constants
max_len = 100  # Ensure this matches your training data

# Streamlit UI setup
st.title("Emotion Detection from Text")

# Description and instructions
st.markdown("""
This application uses a deep learning model to predict the emotion of a given text. 
Simply enter a sentence and click on the **"Predict Emotion"** button to get the emotion detected.
""")

# Sidebar with theme and settings (Optional)
st.sidebar.header("Settings")
theme_color = st.sidebar.color_picker("Pick a theme color", "#9b59b6")
st.sidebar.markdown("""
### Model Information
- **Model**: Emotion Detection from Text
- **Version**: 1.0
- **By**: Uzma
""")

# Input text from user
input_text = st.text_area("Enter a sentence to detect emotion", height=200)

# Predict button
if st.button("Predict Emotion"):
    if input_text:  # Check if the input text is not empty
        # Preprocess the input text
        sequence = tokenizer.texts_to_sequences([input_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len)

        # Make prediction
        pred = model.predict(padded_sequence)
        emotion = label_encoder.inverse_transform([np.argmax(pred)])

        # Display result with custom styling
        st.subheader(f"The emotion detected is: **{emotion[0]}**")

        # Additional styling for prediction
        if emotion[0].lower() == 'happy':
            st.markdown(f"<h3 style='color:green;'>You seem happy! 😊</h3>", unsafe_allow_html=True)
        elif emotion[0].lower() == 'sad':
            st.markdown(f"<h3 style='color:blue;'>You seem sad. 😞</h3>", unsafe_allow_html=True)
        elif emotion[0].lower() == 'angry':
            st.markdown(f"<h3 style='color:red;'>You seem angry! 😡</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:gray;'>Emotion not detected clearly. 🤔</h3>", unsafe_allow_html=True)
    else:
        st.error("Please enter a sentence to detect emotion.")

# Optional: Display theme color selected by the user (for fun)
st.markdown(f"<p style='color:{theme_color};'>The selected theme color is: {theme_color}</p>", unsafe_allow_html=True)

# Footer with custom styling
footer = """
<div style="position: absolute; bottom: 10px; left: 10px; font-size: 12px; color: grey;">
    <p>Train and Design by Uzma (67381) - BS-IT 7th Semester</p>
    <p>Viqar-un-Nisa College for Women, Rawalpindi</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
