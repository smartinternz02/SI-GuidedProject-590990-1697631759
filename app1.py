import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = load_model('diabetic_retinopathy.h5')

st.set_page_config(
    page_title="Diabetic Detector",
    page_icon="🔎👁️",
    layout="wide"
)

#Custom CSS for styling
st.markdown(
    """
    <style>
    .stImage {
        border: 3px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
    }
    .st-text {
        font-size: 18px;
        color: #333;
    </style>
    """,
    unsafe_allow_html=True,
)

# Page title with Markdown
st.markdown("## Automated Diabetic Retinopathy Detection", unsafe_allow_html=True)

st.text("Please provide an EYE Image for Analysis.")
uploaded_file = st.file_uploader("Choose an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=540, use_column_width=True)

    st.write("Classifying...")

    img = image.convert('RGB')
    img = img.resize((224, 224))

    img = np.array(img)
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    tempDr = prediction[0][0]
    tempndr = prediction[0][1]

    if tempDr > tempndr:
        st.write("The patient has been diagnosed with Diabetic Retinopathy")
        st.markdown('<i class="fas fa-check-circle" style="color: #4CAF50;"></i>', unsafe_allow_html=True)
    else:
        st.write("The patient has not been diagnosed with Diabetic Retinopathy")
        st.markdown('<i class="fas fa-times-circle" style="color: #FF5733;"></i>', unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: green; font-size: 14px; margin-top: 50px;'>DEVELOPED BY - SAI KRISHNA, LOKESH, VARUN KUMAR</p>", unsafe_allow_html=True)