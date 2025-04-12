import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("soldier_uni_classifier.h5")
class_indices = {'BSF': 0, 'CRPF': 1, 'Jammu & Kashmir Police': 2}
reverse_class = {v: k for k, v in class_indices.items()}

st.set_page_config(page_title="Soldier Classifier", page_icon="🪖", layout="centered")

st.title("🪖 Soldier Uniform Classifier")

uploaded = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png','WebP'])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)


    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    label = reverse_class[np.argmax(pred)]

    st.markdown(f"### 🎯 Predicted: `{label}`")
