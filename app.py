import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

IMG_SIZE = 224

# Load the trained CNN model
model = tf.keras.models.load_model(
    os.path.join("saved_model", "transfer_learning_model.h5")
)


st.title("Breast Cancer Classifier (BreakHis)")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    # Convert the image to PNG format
    img = img.convert("RGB")

    # Preprocess the uploaded image for the model
    img = img.resize((IMG_SIZE, IMG_SIZE))
    image_arr = np.array(img)

    # Make a prediction using the model
    pred = model.predict(np.expand_dims(image_arr, axis=0))

    # class_idx = np.argmax(pred)
    thres = 0.4
    class_idx = (pred > thres).astype(int).item()

    # if class_idx == 1:
    #     class_names = "malignant"
    # else:
    #     class_names = "benign"
    class_names = ["benign", "malignant"]

    st.write(f"Prediction: {class_idx}")
    st.write(f"Prediction: {class_names[class_idx]}")

    if class_names[class_idx] == "malignant":
        conf_prct = pred[0] * 100
    else:
        conf_prct = (1 - pred[0]) * 100
    st.write(f"Confidence: {conf_prct[0]:.2f} %")
