import numpy as np
import streamlit as st
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from streamlit_webrtc import webrtc_streamer

import joblib

final_model = joblib.load('svc_final_wholedataset.pkl')
scaler = StandardScaler()

def preprocess_image(img):
    img = cv2.imdecode(img, 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img = 255 - img_gray
    img_resized = cv2.resize(inverted_img, (28, 28))  # Resize the image to match MNIST dimensions
    blurred_img = cv2.GaussianBlur(img_resized, (3, 3), 4)
    img_flat = img_resized.flatten() / 255.0  # Flatten and normalize
    img_scaled = scaler.fit_transform(img_flat.reshape(-1, 1)).reshape(1, -1)
    return img_scaled, img_resized

def predict_image(img):
    img_preprocessed, _ = preprocess_image(img)
    prediction = final_model.predict(img_preprocessed)
    return prediction

nav = st.sidebar.radio("Navigation", ["Purpose", "Digit Predictor"])

if nav == "Purpose":
    st.title("Exam work")

if nav == "Digit Predictor":
    st.title("Digit Predictor")
    st.write("Upload an image containing a handwritten digit (0-9). Make sure the digit is clear and centered.")
 # File upload 
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "jpeg", "png"])

    # Create predict button
    if st.button("Predict"):
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', width=200)
        
            # Predict the processed image
            prediction = predict_image(file_bytes)
            st.success("Predicted Number: {}".format(prediction))
