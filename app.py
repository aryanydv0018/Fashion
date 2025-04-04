import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
import streamlit as st
import io
from PIL import Image
from numpy.linalg import norm  # Added missing import

# Set base directory (where app.py and myntradataset are located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'myntradataset', 'images')

st.header('Fashion Recommendation System')

# Load precomputed image features and filenames
Image_features = pkl.load(open(os.path.join(BASE_DIR, 'Images_features.pkl'), 'rb'))
filenames = pkl.load(open(os.path.join(BASE_DIR, 'filenames.pkl'), 'rb'))

# Function to extract image features
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)  # norm is now defined
    return norm_result

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.models.Sequential([base_model, GlobalMaxPool2D()])

# Fit Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Upload image
upload_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if upload_file is not None:
    # Convert uploaded file to image
    image_bytes = upload_file.read()
    img = Image.open(io.BytesIO(image_bytes))
    
    # Save temporarily
    temp_path = os.path.join(BASE_DIR, "temp.jpg")
    img.save(temp_path)
    
    st.subheader('Uploaded Image')
    st.image(img)

    # Extract features
    input_img_features = extract_features_from_images(temp_path, model)
    distance, indices = neighbors.kneighbors([input_img_features])

    st.subheader('Recommended Images')
    col1, col2, col3, col4 = st.columns(4)

    # Display recommended images with absolute paths
    for i, idx in enumerate(indices[0][1:5], 1):  # Skip the first index (input image itself)
        recommended_path = os.path.join(BASE_DIR, filenames[idx])
        if os.path.exists(recommended_path):
            if i == 1:
                with col1:
                    st.image(recommended_path)
            elif i == 2:
                with col2:
                    st.image(recommended_path)
            elif i == 3:
                with col3:
                    st.image(recommended_path)
            elif i == 4:
                with col4:
                    st.image(recommended_path)
        else:
            st.error(f"Image not found at {recommended_path}")

    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)