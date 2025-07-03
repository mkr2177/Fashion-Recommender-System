import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Ensure 'uploads' folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load feature list and filenames
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
except Exception as e:
    st.error(f"❌ Failed to load embeddings or filenames: {e}")
    st.stop()

# Load pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Title
st.title('👗 Fashion Recommender System')

# Save uploaded image to 'uploads/' folder
def save_uploaded_file(uploaded_file):
    try:
        filepath = os.path.join('uploads', uploaded_file.name)
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return filepath
    except Exception as e:
        st.error(f"❌ File saving failed: {e}")
        return None

# Extract feature from uploaded image
def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"❌ Feature extraction failed: {e}")
        return None

# Find similar items
def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"❌ Recommendation failed: {e}")
        return None

# Upload file
uploaded_file = st.file_uploader("📤 Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    filepath = save_uploaded_file(uploaded_file)
    if filepath:
        # Display uploaded image
        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)

        # Extract features
        features = feature_extraction(filepath, model)
        if features is not None:
            # Get recommendations
            indices = recommend(features, feature_list)
            if indices is not None:
                st.subheader("🎯 You may also like:")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    with col:
                        st.image(filenames[indices[0][i]], use_column_width=True)
    else:
        st.warning("⚠️ Upload failed. Please try again.")
