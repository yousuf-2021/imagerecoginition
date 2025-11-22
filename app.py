import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import joblib
import os

# --- Configuration (Must match the training script) ---
# CHANGED: Use the directory name for the native Keras/TensorFlow SavedModel format
MODEL_SAVE_PATH = 'cnn_keras_model' 
LABEL_ENCODER_SAVE_PATH = 'label_encoder.joblib'
IMAGE_SIZE = (128, 128) 

# --- Function to load model and encoder (Cached for efficiency) ---
@st.cache_resource
def load_artifacts():
    """Loads the trained model and LabelEncoder from disk."""
    try:
        # load_model() handles the Keras directory format correctly
        model = load_model(MODEL_SAVE_PATH)
        # Load the LabelEncoder (for decoding the numerical prediction)
        label_encoder = joblib.load(LABEL_ENCODER_SAVE_PATH)
        return model, label_encoder
    except FileNotFoundError as e:
        # Display an error if the files are not found (meaning the training script wasn't run)
        st.error(f"Required file not found. Please run full_image_pipeline.py first to create the model and encoder.")
        st.error(f"Missing file: {e.filename}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading artifacts: {e}")
        return None, None

# --- Prediction Pipeline Function (Adapted for Streamlit's file object) ---
def predict_uploaded_image(model, label_encoder, uploaded_file):
    """
    Applies the CNN Prediction Pipeline to an image file uploaded by the user.
    """
    try:
        # STEP 1: Load and Resize the image
        img = load_img(uploaded_file, target_size=IMAGE_SIZE)
        
        # STEP 2: Convert to Array and Normalize pixel values (0-255 -> 0-1)
        img_array = img_to_array(img) / 255.0
        
        # STEP 3: Add Batch Dimension (CRITICAL for Keras models)
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # STEP 4a: Make Prediction
        prediction_probabilities = model.predict(img_tensor, verbose=0)[0]
        
        # 4b. Decode: Finds the highest probability and converts to class name
        predicted_index = np.argmax(prediction_probabilities)
        predicted_class = label_encoder.inverse_transform([predicted_index])[0]
        confidence = prediction_probabilities[predicted_index]
        
        return predicted_class, confidence, img
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# --- STREAMLIT APP LAYOUT ---

def app():
    st.set_page_config(page_title="CNN Image Classifier", layout="centered")

    st.title("🖼️ Image Classification Deployment (CNN)")
    st.markdown("Upload an image (e.g., bike, car, cat, dog) to get a real-time prediction from the trained model.")

    # 1. Load the model and encoder
    model, label_encoder = load_artifacts()

    if model is not None and label_encoder is not None:
        
        st.sidebar.subheader("Model Information")
        st.sidebar.markdown(f"**Loaded Model:** `{MODEL_SAVE_PATH}` (Keras/TF format)")
        st.sidebar.markdown(f"**Trained Classes:** {', '.join(list(label_encoder.classes_))}")
        st.sidebar.markdown(f"**Input Size:** {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")

        # 2. File Uploader component
        uploaded_file = st.file_uploader(
            "Choose an image file...", 
            type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            st.write("---")
            
            # 3. Prediction Button
            if st.button("Classify Image"):
                with st.spinner('Analyzing image with CNN...'):
                    # Perform the prediction
                    predicted_class, confidence, img_for_prob_display = predict_uploaded_image(
                        model, label_encoder, uploaded_file
                    )
                
                if predicted_class:
                    st.success("Classification Complete!")
                    
                    st.metric(
                        label="Predicted Class", 
                        value=predicted_class.upper(),
                        delta=f"{confidence*100:.2f}% Confidence"
                    )
                    
                    # Display all probabilities
                    st.subheader("Full Probability Breakdown")
                    
                    # Rerunning prediction to get full probabilities based on the preprocessed image
                    prediction_probabilities = model.predict(
                        np.expand_dims(img_to_array(img_for_prob_display) / 255.0, axis=0), verbose=0
                    )[0]
                    
                    # Create a dictionary for the dataframe
                    data = {
                        "Class": label_encoder.classes_,
                        "Probability (%)": [f"{p*100:.2f}" for p in prediction_probabilities]
                    }
                    
                    st.dataframe(data, use_container_width=True)

if __name__ == "__main__":
    # Ensure TensorFlow verbosity is minimal for a cleaner Streamlit output
    tf.get_logger().setLevel('ERROR') 
    app()