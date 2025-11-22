🖼️ Keras CNN Image Classifier Deployment

This repository contains the code for an image classification pipeline built using TensorFlow/Keras, deployed as an interactive web application using Streamlit. The model is trained to classify images into different categories (e.g., car, bike, cat, dog) based on the provided dataset.

The model is saved using the native Keras/TensorFlow SavedModel format (a directory structure), which is the recommended practice for deployment.

🚀 Getting Started

Follow these steps to set up the environment, train the model, and run the web application locally.

1. Prerequisites

You need Python 3.8+ installed on your system.

2. Project Structure

The project relies on the following files:

File Name

Description

full_image_pipeline.py

Training Script: Loads data, defines, trains, and saves the CNN model (cnn_keras_model) and the LabelEncoder.

streamlit_app.py

Deployment Script: The Streamlit web application that loads the saved model and allows users to upload images for prediction.

requirements.txt

List of all necessary Python dependencies.

3. Setup and Installation

Clone the Repository (If applicable) or ensure all project files (.py files and requirements.txt) are in the same directory.

Install Dependencies:
You must install all required Python packages using the generated requirements.txt file:

pip install -r requirements.txt


Data Setup:
The training script is configured to look for images in a specific directory:

DATA_DIRECTORY = 'D:\\SMIT\\archive\\data\\all' 


Action Required: Ensure your image dataset is present at this path, or modify the DATA_DIRECTORY variable in full_image_pipeline.py to point to your data location.

🧠 Model Training

The full_image_pipeline.py script trains a Convolutional Neural Network (CNN) using K-Fold Cross-Validation (5 folds) and saves the final model and the label encoder.

Run the training script:

python full_image_pipeline.py


Generated Artifacts

Upon successful completion of the training script, two essential artifacts will be created in your project directory:

cnn_keras_model/ (Directory): The trained CNN model saved in the native TensorFlow/Keras SavedModel format.

label_encoder.joblib: The fitted sklearn.preprocessing.LabelEncoder object, necessary for converting the model's numerical output back into human-readable class names (e.g., 0 -> 'cat').

🌐 Running the Web Application

Once the model has been trained and the artifacts are generated, you can launch the Streamlit web application.

Run the Streamlit application:

streamlit run streamlit_app.py


This command will open the application in your default web browser, allowing you to upload images and test the model's predictions.
