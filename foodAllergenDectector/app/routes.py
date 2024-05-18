from flask import Blueprint, request, render_template, redirect, url_for
import joblib
import numpy as np
from app.forms import UploadForm
from app.utils import extract_text_from_image, preprocess_text
import os

main = Blueprint('main', __name__)

# Paths to the model files
model_path = 'models/allergen_model.joblib'
scaler_path = 'models/scaler.joblib'
encoder_path = 'models/encoder.joblib'

# Debug prints to check if the files exist
print(f"Model path exists: {os.path.exists(model_path)}")
print(f"Scaler path exists: {os.path.exists(scaler_path)}")
print(f"Encoder path exists: {os.path.exists(encoder_path)}")

# Load the model, scaler, and encoder
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    print("Model, scaler, and encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model, scaler, or encoder: {e}")

@main.route('/')
def index():
    form = UploadForm()
    return render_template('index.html', form=form)

@main.route('/predict', methods=['POST'])
def predict():
    form = UploadForm()
    if form.validate_on_submit():
        try:
            # Get the image file from the form
            image_file = form.image.data
            # Extract text from the image
            extracted_text = extract_text_from_image(image_file)
            # Preprocess the extracted text for model prediction
            features = preprocess_text(extracted_text, encoder, scaler)

            # Predict using the trained model
            prediction = model.predict(features)
            result = 'Allergen Detected' if prediction[0] == 1 else 'No Allergen Detected'

            return render_template('result.html', result=result, extracted_text=extracted_text)
        except Exception as e:
            return str(e)
    return render_template('index.html', form=form)
