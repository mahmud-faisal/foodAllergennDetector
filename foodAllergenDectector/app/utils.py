# app/utils.py

import pytesseract
from PIL import Image
import numpy as np

def extract_text_from_image(image_file):
    # Convert the uploaded image file to a PIL Image object
    image = Image.open(image_file)
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(image)
    return text

def preprocess_text(text):
    # Implement text preprocessing logic here
    # For example, extracting features from the text for model prediction
    # This is a placeholder function and should be adapted to your use case
    # Example: Let's assume the text is space-separated ingredient values
    features = list(map(float, text.split()))
    return features
