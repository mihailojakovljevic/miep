import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

# Load the pre-trained model
model = load_model('model/mias_model.h5')  # Update this path to your saved model location

# Image dimensions
img_height, img_width = 224, 224

# Preprocess a single image (from PGM format to RGB)
def preprocess_image(image_path):
    # Read the image in grayscale
    pgm_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Convert to RGB
    rgb_image = cv2.cvtColor(pgm_image, cv2.COLOR_GRAY2RGB)
    # Resize the image to match input size of model
    resized_image = cv2.resize(rgb_image, (img_height, img_width))
    # Rescale pixel values
    resized_image = resized_image / 255.0
    return resized_image

# Batch process multiple images for inference
def load_and_preprocess_images(image_dir):
    images = []
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.pgm')]

    #print(image_paths)
    
    for image_path in image_paths:
        processed_image = preprocess_image(image_path)
        images.append(processed_image)

    # Convert list of images to numpy array and expand dimensions for batch processing
    return np.array(images)

# Run inference on all images in the provided directory
def run_inference(image_dir):
    # Preprocess the images
    images = load_and_preprocess_images(image_dir)
    
    # Run model inference
    predictions = model.predict(images)
    
    # Output predictions
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

# Main inference function
def make_prediction(model_path, img_path):
    """Load model, preprocess image, and make prediction."""
    # Load the pre-trained model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Preprocess the input image
    img_array = load_and_preprocess_images(img_path)
    #print(img_array)
    #print(f"Image preprocessed from {img_path}")
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = ['A', 'B', 'C', 'D']  # Assuming 4 classes as in your project
    
    # Output the prediction result
    print(f"Predicted class: {class_labels[predicted_class[0]]} (Class index: {predicted_class[0]})")
    return class_labels[predicted_class[0]]

if __name__ == "__main__":
    # Set up argument parsing
    #parser = argparse.ArgumentParser(description="Inference for breast density classification model.")
    #parser.add_argument('--model', type=str, required=True, help="Path to the pre-trained model (.h5 file)")
    #parser.add_argument('--images', type=str, required=True, help="Path to the image file for prediction")
    #args = parser.parse_args()

    make_prediction('model/mias_model.h5', 'data/images')

    # C:\Users\Thinkpad\Desktop\MIEP_IT66_2019\models\mias_model.h5

    # Run the prediction
    #make_prediction(args.model, args.images)

