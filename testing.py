import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def load_trained_model(model_path):
    '''
    Load the trained model from the specified path.
    '''
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    '''
    Preprocess the image to match the input format expected by the model.
    
    Parameters:
        - image_path: str, path to the image file
        - target_size: tuple, size to which the image is resized
    
    Returns:
        - preprocessed_image: numpy array, preprocessed image
    '''
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_flower_class(model, image, class_names, base_model=None):
    '''
    Predict the class of the flower image using the trained model.
    
    Parameters:
        - model: tf.keras.Model, trained model
        - image: numpy array, preprocessed image
        - class_names: list of str, class names
        - base_model: tf.keras.Model, base model for feature extraction (optional)
    
    Returns:
        - predicted_class: str, the predicted class name
        - predicted_proba: float, the probability of the predicted class
    '''
    if base_model is not None:
        # Extract features using the base model
        image = base_model.predict(image)
    
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_idx]
    predicted_proba = np.max(predictions, axis=1)[0]
    return predicted_class, predicted_proba

def main(image_path, model_path, class_names, is_accelerated=False):
    '''
    Main function to load the model, preprocess the image, make a prediction,
    and display the result.
    
    Parameters:
        - image_path: str, path to the image file
        - model_path: str, path to the trained model
        - class_names: list of str, class names
        - is_accelerated: bool, whether the model is for accelerated learning
    '''
    # Load the trained model
    model = load_trained_model(model_path)
    
    # If accelerated learning model, load the base model
    base_model = None
    if is_accelerated:
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Ensure the base model is not trainable
    
    # Preprocess the input image
    image = preprocess_image(image_path)
    
    # Make a prediction
    predicted_class, predicted_proba = predict_flower_class(model, image, class_names, base_model)
    
    # Display the result
    print(f"Predicted class: {predicted_class} with probability {predicted_proba:.2f}")

    # Plot the image
    original_image = load_img(image_path)
    plt.imshow(original_image)
    plt.title(f"Predicted: {predicted_class} ({predicted_proba:.2f})")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Define the paths
    image_path = "/home/imhaom/CAB420_Machine_Learning/CAB320/flower_testing_1.png"
    model_path = "/home/imhaom/CAB420_Machine_Learning/CAB320/trained_model_transfer.h5"  # or trained_model_transfer.h5"
    
    # Define the class names (ensure these match the classes used in training)
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    
    # Indicate if the model is for accelerated learning
    is_accelerated = False  # Set to True for accelerated learning model
    
    # Run the main function
    main(image_path, model_path, class_names, is_accelerated)