import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from src.utils.logging_setup import logger

class AnimalPredictor:
    def __init__(self, 
        model_path,
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.0,
        verbose: bool = True,
        label_lang: str = "en",
        model_choice: str = "EfficientNetB0"):
        """
        Initialize the predictor by loading the trained model.
        """
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.label_lang = label_lang
        self.model_choice = model_choice
        try:
            self.model = load_model(model_path)
            self.class_names = class_names if class_names else [
                'cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
                'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
            ]
            self.translation_map = {
                'cane': 'dog',
                'cavallo': 'horse',
                'elefante': 'elephant',
                'farfalla': 'butterfly',
                'gallina': 'chicken',
                'gatto': 'cat',
                'mucca': 'cow',
                'pecora': 'sheep',
                'ragno': 'spider',
                'scoiattolo': 'squirrel'
            }
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None


    def _translate_label(self, label):
        """Translates a label to English if the label_lang is 'en'."""
        if self.label_lang == "en" and label in self.translation_map:
            return self.translation_map[label]
        return label


    def predict(self, image_path):
        """
        Load an image, preprocess it, and make a prediction.
        """
        if self.model is None:
            logger.error("Model not loaded. Please train a model first.")
            return
        
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            # Apply appropriate preprocessing based on the model choice
            if self.model_choice == 'EfficientNetB0':
                img_array = efficientnet_preprocess(img_array)
            elif self.model_choice == 'ResNet50V2':
                img_array = resnet_preprocess(img_array)
            else: # Assuming CustomCNN or other models use simple rescaling
                img_array /= 255.0

            predictions = self.model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_name_italian = self.class_names[predicted_index]
            predicted_name_english = self._translate_label(predicted_name_italian)
            confidence = predictions[0][predicted_index] * 100

            return f"Predicted: {predicted_name_english} ({confidence:.2f}%)"
        except Exception as e:
            return f"Error during prediction: {e}"
