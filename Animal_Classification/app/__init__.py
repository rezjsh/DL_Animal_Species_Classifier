import os
from flask import Flask
from .predictor import AnimalPredictor
from .routes import register_routes

def create_app(model_path='trained_model.keras', upload_folder='uploads', model_choice='EfficientNetB0'):
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = upload_folder
    os.makedirs(upload_folder, exist_ok=True)

    predictor = AnimalPredictor(model_path, label_lang="en", confidence_threshold=0.5, model_choice=model_choice)
    register_routes(app, predictor)

    return app
