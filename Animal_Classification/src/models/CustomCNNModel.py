from tensorflow.keras import layers, models
from src.models.base_model import BaseModel
from src.utils.logging_setup import logger

class CustomCNNModel(BaseModel):
    """Concrete strategy for building a custom CNN model."""

    def build_model(self) -> models.Model:
        logger.info(f"Building CustomCNNModel with input shape {self.input_shape}")
        model = models.Sequential([
            layers.Rescaling(1. / 255, input_shape=self.input_shape, name="rescaling"),
            layers.Conv2D(32, (3, 3), activation='relu', name="conv1"),
            layers.MaxPooling2D((2, 2), name="maxpool1"),
            layers.Conv2D(64, (3, 3), activation='relu', name="conv2"),
            layers.MaxPooling2D((2, 2), name="maxpool2"),
            layers.Conv2D(128, (3, 3), activation='relu', name="conv3"),
            layers.MaxPooling2D((2, 2), name="maxpool3"),
            layers.Flatten(name="flatten"),
        ], name="CustomCNN")
        logger.info("CustomCNNModel built successfully.")
        return model