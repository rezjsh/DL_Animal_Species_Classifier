from src.models import CustomCNNModel, EfficientNetModel, ResNetModel
from src.utils.logging_setup import logger
from tensorflow.keras import models

class ModelFactory:
    """
    Factory to create and configure different base models.
    Uses a registry of model strategies for easy extensibility.
    """

    MODEL_REGISTRY = {
        'EfficientNetB0': EfficientNetModel,
        'ResNet50V2': ResNetModel,
        'CustomCNN': CustomCNNModel
    }

    @staticmethod
    def get_base_model(model_choice: str, img_size: int, **kwargs) -> models.Model:
        """
        Selects and returns a base model instance based on the provided model_choice.

        Args:
            model_choice (str): Name of the model to build. Must be in MODEL_REGISTRY.
            img_size (int): Input image size (height and width).

        Keyword Args:
            Additional keyword arguments forwarded to model constructors (e.g., include_top, weights).

        Returns:
            models.Model: An uncompiled Keras Model instance.

        Raises:
            ValueError: If model_choice is not supported.
        """
        if model_choice not in ModelFactory.MODEL_REGISTRY:
            valid_choices = list(ModelFactory.MODEL_REGISTRY.keys())
            message = f"Model '{model_choice}' not supported. Valid options: {valid_choices}"
            logger.error(message)
            raise ValueError(message)

        model_class = ModelFactory.MODEL_REGISTRY[model_choice]
        logger.info(f"Creating model using strategy: {model_choice}")

        model_instance = model_class(img_size, **kwargs)
        model = model_instance.build_model()

        logger.info(f"Model {model_choice} created successfully.")
        return model