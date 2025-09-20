from src.models.base_model import BaseModel
from tensorflow.keras.applications import ResNet50V2
from src.utils.logging_setup import logger
from tensorflow.keras import models
from typing import Optional


class ResNetModel(BaseModel):
    """Concrete strategy for building ResNet50V2 model."""

    def __init__(self, img_size: int, include_top: bool = False, weights: Optional[str] = 'imagenet'):
        """
        Args:
            img_size (int): Input image size.
            include_top (bool): Whether to include the fully-connected layer at the top.
            weights (Optional[str]): Pretrained weights or None.
        """
        super().__init__(img_size)
        self.include_top = include_top
        self.weights = weights

    def build_model(self) -> models.Model:
        logger.info(f"Building ResNet50V2 model with input shape: {self.input_shape}, "
                    f"include_top={self.include_top}, weights={self.weights}")
        model = ResNet50V2(input_shape=self.input_shape, include_top=self.include_top, weights=self.weights)
        logger.info("ResNet50V2 model built successfully.")
        return model