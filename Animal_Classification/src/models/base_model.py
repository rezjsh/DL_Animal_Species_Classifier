from abc import ABC, abstractmethod
from typing import Optional
from tensorflow.keras import models
from src.utils.logging_setup import logger


class BaseModel(ABC):
    """
    Abstract base class for all model building strategies.
    Defines a consistent interface for building models.
    
    Args:
        img_size (int): The height and width of input images (assumed square).
    """

    def __init__(self, img_size: int):
        if not isinstance(img_size, int) or img_size <= 0:
            raise ValueError(f"img_size must be a positive integer; got {img_size}")
        self.img_size = img_size

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Returns the expected input shape. Can be overridden by subclasses."""
        return (self.img_size, self.img_size, 3)

    @abstractmethod
    def build_model(self) -> models.Model:
        """
        Build and return the Keras model instance.
        Must be implemented by subclasses.
        """
        pass
