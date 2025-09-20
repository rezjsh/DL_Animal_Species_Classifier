from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from src.utils.logging_setup import logger
from src.entity.config_entity import AugmentationConfig

class Augmentation:
    """
    A class to encapsulate and build the data augmentation pipeline.
    This follows the Builder design pattern.
    """
    def __init__(self, config: AugmentationConfig):
        logger.info("Initializing data augmentation with config: %s", config)   
        self.config = config

    def build_augmentation_layers(self) -> models.Sequential:
        """
        Builds and returns a Keras Sequential model for data augmentation.
        Only includes augmentation layers that are configured.
        """
        layers = []
        if self.config.horizontal_flip:
            layers.append(layers.RandomFlip("horizontal"))
        if self.config.vertical_flip:
            layers.append(layers.RandomFlip("vertical"))
        if self.config.rotation_range:
            layers.append(layers.RandomRotation(self.config.rotation_range))
        if self.config.brightness_factor:
            layers.append(layers.RandomBrightness(self.config.brightness_factor))
        if self.config.zoom_range:
            layers.append(layers.RandomZoom(self.config.zoom_range))
        if self.config.contrast_range:
            layers.append(layers.RandomContrast(self.config.contrast_range))
        if self.config.width_shift_range or self.config.height_shift_range:
            tx = self.config.height_shift_range if self.config.height_shift_range else 0.0
            ty = self.config.width_shift_range if self.config.width_shift_range else 0.0
            layers.append(layers.RandomTranslation(height_factor=tx, width_factor=ty))
        return models.Sequential(layers, name="augmentation_layers")
    

    def plot_and_save_augmentation_samples(
        self,
        dataset,
    ):
        """
        Visualizes original and augmented images side-by-side and saves the figure.
        Parameters:
        - dataset: A tf.data.Dataset yielding (images, labels)
        """
        augmentation_pipeline = self.build_augmentation_layers()
        
        for images, _ in dataset.take(1):
            original_images = images[:self.config.num_images]
            augmented_images = augmentation_pipeline(original_images)

            plt.figure(figsize=(self.config.num_images * 3, 6))
            for i in range(self.config.num_images):
                # Plot original image
                plt.subplot(2, self.config.num_images, i + 1)
                plt.imshow(tf.cast(original_images[i], tf.uint8).numpy())
                plt.title("Original")
                plt.axis('off')

                # Plot augmented image
                plt.subplot(2, self.config.num_images, i + 1 + self.config.num_images)
                plt.imshow(tf.cast(augmented_images[i], tf.uint8).numpy())
                plt.title("Augmented")
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(self.config.save_path)
            plt.show()
            break