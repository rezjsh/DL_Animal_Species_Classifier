from src.modules.augmentation import Augmentation
from src.entity.config_entity import AnimalSpeciesClassifierModelConfig
from src.models import ModelFactory
from src.modules.callbacks import CustomCallbacks
from tensorflow.keras import layers, models
import tensorflow as tf
from src.utils.logging_setup import logger


class AnimalSpeciesClassifierModel:
    """
    Model class for animal species classification encompassing model building,
    compilation, and optional augmentation.
    """

    def __init__(self, config: AnimalSpeciesClassifierModelConfig, augmentation: Augmentation, num_classes: int):
        """
        Initializes the classifier model with given configuration and augmentation.

        Args:
            config (AnimalSpeciesClassifierModelConfig): Configuration object containing
                model parameters such as image size, model type, learning rate, etc.
            augmentation (Augmentation): Augmentation pipeline to be applied to inputs.
        """
        self.config = config
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.model = None

    def create_model(self):
        """
        Helper method to build and compile the model.
        """
        logger.info("Starting to build the model...")
        self.model = self._build_model()
        logger.info("Model built successfully. Now compiling...")
        self._compile_model()
        logger.info("Model compilation complete.")
        return self.model

    def _build_model(self) -> tf.keras.Model:
        """
        Builds the Keras model architecture based on the specified base model choice
        and configuration settings.

        Returns:
            tf.keras.Model: The constructed Keras model.
        """
        inputs = layers.Input(shape=(self.config.img_size, self.config.img_size, 3), name="input_layer")

        # Apply augmentation if enabled
        if self.config.augmentation_enabled:
            logger.info("Augmentation enabled: Applying augmentation to inputs.")
            augmentation_layers = self.augmentation.build_augmentation_layers()
            x = augmentation_layers(inputs)
        else:
            x = inputs

        # Load base model from factory
        logger.info(f"Loading base model: {self.config.model_choice}")
        base_model = ModelFactory.get_base_model(self.config.model_choice, self.config.img_size)

        # If using a pretrained model other than CustomCNN, freeze base layers
        if self.config.model_choice != 'CustomCNN':
            base_model.trainable = False
            logger.info(f"Freezing base model layers for {self.config.model_choice}")
            if self.config.num_layers_to_unfreeze > 0:
                total_layers = len(base_model.layers)
                logger.info(f"Total layers in base model: {total_layers}")
                layers_to_unfreeze = base_model.layers[-self.config.num_layers_to_unfreeze:]

                for layer in layers_to_unfreeze:
                    layer.trainable = True
                logger.info(f"Unfroze the last {self.config.num_layers_to_unfreeze} layers for fine-tuning.")
            else:
                logger.info("No layers unfrozen; entire base model remains frozen.")

        # Forward pass through base model with training flag for CustomCNN
        x = base_model(x, training=(self.config.model_choice == 'CustomCNN'))

        # Add pooling layer if not using CustomCNN to reduce spatial dimensions
        if self.config.model_choice != 'CustomCNN':
            x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

        # Add dropout for regularization
        x = layers.Dropout(0.2, name="dropout_layer")(x)

        # Final softmax classification layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name="output_layer")(x)

        # Create Keras model instance
        model = models.Model(inputs=inputs, outputs=outputs, name="AnimalSpeciesClassifier")

        logger.info("Model architecture summary:")
        model.summary(print_fn=logger.info)

        return model

    def _compile_model(self):
        """
        Compiles the Keras model using Adam optimizer and sparse categorical cross-entropy
        loss, suitable for multi-class classification tasks.
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        logger.info(f"Compiled model with Adam optimizer and learning rate {self.config.learning_rate}.")