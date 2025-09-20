import os
import matplotlib.pyplot as plt
from tensorflow.keras import models
from src.entity.config_entity import ModelTrainingConfig
from src.utils.logging_setup import logger
import tensorflow as tf

class ModelTraining:
    """
    Provides a structured interface for training, evaluating,
    saving Keras models, and plotting training metrics.

    Args:
        model (tf.keras.Model): The Keras model to train.
        config (ModelTrainingConfig): Configuration parameters for training.
    """

    def __init__(
        self,
        config: ModelTrainingConfig,
        model: models.Model):
        self.config = config
        self.model = model
        self.history = None

    def train(self, train_data, val_data, callbacks) -> tf.keras.callbacks.History:
        """
        Run model training with the configured datasets and callbacks.
        Returns:
            History object with training metrics.
        """
        logger.info("Starting model training...")
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=self.config.verbose,
            callbacks=[callbacks],
        )
        logger.info("Model training complete.")
        return self.history


    def save(self) -> None:
        """
        Save the trained model to disk.
        """
        save_path = os.path.join(self.config.save_dir, self.config.model_name)
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}.")

    def plot_training_history(self) -> None:
        """
        Plot and save training & validation accuracy and loss curves.
        """
        if self.history is None:
            logger.error("No training history found. Please train the model before plotting.")
            return

        plot_dir = self.config.save_plot_dir
        model_prefix = self.config.model_name or 'model'

        # Accuracy plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.history.history.get('accuracy', []), label='Train Accuracy')
        plt.plot(self.history.history.get('val_accuracy', []), label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        acc_plot_path = os.path.join(plot_dir, f'{model_prefix}_accuracy.png')
        plt.savefig(acc_plot_path)
        plt.close()
        logger.info(f"Saved accuracy plot to {acc_plot_path}")

        # Loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.history.history.get('loss', []), label='Train Loss')
        plt.plot(self.history.history.get('val_loss', []), label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        loss_plot_path = os.path.join(plot_dir, f'{model_prefix}_loss.png')
        plt.savefig(loss_plot_path)
        plt.close()
        logger.info(f"Saved loss plot to {loss_plot_path}")
