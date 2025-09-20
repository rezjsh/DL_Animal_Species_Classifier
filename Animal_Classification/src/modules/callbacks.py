import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from src.entity.config_entity import CustomCallbacksConfig
from src.utils.logging_setup import logger
import os


class CustomCallbacks(Callback):
    """
    Custom callback class for TensorFlow Keras to monitor training, validation,
    and model saving with early stopping and learning rate scheduling support.

    Args:
       config (CustomCallbacksConfig): Configuration object for custom callbacks.
    """

    def __init__(self, config: CustomCallbacksConfig):
        super().__init__()
        self.patience = config.patience
        self.checkpoint_dir = config.checkpoint_dir
        self.min_delta = config.min_delta
        self.lr_schedule = config.lr_schedule
        self.verbose = config.verbose

        self.best_val_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0


    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_val_loss = float('inf')
        self.stopped_epoch = 0
        if self.verbose:
            logger.info("Training started.")

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            logger.info(f"Epoch {epoch + 1} start.")

        # Apply learning rate scheduler if provided
        if self.lr_schedule is not None:
            old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            new_lr = self.lr_schedule(epoch)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose:
                logger.info(f"Learning rate updated from {old_lr:.6f} to {new_lr:.6f}.")

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss is not None:
            if current_val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = current_val_loss
                self.wait = 0
                self._save_checkpoint(epoch, logs)
                if self.verbose:
                    logger.info(f"Validation loss improved to {current_val_loss:.4f}. Saving checkpoint.")
            else:
                self.wait += 1
                if self.verbose:
                    logger.info(f"No improvement in validation loss for {self.wait} epochs.")

                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.model.stop_training = True
                    if self.verbose:
                        logger.info(f"Early stopping triggered at epoch {self.stopped_epoch}.")

        else:
            if self.verbose:
                logger.warning("Validation loss unavailable; early stopping is disabled.")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(f"Training stopped early at epoch {self.stopped_epoch}.")
        elif self.verbose:
            logger.info("Training completed.")

    def on_train_batch_begin(self, batch, logs=None):
        if self.verbose:
            logger.debug(f"Starting training batch {batch}.")

    def on_train_batch_end(self, batch, logs=None):
        if self.verbose:
            loss = logs.get('loss')
            accuracy = logs.get('accuracy')
            logger.debug(f"Finished training batch {batch}. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def on_test_begin(self, logs=None):
        if self.verbose:
            logger.info("Evaluation started.")

    def on_test_end(self, logs=None):
        if self.verbose:
            logger.info("Evaluation finished.")

    def on_predict_begin(self, logs=None):
        if self.verbose:
            logger.info("Prediction started.")

    def on_predict_end(self, logs=None):
        if self.verbose:
            logger.info("Prediction finished.")

    def _save_checkpoint(self, epoch, logs):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1:02d}_val_loss_{logs.get('val_loss', 0):.4f}.keras")
        self.model.save(checkpoint_path)
        if self.verbose:
            logger.info(f"Model checkpoint saved at: {checkpoint_path}")
