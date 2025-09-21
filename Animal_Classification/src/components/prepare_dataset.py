import tensorflow as tf
import os
from pathlib import Path
from src.utils.logging_setup import logger
from src.entity.config_entity import DataPreparationConfig


class PrepareDataset:
    """
    A class to handle data loading, splitting, and preprocessing.
    This follows the Facade design pattern.
    """
    def __init__(self, config: DataPreparationConfig):
        logger.info("Initializing dataset preparation")
        self.config = config

    def _load_data(self):
        """Loads the Animals-10 dataset."""
        if not self.config.data_dir.exists():
            raise FileNotFoundError(f"Data directory '{self.config.data_dir}' not found.")

        logger.info(f"Loading images from directory: {self.config.data_dir}")

        # Dynamically determine class names from existing subdirectories
        available_classes = [d.name for d in self.config.data_dir.iterdir() if d.is_dir()]
        available_classes.sort() # Sort to ensure consistent order

        if not available_classes:
            raise ValueError(f"No class directories found in {self.config.data_dir}. Please check data ingestion.")

        logger.info(f"Found {len(available_classes)} available classes: {available_classes}")

        # Use available classes to load the datasets
        try:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                self.config.data_dir,
                labels='inferred',
                label_mode='int',
                validation_split=self.config.validation_split,
                shuffle=True,
                subset='training',
                seed=42,
                image_size=(self.config.img_size, self.config.img_size),
                batch_size=self.config.batch_size,
                class_names=available_classes # Use available classes found
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.config.data_dir,
                labels='inferred',
                label_mode='int',
                validation_split=self.config.validation_split,
                subset='validation',
                seed=42,
                image_size=(self.config.img_size, self.config.img_size),
                batch_size=self.config.batch_size,
                class_names=available_classes # Use available classes found
            )

            if tf.data.experimental.cardinality(train_ds).numpy() == 0:
                 raise ValueError("Training dataset is empty. Please check data source and validation.")
            # It's possible for validation/test to be empty if split is too high for small datasets
            if tf.data.experimental.cardinality(val_ds).numpy() == 0:
                 logger.warning("Validation dataset is empty.")


            logger.info("Data loading complete")

            return train_ds, val_ds, available_classes # Return available classes


        except Exception as e:
             logger.error(f"Error loading dataset: {e}")
             raise e


    def _prepare_datasets(self, train_ds, val_ds):
        """Prepares datasets for training."""
        # Split validation data into validation and test sets
        # Ensure val_ds is not empty before splitting
        val_batches = tf.data.experimental.cardinality(val_ds)
        if val_batches.numpy() > 1: # Need at least 2 batches to split into val and test
            test_ds = val_ds.take(val_batches // 2)
            val_ds = val_ds.skip(val_batches // 2)
            logger.info(f"Split validation data into {val_batches.numpy() // 2} test batches and {val_batches.numpy() - (val_batches.numpy() // 2)} validation batches.")
        else:
             logger.warning("Not enough validation batches to split into test set. Test set will be empty.")
             test_ds = tf.data.Dataset.from_tensor_slices((tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.int32))) # Create empty dataset


        # Optimize datasets for performance
        train_ds = train_ds.shuffle(self.config.shuffle_buffer_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        logger.info("Dataset preparation complete")
        return train_ds, val_ds, test_ds

    def load_and_prepare(self):
        """Main method to load and prepare all datasets."""
        logger.info("Loading and preparing datasets")
        train_ds, val_ds, class_names = self._load_data()
        train_ds, val_ds, test_ds = self._prepare_datasets(train_ds, val_ds)
        logger.info(f"Found {len(class_names)} classes during preparation: {class_names}")
        return train_ds, val_ds, test_ds, class_names