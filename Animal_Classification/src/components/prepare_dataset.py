import tensorflow as tf
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

        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.data_dir,
            labels='inferred',
            label_mode='int',
            validation_split=self.config.validation_split,
            shuffle=True,
            subset='training',
            seed=42,
            image_size=(self.config.img_size, self.config.img_size),
            batch_size=self.config.batch_size
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.data_dir,
            labels='inferred',
            label_mode='int',
            validation_split=self.config.validation_split,
            subset='validation',
            seed=42,
            image_size=(self.config.img_size, self.config.img_size),
            batch_size=self.config.batch_size
        )
        logger.info("Data loading complete")
        # Get class names before caching and prefetching
        class_names = train_ds.class_names
        return train_ds, val_ds, class_names

    def _prepare_datasets(self, train_ds, val_ds):
        """Prepares datasets for training."""
        val_batches = tf.data.experimental.cardinality(val_ds)
        test_ds = val_ds.take(val_batches // 2)
        val_ds = val_ds.skip(val_batches // 2)

        # Optimize datasets for performance
        train_ds = train_ds.cache().shuffle(self.config.shuffle_buffer_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
        logger.info("Dataset preparation complete")
        return train_ds, val_ds, test_ds

    def load_and_prepare(self):
        """Main method to load and prepare all datasets."""
        logger.info("Loading and preparing datasets")
        train_ds, val_ds, class_names = self._load_data()
        train_ds, val_ds, test_ds = self._prepare_datasets(train_ds, val_ds)
        logger.info(f"Found {len(class_names)} classes: {class_names}")
        return train_ds, val_ds, test_ds, class_names