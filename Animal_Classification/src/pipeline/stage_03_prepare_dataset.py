
from src.components.prepare_dataset import PrepareDataset
from src.entity.config_entity import DataPreparationConfig
from src.utils.logging_setup import logger

class PrepareDatasetPipeline:
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.get_dataset_preparation = PrepareDataset(self.config.get_dataset_preparation_config())
        logger.info("Initialized PrepareDatasetPipeline with config: %s", self.config)

    def run_pipeline(self):
        logger.info("Running dataset preparation pipeline")
        train_ds, val_ds, test_ds, class_names = self.get_dataset_preparation.load_and_prepare()
        logger.info("Dataset preparation pipeline completed")
        return train_ds, val_ds, test_ds, class_names