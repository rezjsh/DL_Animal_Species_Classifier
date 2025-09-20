from src.components.model import AnimalSpeciesClassifierModel
from src.modules.augmentation import Augmentation
from src.config.configuration import ConfigurationManager
from src.utils.logging_setup import logger


class AnimalSpeciesClassifierModelPipeline:
    """
    Pipeline class for the animal species classification.
    """
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.model_config = self.config.get_animal_species_classifier_model_config()
        self.augmentation = Augmentation(self.config.get_augmentation_config())

    def run_pipeline(self, class_names: list):
        num_classes = len(class_names)
        logger.info("Starting animal species classification pipeline...")
        model = AnimalSpeciesClassifierModel(self.model_config, self.augmentation, num_classes).create_model()
        logger.info("Animal species classification model created.")
        return model
        
