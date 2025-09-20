from src.modules.callbacks import CustomCallbacks
from src.components.model_trainer import ModelTraining
from src.entity.config_entity import ModelTrainingConfig
from src.utils.logging_setup import logger

class ModelTrainerPipeline:
    def __init__(self, config: ConfigurationManager, model):
        self.config_manager = config
        self.model_training_config = self.config_manager.get_model_training_config()
        self.model_trainer = ModelTraining(
            config=self.model_training_config,
            model=model
        )
        self.callbacks = [
            CustomCallbacks(self.config_manager.get_custom_callbacks_config())
        ]

    def run_pipeline(self, train_data, val_data):
        logger.info("Starting model training...")
        history = self.model_trainer.train(train_data, val_data, self.callbacks)
        logger.info("Model training completed.")
        self.model_trainer.save()
        logger.info("Model saved.")
        self.model_trainer.plot_training_history()
        logger.info("Model training history plotted.")
        return history