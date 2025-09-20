from src.components.data_validation import DatasetValidator
from src.config.configuration import ConfigurationManager
from src.utils.logging_setup import logger

class DataValidationPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.data_validation = DatasetValidator(config=self.config.get_data_validation_config())

    def run_pipeline(self):
        try:
            logger.info("Starting data validation pipeline")
            validator = self.data_validation.validate()
            if validator:
                logger.info("Data validation successful")
                self.data_validation.save_report()
            else:
                logger.error("Data validation failed")
            logger.info("Data validation pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error in data validation pipeline: {e}")
            raise e

if __name__ == "__main__":
    config = ConfigurationManager()
    data_validation_pipeline = DataValidationPipeline(config=config)
    data_validation_pipeline.run_pipeline()