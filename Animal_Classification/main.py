from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_02_data_validation import DataValidationPipeline
from src.pipeline.stage_03_prepare_dataset import PrepareDatasetPipeline
from src.pipeline.stage_04_model import AnimalSpeciesClassifierModelPipeline
from src.pipeline.stage_05_model_trainer import ModelTrainerPipeline
from src.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline
from src.pipeline.stage_07_model_prediction import ModelPredictionPipeline
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        setup_gpu_memory_growth()

        # --- Data Ingestion Stage ---
        STAGE_NAME = "Data Ingestion Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline(config=config_manager)
        data_ingestion_pipeline.run_pipeline()

        # --- Data Validation Stage ---
        STAGE_NAME = "Data Validation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_validation_pipeline = DataValidationPipeline(config=config_manager)
        data_validation_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Dataset Preparation Stage ---
        STAGE_NAME = "Dataset Preparation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        dataset_preparation_pipeline = PrepareDatasetPipeline(config=config_manager)
        train_ds, val_ds, test_ds, class_names = dataset_preparation_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Creating Model ---
        STAGE_NAME = "Create Model"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_pipeline = AnimalSpeciesClassifierModelPipeline(config=config_manager)
        model = model_pipeline.run_pipeline(class_names=class_names)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        print('model is', model)

        # --- Model Training Stage ---
        STAGE_NAME = "Model Training Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer_pipeline = ModelTrainerPipeline(config=config_manager, model=model)
        history = model_trainer_pipeline.run_pipeline(train_data=train_ds, val_data=val_ds)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Model Evaluation Stage ---
        STAGE_NAME = "Model Evaluation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evaluation_pipeline = ModelEvaluationPipeline(config=config_manager)
        model_evaluation_pipeline.run_pipeline(model=model, test_data=test_ds, class_names=class_names)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Model Prediction Stage ---
        STAGE_NAME = "Model Prediction Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_prediction_pipeline = ModelPredictionPipeline(config=config_manager, model=model, class_names=class_names)
        model_prediction_pipeline.run_pipeline(inputs=test_ds)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.error(f"Error occurred during {STAGE_NAME} stage: {e}")
        raise e

