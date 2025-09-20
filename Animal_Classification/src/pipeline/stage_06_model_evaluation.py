from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.logging_setup import logger

class ModelEvaluationPipeline:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def run_pipeline(self, model, test_data, class_names):
        logger.info("Running model evaluation pipeline")
        get_model_evaluation_config = self.config.get_model_evaluation_config()
        evaluator = ModelEvaluation(config = get_model_evaluation_config, model=model, test_data = test_data, class_names=class_names)
        metrics = evaluator.evaluate()
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics

if __name__ == "__main__":
    logger.info("Model Evaluation Pipeline")
    config_manager = ConfigurationManager()
    model_evaluation_config = config_manager.get_model_evaluation_config()
    model_evaluation_pipeline = ModelEvaluationPipeline(config=model_evaluation_config)
    model_evaluation_pipeline.run_pipeline(model=model, test_data=test_data, class_names=class_names)
    logger.info("Model Evaluation Pipeline completed")