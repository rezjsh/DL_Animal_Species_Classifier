
class ModelPredictionPipeline:
    def __init__(self, config: ModelPredictionConfig, model, class_names):
        self.config = config
        self.model = model
        self.class_names = class_names

    def run_pipeline(self, inputs):
        logger.info("Running model prediction pipeline")
        get_model_prediction_config = self.config.get_model_prediction_config()
        predictor = ModelPredictor(config=get_model_prediction_config, model=self.model, class_names=self.class_names)
        predictions = predictor.predict(inputs=inputs)
        predictor.export_predictions_to_csv(inputs=inputs)
        logger.info(f"Model predictions: {predictions}")
        return predictions
