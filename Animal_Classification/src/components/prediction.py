import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
from src.entity.config_entity import ModelPredictionConfig
from src.utils.logging_setup import logger

translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}


class ModelPredictor:
    """
    Class for running predictions on new data, supports:
    - optional model loading,
    - bidirectional label translation,
    - confidence thresholding,
    - batch prediction with tqdm progress,
    - exporting prediction CSV reports.

    Args:
        config (ModelPredictionConfig): Configuration for prediction.
        model (tf.keras.Model | None): Loaded model or None to load from config.
        class_names (list[str], optional): List of class labels.
        confidence_threshold (float): Minimum confidence for accepted predictions.
        verbose (bool): Whether to show progress bar and logs.
        label_lang (str): Language for returned labels ("en" or "it").
    """

    def __init__(
        self,
        config: ModelPredictionConfig,
        model: tf.keras.Model | None = None,
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.0,
        verbose: bool = False,
        label_lang: str = "en",
    ):
        self.config = config
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.label_lang = label_lang

        if model is None:
            if not config.model_path or not Path(config.model_path).exists():
                raise FileNotFoundError(f"Model file not found at {config.model_path}")
            if verbose:
                logger.info(f"Loading model from {config.model_path}")
            self.model = tf.keras.models.load_model(config.model_path)
        else:
            self.model = model

    def _translate_labels(self, labels):
        if 'translate' in globals() and self.label_lang != "en":
            if self.label_lang == "it":
                rev_translate = {v: k for k, v in translate.items()}
                return [rev_translate.get(label, label) for label in labels]
            # Add here more languages if supported
        return labels


    def predict(self, inputs, return_labels: bool = True):
        # tqdm only for iterables/datasets
        from tqdm import tqdm
        batches = tqdm(inputs, disable=not self.verbose) if hasattr(inputs, "__iter__") and self.verbose else [inputs]
        all_preds = []
        all_confs = []
        for batch in batches:
            x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
            preds_prob = self.model.predict(x_batch, verbose=0)
            confs = np.max(preds_prob, axis=1) if preds_prob.shape[1] > 1 else preds_prob.flatten()
            preds_indices = preds_prob.argmax(axis=1) if preds_prob.shape[1] > 1 else (preds_prob > 0.5).astype(int).flatten()
            mask = confs >= self.confidence_threshold
            filtered_preds = preds_indices[mask]
            filtered_confs = confs[mask]
            all_preds.extend(filtered_preds)
            all_confs.extend(filtered_confs)

        if return_labels and self.class_names:
            all_preds = self._translate_labels([self.class_names[i] for i in all_preds])

        return all_preds, all_confs

    def export_predictions_to_csv(
        self,
        inputs,
        filenames: list[str] | None = None,
        csv_path: Path | str = "predictions.csv",
        return_labels: bool = True,
    ):
        preds, confs = self.predict(inputs, return_labels=return_labels)

        if filenames and len(filenames) != len(preds):
            logger.warning("Mismatch between filenames length and predictions count; ignoring filenames.")
            filenames = None

        df = pd.DataFrame({
            "filename": filenames if filenames else [f"sample_{i}" for i in range(len(preds))],
            "prediction": preds,
            "confidence": confs,
        })
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")
        return csv_path