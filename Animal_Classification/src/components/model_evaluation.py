import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logging_setup import logger
from src.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    """
    Robust class for evaluating ML classification models with extended metrics,
    confusion matrix plotting, and saving reports/images.

    Args:
        config (ModelEvaluationConfig): Configuration for model evaluation.
        model (tf.keras.Model | None): Trained Keras model or None to load from config.model_path.
        test_data (tuple or tf.data.Dataset): Test dataset or (X_test, y_test) tuple.
        class_names (list[str], optional): List of class labels.
    """

    def __init__(
        self,
        config: ModelEvaluationConfig,
        model: tf.keras.Model | None,
        test_data,
        class_names: list[str] | None = None,
    ):
        self.config = config
        self.model = model
        self.test_data = test_data
        self.class_names = class_names

        # Load model from path if no model provided
        if self.model is None:
            model_path = os.path.join(self.config.model_path, self.config.saved_model_name)
            if not model_path or not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        # Ensure save directory exists
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> dict | None:
        """
        Evaluate the model and produce metrics and reports.

        Returns:
            dict: Basic loss/accuracy metrics, or None if failed.
        """
        try:
            logger.info("Starting model evaluation...")

            if isinstance(self.test_data, tuple) and len(self.test_data) == 2:
                X_test, y_test = self.test_data
                preds_prob = self.model.predict(X_test)
                if preds_prob.shape[1] > 1:
                    preds = preds_prob.argmax(axis=1)
                else:
                    preds = (preds_prob > 0.5).astype(int).flatten()

                if self.class_names:
                    y_true = np.array([self.class_names[i] for i in y_test])
                    y_pred_labels = np.array([self.class_names[i] for i in preds])
                else:
                    y_true, y_pred_labels = y_test, preds

                # Basic metrics
                basic_metrics = dict(
                    zip(self.model.metrics_names, self.model.evaluate(X_test, y_test, verbose=0))
                )
                logger.info(f"Basic metrics: {basic_metrics}")

                # Classification report
                report = classification_report(y_true, y_pred_labels, target_names=self.class_names, zero_division=0)
                logger.info("Classification Report:\n" + report)
                self._save_report(report)

                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred_labels, labels=self.class_names)
                self._save_confusion_matrix(cm)

                # ROC-AUC (multi-class or binary)
                if preds_prob.shape[1] > 1:
                    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=preds_prob.shape[1])
                    auc = roc_auc_score(y_test_onehot, preds_prob, average='macro', multi_class='ovr')
                else:
                    auc = roc_auc_score(y_test, preds_prob)
                logger.info(f"ROC AUC Score: {auc:.4f}")

                return basic_metrics

            else:
                # test_data is tf.data.Dataset or similar
                results = self.model.evaluate(self.test_data, verbose=1)
                metrics = dict(zip(self.model.metrics_names, results))
                logger.info(f"Evaluation metrics: {metrics}")
                return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None

    def _save_report(self, report_text: str) -> None:
        report_path = Path(self.config.save_dir) / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Saved classification report to {report_path}")


    def _save_confusion_matrix(self, cm: np.ndarray) -> None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cmap='Blues',
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_path = Path(self.config.save_dir) / 'confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {cm_path}")
