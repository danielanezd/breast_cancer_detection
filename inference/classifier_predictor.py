from typing import Union
import joblib
import numpy as np


class ClassifierPredictor:
    """
    Predictor for classifiers like SVM or XGBoost using handcrafted features.
    This class performs inference from feature vectors (not raw images).
    """
    def __init__(self, strategy, model_path, label_list=None):
        """
        Args:
            strategy: A BaseClassifier instance (e.g., SVMClassifier or XGBoostClassifier)
            model_path (str): Path to the saved joblib file
        """
        self.strategy = strategy
        data = joblib.load(model_path)
        if isinstance(data, dict) and "model" in data and "scaler" in data:
            self.strategy.model = data["model"]
            self.strategy.scaler = data["scaler"]
        else:
            # Legacy: just a model, no scaler
            self.strategy.model = data
            self.strategy.scaler = None
        self.label_list = label_list or ["CLASS_0", "CLASS_1", "CLASS_2"]

    def predict_features(self, features: Union[np.ndarray, list]) -> dict:
        """
        Predict the class given a feature vector or matrix.
        Args:
            features: 1D array-like for a single sample or 2D array-like (n_samples, n_features)
        Returns:
            dict: {
                "prediction": int,
                "class_label": str,
                "confidence": float,
                "all_probs": np.ndarray
            }
        """
        features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if self.strategy.scaler is not None:
            features_scaled = self.strategy.scaler.transform(features)
        else:
            features_scaled = features
        probs = self.strategy.model.predict_proba(features_scaled)
        # If multiple samples, return the first by default to match prior API
        probs_row = probs[0]
        pred = int(np.argmax(probs_row))
        class_label = self.label_list[pred] if pred < len(self.label_list) else str(pred)
        return {
            "prediction": pred,
            "class_label": class_label,
            "confidence": round(float(probs_row[pred]), 3),
            "all_probs": np.round(probs_row, 3)
        }
