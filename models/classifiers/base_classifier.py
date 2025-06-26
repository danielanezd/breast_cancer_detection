import os
from abc import ABC, abstractmethod
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


class BaseClassifier(ABC):
    def __init__(self, config=None):
        self.model = None  # To be defined in subclasses
        self.scaler = StandardScaler()
        self.feature_columns = [
                    "mean_intensity",
                    "std_intensity",
                    "width",
                    "height",
                    "assessment",
                    "breast_density",
                    "subtlety"
                ]

        self.label_column = "pathology"
        self.config = config
        self.label_map = {
            "BENIGN": 0,
            "BENIGN_WITHOUT_CALLBACK": 1,
            "MALIGNANT": 2
        }
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    @abstractmethod
    def build_model(self):
        """
        Abstract method that must be implemented by subclasses to create and return a model instance.
        Examples of model types include SVC, XGBClassifier, etc.
        """
        pass

    def fit(self, df: pd.DataFrame):
        """
        Train the classifier using the provided DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing feature columns and 'label' column
        """
        if self.model is None:
            self.model = self.build_model()

        X = df[self.feature_columns].values
        y = df[self.label_column].map(self.label_map).values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, df: pd.DataFrame):
        """
        Generate predictions for the input DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing feature columns
            
        Returns:
            np.ndarray: Array of predicted class labels
        """
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Calculate and return various classification metrics for the model's performance.
        
        Args:
            df (pd.DataFrame): DataFrame containing feature columns and 'label' column
            
        Returns:
            dict: Dictionary containing the following metrics:
                - accuracy: Overall accuracy score
                - precision: Macro-averaged precision score
                - recall: Macro-averaged recall score
                - f1_score: Macro-averaged F1 score
                - auc_roc: Area Under ROC Curve (if available)
                - per_view_metrics: Dictionary for view-specific metrics (empty by default)
                - per_view_predictions: Dictionary for view-specific predictions (empty by default)
        """
        X = df[self.feature_columns].values
        y = df[self.label_column].map(self.label_map).values
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        try:
            y_proba = self.model.predict_proba(X_scaled)
            auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        except (AttributeError, ValueError):
            auc = np.nan

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y, y_pred, average="macro", zero_division=0),
            "f1_score": f1_score(y, y_pred, average="macro", zero_division=0),
            "auc_roc": auc,
        }

    def save_model(self, path: str):
        """
        Save the model and scaler to disk using joblib.

        Args:
            path (str): Path where the model will be saved
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        print(f"Classifier model saved to: {path}")

    def load_model(self, path: str):
        """
        Load the model and scaler from disk using joblib.

        Args:
            path (str): Path from where the model will be loaded
        """
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        print(f"Classifier model loaded from: {path}")