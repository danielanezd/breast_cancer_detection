import shap
import numpy as np
import pandas as pd
import pickle
import json

from shap.plots._waterfall import waterfall_legacy
from IPython.display import display
from inference.explainer_interface import ExplainerInterface


class SHAPExplainer(ExplainerInterface):
    def __init__(self, model_path="xgboost_modelo.pkl",
                 scaler_path="scaler.pkl",
                 features_path="features.json",
                 label_encoder_path="label_encoder.pkl"):
        """
        SHAP-based explainer for XGBoost models.
        """
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        with open(features_path, "r") as f:
            self.feature_names = json.load(f)

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.explainer = None

    def _prepare_input(self, raw_input: dict) -> pd.DataFrame:
        input_df = pd.DataFrame([raw_input], columns=self.feature_names)
        input_scaled = self.scaler.transform(input_df)
        return pd.DataFrame(input_scaled, columns=self.feature_names)

    def explain(self, raw_input: dict):
        input_scaled = self._prepare_input(raw_input)
        if self.explainer is None:
            self.explainer = shap.Explainer(self.model, input_scaled)
        return self.explainer(input_scaled)

    def show_image(self, shap_values, class_index=0, max_display=7):
        """
        Visualiza el gráfico SHAP únicamente si hay valores diferentes de cero.
        Evita errores con imágenes vacías o inválidas.
        """
        try:
            class_values = shap_values.values[0, :, class_index]
            if np.all(class_values == 0):
                print(f"⚠️ Todos los valores SHAP para la clase {class_index} son cero. Nada que mostrar.")
                return

            explanation = shap.Explanation(
                values=class_values,
                base_values=shap_values.base_values[0, class_index],
                data=shap_values.data[0],
                feature_names=shap_values.feature_names
            )

            display(waterfall_legacy(explanation, max_display=max_display))

        except Exception as e:
            print("❌ Error al renderizar imagen SHAP:")
            print(e)