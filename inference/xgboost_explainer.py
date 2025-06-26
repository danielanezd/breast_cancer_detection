import shap
import pandas as pd
# Ajusta estos imports para usar el paquete completo:
from inference.classifier_predictor import ClassifierPredictor
from utils.constants import LABEL_MAP, MODELS_OUTPUT_PATH


class XGBoostExplainer:
    """
    Genera interpretaciones SHAP para un modelo XGBoost,
    recibiendo el modelo y el scaler directamente.
    """
    def __init__(self, model, scaler, feature_cols):
        self.model        = model
        self.scaler       = scaler
        self.feature_cols = feature_cols
        self.explainer    = shap.TreeExplainer(self.model)

    def explain(self, X):
        # Asegura DataFrame con nombres de columnas
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_cols)
        # Escalado si existe scaler
        if self.scaler:
            arr = X_df.values  # usar numpy array para evitar el error
            X_scaled = pd.DataFrame(self.scaler.transform(arr), columns=self.feature_cols)
        else:
            X_scaled = X_df
        # Devuelve valores SHAP
        return self.explainer.shap_values(X_scaled)