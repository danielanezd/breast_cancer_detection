import shap
import pandas as pd
from inference.explainer_base import ExplainerBase


class XGBoostExplainer(ExplainerBase):
    """
    Genera interpretaciones SHAP para un modelo XGBoost,
    recibiendo el modelo y el scaler directamente.
    """
    def __init__(self, model, scaler, feature_cols, label_list=None):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.label_list = label_list
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, X):
        # Asegura DataFrame con nombres de columnas
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_cols)
        # Escalado si existe scaler
        if self.scaler is not None:
            arr = X_df.values
            X_scaled = pd.DataFrame(self.scaler.transform(arr), columns=self.feature_cols)
        else:
            X_scaled = X_df

        shap_values = self.explainer.shap_values(X_scaled)

        # Plot summary: handle binary/single vs multiclass (list of arrays)
        if isinstance(shap_values, list):
            n_classes = len(shap_values)
            class_names = self.label_list if (self.label_list and len(self.label_list) == n_classes) else [f"class_{i}" for i in range(n_classes)]
            for i, (sv, name) in enumerate(zip(shap_values, class_names)):
                print(f"\nSHAP para clase {i} → {name}")
                shap.summary_plot(sv, X_df, feature_names=self.feature_cols, plot_size=(8, 6))
        else:
            shap.summary_plot(shap_values, X_df, feature_names=self.feature_cols, plot_size=(8, 6))

        return shap_values
