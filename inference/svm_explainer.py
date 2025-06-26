import shap
import pandas as pd
# Ajusta estos imports para usar el paquete completo:
from inference.classifier_predictor import ClassifierPredictor
from utils.constants import LABEL_MAP, MODELS_OUTPUT_PATH
from sklearn.cluster import KMeans


class svmExplainer:
    """
    Genera interpretaciones SHAP para un modelo svmExplainer,
    recibiendo el modelo y el scaler directamente.
    """
    def __init__(self, model, scaler, feature_cols):
        self.model        = model
        self.scaler       = scaler
        self.feature_cols = feature_cols
        self.explainer    = shap.TreeExplainer(self.model)

    def explain(self, X_sample):
        # Asegura DataFrame con nombres de columnas
        X_df = X_sample.copy() if isinstance(X_sample, pd.DataFrame) else pd.DataFrame(X_sample, columns=self.feature_cols)
        if self.scaler:
            arr = X_df.values  # usar numpy array para evitar el error
            X_scaled = pd.DataFrame(self.scaler.transform(arr), columns=self.feature_cols)
        else:
            X_scaled = X_df   
        background = KMeans(n_clusters=10, random_state=42).fit(X_scaled).cluster_centers_
        predict_proba = self.model.predict_proba
        clases = ['BENIGN_WITHOUT_CALLBACK', 'BENIGN', 'MALIGNANT']

        # Escalado si existe scaler
        for i, clase in enumerate(clases):
            print(f"\n SHAP para clase {i} → {clase}")

            # Función que devuelve solo la probabilidad de la clase i
            def prob_clase_i(X):
                return predict_proba(X)[:, i]
            

            # Crear un nuevo explicador SHAP solo para esa clase
            explainer_i = shap.KernelExplainer(prob_clase_i, background)

            # Calcular valores SHAP solo para esa clase
            shap_values_i = explainer_i.shap_values(X_sample)

            # Visualizar el resumen para esa clase
            shap.summary_plot(shap_values_i, X_df, feature_names=self.feature_cols, plot_size=(8, 6))