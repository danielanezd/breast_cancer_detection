import pandas as pd
import shap
from sklearn.cluster import KMeans

from inference.explainer_base import ExplainerBase


class SVMExplainer(ExplainerBase):
    """
    SHAP explanations for an SVM-like classifier using KernelExplainer.

    Parameters
    - model: estimator with predict_proba
    - scaler: optional fitted scaler with transform
    - feature_cols: list of feature names
    - label_list: optional class names for plotting
    """

    def __init__(self, model, scaler, feature_cols, label_list=None):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.label_list = label_list

    def explain(self, X_sample):
        # Ensure DataFrame with column names
        X_df = X_sample.copy() if isinstance(X_sample, pd.DataFrame) else pd.DataFrame(X_sample, columns=self.feature_cols)

        # Scale if scaler exists
        if self.scaler is not None:
            arr = X_df.values
            X_scaled = pd.DataFrame(self.scaler.transform(arr), columns=self.feature_cols)
        else:
            X_scaled = X_df

        # Background for KernelExplainer (KMeans centers in scaled space)
        n_bg = min(10, max(1, len(X_scaled)))
        background = KMeans(n_clusters=n_bg, random_state=42).fit(X_scaled).cluster_centers_

        predict_proba = self.model.predict_proba
        n_classes = predict_proba(X_scaled[:1]).shape[1]
        class_names = (
            self.label_list if (self.label_list and len(self.label_list) == n_classes)
            else [f"class_{i}" for i in range(n_classes)]
        )

        shap_values_all = []
        for i, clase in enumerate(class_names):
            print(f"\nSHAP para clase {i} → {clase}")

            # One-vs-rest probability function for class i
            def prob_clase_i(X):
                return predict_proba(X)[:, i]

            explainer_i = shap.KernelExplainer(prob_clase_i, background)

            # Compute SHAP on scaled inputs to match background/model domain
            shap_values_i = explainer_i.shap_values(X_scaled)
            shap_values_all.append(shap_values_i)

            # Plot summary for this class using original feature names
            shap.summary_plot(shap_values_i, X_df, feature_names=self.feature_cols, plot_size=(8, 6))

        return shap_values_all

