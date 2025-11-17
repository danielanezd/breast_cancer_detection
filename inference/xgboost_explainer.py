import shap
import pandas as pd
import matplotlib.pyplot as plt
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

    def explain(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_cols)
        
        if self.scaler is not None:
            arr = X_df.values
            X_scaled = pd.DataFrame(self.scaler.transform(arr), columns=self.feature_cols)
        else:
            X_scaled = X_df

        explainer = shap.TreeExplainer(self.model)

        shap_values = explainer(X_scaled)
        
        
        if len(shap_values.values.shape) == 3:
            n_classes = shap_values.values.shape[2]
            class_names = self.label_list if (self.label_list and len(self.label_list) == n_classes) else [f"Class {i}" for i in range(n_classes)]
            
            for i in range(n_classes):
                print(f"\n{'='*60}")
                print(f"SHAP Summary Plot para: {class_names[i]}")
                print(f"{'='*60}")
                
                class_shap = shap.Explanation(
                    values=shap_values.values[:, :, i],
                    base_values=shap_values.base_values[:, i] if len(shap_values.base_values.shape) > 1 else shap_values.base_values,
                    data=shap_values.data,
                    feature_names=self.feature_cols
                )
                
                plt.figure()
                shap.plots.beeswarm(class_shap, show=False, max_display=len(self.feature_cols))
                plt.title(f"SHAP Feature Importance - {class_names[i]}")
                plt.tight_layout()
                plt.show()
                
        else:
            print(f"\n{'='*60}")
            print("SHAP Summary Plot")
            print(f"{'='*60}")
            
            plt.figure()
            shap.plots.beeswarm(shap_values, show=False, max_display=len(self.feature_cols))
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            plt.show()

        return shap_values