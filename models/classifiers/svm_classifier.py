from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from .base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    def __init__(self, search_hyperparameters: bool = False, config=None):
        super().__init__(config)
        self.search_hyperparameters = search_hyperparameters

    def build_model(self):
        return SVC(probability=True)

    def fit(self, df):
        X = df[self.feature_columns].values
        y = df[self.label_column].map(self.label_map).values

        X_scaled = self.scaler.fit_transform(X)
        print(self.feature_columns)
        print(self.label_column)
        print(X_scaled,y)
        if self.search_hyperparameters:
            print("Performing GridSearchCV for SVM hyperparameter tuning...")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'],
                'class_weight': [None, 'balanced']
            }
            
            svc = SVC(probability=True)
            grid = GridSearchCV(
                estimator=svc,
                param_grid=param_grid,
                scoring='f1_weighted',
                cv=cv,
                verbose=1,
                n_jobs=-1
            )

            grid.fit(X_scaled, y)

            print(" Mejor combinación encontrada:")
            print(grid.best_params_)
            print(f"F1 ponderado en validación cruzada: {grid.best_score_:.4f}")

            self.model = grid.best_estimator_
        else:
            self.model = self.build_model()
            self.model.fit(X_scaled, y)