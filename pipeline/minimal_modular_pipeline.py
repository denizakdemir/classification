import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import optuna
import joblib
from pipeline.base_pipeline import BasePipeline

class MinimalPreprocessing:
    def __init__(self, impute_strategy='mean', scaling='standard'):
        self.impute_strategy = impute_strategy
        self.scaling = scaling
        self.imputer = None
        self.scaler = None

    def fit_transform(self, X):
        if self.impute_strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif self.impute_strategy == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif self.impute_strategy == 'knn':
            self.imputer = KNNImputer()
        else:
            raise ValueError(f"Unknown impute_strategy: {self.impute_strategy}")
        X_imp = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        if self.scaling == 'standard':
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X_imp), columns=X_imp.columns)
            return X_scaled
        else:
            return X_imp

    def transform(self, X):
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        if self.scaler:
            X_scaled = pd.DataFrame(self.scaler.transform(X_imp), columns=X_imp.columns)
            return X_scaled
        else:
            return X_imp

class MinimalFeatureEngineering:
    def __init__(self, add_polynomial=False):
        self.add_polynomial = add_polynomial

    def fit_transform(self, X):
        if self.add_polynomial:
            X = X.copy()
            for col in X.columns:
                X[f'{col}_squared'] = X[col] ** 2
        return X

    def transform(self, X):
        return self.fit_transform(X)  # stateless for demo

class MinimalModel:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)

class MinimalPipeline(BasePipeline):
    def __init__(self, config=None):
        self.config = config
        self.preprocessing = None
        self.fe = None
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, config=None, trial=None):
        if config is not None:
            self.config = config
        # Hyperparameters from trial or config
        impute_strategy = trial.suggest_categorical('impute_strategy', self.config['preprocessing']['impute_strategy']) if trial else self.config['preprocessing']['impute_strategy'][0]
        scaling = trial.suggest_categorical('scaling', self.config['preprocessing']['scaling']) if trial else self.config['preprocessing']['scaling'][0]
        add_poly = trial.suggest_categorical('add_polynomial', self.config['feature_engineering']['add_polynomial']) if trial else self.config['feature_engineering']['add_polynomial'][0]
        n_estimators = trial.suggest_categorical('n_estimators', self.config['model']['n_estimators']) if trial else self.config['model']['n_estimators'][0]

        self.preprocessing = MinimalPreprocessing(impute_strategy, scaling)
        self.fe = MinimalFeatureEngineering(add_poly)
        self.model = MinimalModel(n_estimators)

        X_train_prep = self.preprocessing.fit_transform(X_train)
        X_train_feat = self.fe.fit_transform(X_train_prep)
        self.model.fit(X_train_feat, y_train)

        if X_val is not None and y_val is not None:
            X_val_prep = self.preprocessing.transform(X_val)
            X_val_feat = self.fe.transform(X_val_prep)
            val_pred = self.model.predict_proba(X_val_feat)
            auc = roc_auc_score(y_val, val_pred)
            return auc
        return None

    def predict(self, X):
        X_prep = self.preprocessing.transform(X)
        X_feat = self.fe.transform(X_prep)
        return self.model.predict(X_feat)

    def predict_proba(self, X):
        X_prep = self.preprocessing.transform(X)
        X_feat = self.fe.transform(X_prep)
        return self.model.predict_proba(X_feat)

    def evaluate(self, X, y):
        preds = self.predict_proba(X)
        return {'roc_auc': roc_auc_score(y, preds)}

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

# Example usage
if __name__ == '__main__':
    # Config with search spaces
    config = {
        'preprocessing': {
            'impute_strategy': ['mean', 'median', 'knn'],
            'scaling': ['standard', 'none']
        },
        'feature_engineering': {
            'add_polynomial': [True, False]
        },
        'model': {
            'n_estimators': [100, 200]
        }
    }
    # Load data
    df = pd.read_csv('datasets/heart_train.csv')
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        pipe = MinimalPipeline(config)
        auc = pipe.fit(X_train, y_train, X_val, y_val, trial=trial)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    print('Best trial:', study.best_trial.params)

    # Train best pipeline
    best_trial = study.best_trial
    pipe = MinimalPipeline(config)
    pipe.fit(X_train, y_train, X_val, y_val, trial=optuna.trial.FixedTrial(best_trial.params))
    # Predict on validation set
    val_pred = pipe.predict_proba(X_val)
    print('Validation ROC AUC:', roc_auc_score(y_val, val_pred))
    # Save and load demo
    pipe.save('modular_pipeline.joblib')
    loaded = MinimalPipeline.load('modular_pipeline.joblib')
    print('Loaded pipeline ROC AUC:', loaded.evaluate(X_val, y_val)) 