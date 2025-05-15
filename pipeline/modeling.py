"""
Model training, prediction, and cross-validation utilities for the classification pipeline.
"""
from typing import Any, Dict, Optional
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold, StratifiedKFold

class ModelTrainer:
    """Handles model training and cross-validation."""
    def __init__(self, output_path: str, problem_type: str = 'binary', eval_metric: str = 'roc_auc'):
        self.output_path = output_path
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.predictor = None

    def train(self, train_data: pd.DataFrame, target_column: str, eval_data: Optional[pd.DataFrame] = None, **kwargs) -> TabularPredictor:
        self.predictor = TabularPredictor(
            label=target_column,
            path=self.output_path,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric
        )
        self.predictor.fit(
            train_data=train_data,
            tuning_data=eval_data,
            **kwargs
        )
        return self.predictor

    def cross_validate(self, data: pd.DataFrame, target_column: str, n_splits: int = 5, stratified: bool = True, **kwargs) -> Dict:
        results = []
        X = data.drop(columns=[target_column])
        y = data[target_column]
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if stratified else KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X, y):
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            predictor = self.train(train_data, target_column, eval_data=val_data, **kwargs)
            score = predictor.evaluate(val_data)
            results.append(score)
        return {'cv_results': results}

# Add more modeling utilities as needed 