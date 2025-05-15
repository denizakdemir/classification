"""
Model interpretation utilities for the classification pipeline (SHAP-based).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from autogluon.tabular import TabularPredictor
import shap
import logging

logger = logging.getLogger(__name__)

class ModelFeatureImportance:
    """Handles SHAP-based feature importance and effect analysis, including missing value indicators."""
    def __init__(self, model_path: str, missing_handler, shap_class: int = 1):
        self.model_path = model_path
        self.missing_handler = missing_handler
        self.shap_class = shap_class
        self.predictor = TabularPredictor.load(model_path)
        self.feature_importance = None
        self.shap_values = None
        self.shap_explainer = None
        self.combined_importance = None

    def calculate_shap_values(self, df: pd.DataFrame):
        """Calculate SHAP values for the best model."""
        model_name = self.predictor.get_model_best()
        model = self.predictor._trainer.load_model(model_name)
        # Use model.predict_proba for SHAP, explaining the probability of the specified class
        if hasattr(model, "predict_proba"):
            def proba_fn(X):
                proba = model.predict_proba(X)
                if proba.ndim == 1:
                    return proba
                else:
                    return proba[:, self.shap_class]
            self.shap_explainer = shap.Explainer(proba_fn, df)
        else:
            # fallback: use predict (not recommended)
            self.shap_explainer = shap.Explainer(model.predict, df)
        self.shap_values = self.shap_explainer(df)
        return self.shap_values

    def plot_shap_summary(self, df: pd.DataFrame, max_display: int = 20):
        """Plot SHAP summary (global feature effect) for the given data."""
        if self.shap_values is None:
            self.calculate_shap_values(df)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, df, max_display=max_display, show=False)
        plt.tight_layout()
        return plt.gcf()

    def plot_shap_dependence(self, df: pd.DataFrame, feature: str):
        """Plot SHAP dependence plot for a specific feature."""
        if self.shap_values is None:
            self.calculate_shap_values(df)
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature, self.shap_values.values, df, show=False)
        plt.tight_layout()
        return plt.gcf()

    def get_shap_importance(self, combine_missing: bool = True) -> pd.DataFrame:
        """Return SHAP-based feature importances, optionally combining missing indicators."""
        if self.shap_values is None:
            raise ValueError("Call calculate_shap_values first.")
        # Mean absolute SHAP value per feature
        shap_importance = pd.DataFrame({
            'feature': self.shap_values.feature_names,
            'importance': self.shap_values.abs.mean(0).values
        })
        shap_importance = shap_importance.sort_values('importance', ascending=False)
        if combine_missing:
            # Combine importance of variable and its missing indicator
            combined = {}
            for _, row in shap_importance.iterrows():
                feat = row['feature']
                imp = row['importance']
                if feat.endswith('_missing'):
                    orig = feat[:-9]
                    combined[orig] = combined.get(orig, 0) + imp
                else:
                    combined[feat] = combined.get(feat, 0) + imp
            combined_df = pd.DataFrame({
                'feature': list(combined.keys()),
                'importance': list(combined.values())
            }).sort_values('importance', ascending=False)
            return combined_df
        else:
            return shap_importance

    def plot_shap_importance(self, df: pd.DataFrame, combine_missing: bool = True, max_display: int = 20):
        """Plot SHAP-based feature importances (bar plot), optionally combining missing indicators."""
        if self.shap_values is None:
            self.calculate_shap_values(df)
        shap_importance = self.get_shap_importance(combine_missing=combine_missing)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=shap_importance.head(max_display))
        plt.title(f"Top {max_display} Features by SHAP Importance" + (" (Combined)" if combine_missing else " (Separate)"))
        plt.tight_layout()
        return plt.gcf() 