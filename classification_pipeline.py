# AWS SageMaker Classification Pipeline with AutoGluon
# Complete pipeline for training, evaluation, and inference with:
# - Probabilistic predictions
# - Missing data handling
# - Feature importance (including missingness importance)
# - Partial dependence analysis

import os
import sys
import json
import time
import pickle
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer

# AutoGluon and AWS-specific imports
import autogluon
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.common.features.types import R_FLOAT, R_INT, R_CATEGORY, R_OBJECT, S_TEXT
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl

# PDPbox for partial dependence plots
import pdpbox

# For model specific feature importance
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MissingDataHandler:
    """Handles missing data analysis and preprocessing."""
    
    def __init__(self, threshold: float = 0.75):
        """
        Args:
            threshold: Maximum percentage of missing values allowed in a column
        """
        self.threshold = threshold
        self.missing_stats = None
        self.cols_to_drop = None
        self.missing_indicators = {}
    
    def analyze_missing(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data patterns in the dataframe."""
        missing_count = df.isnull().sum()
        missing_percentage = missing_count / len(df) * 100
        
        missing_stats = pd.DataFrame({
            'Column': missing_count.index,
            'Missing Count': missing_count.values,
            'Missing Percentage': missing_percentage.values
        }).sort_values('Missing Percentage', ascending=False)
        
        self.missing_stats = missing_stats
        
        # Identify columns with excessive missing values
        self.cols_to_drop = missing_stats[missing_stats['Missing Percentage'] > (self.threshold * 100)]['Column'].tolist()
        
        return {
            'missing_stats': missing_stats,
            'cols_to_drop': self.cols_to_drop
        }
    
    def create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create missing value indicator columns for model training."""
        df_processed = df.copy()
        
        for col in df.columns:
            if df[col].isnull().any():
                indicator_name = f"{col}_missing"
                df_processed[indicator_name] = df[col].isnull().astype(int)
                self.missing_indicators[col] = indicator_name
                
        return df_processed
    
    def handle_missing(self, df: pd.DataFrame, drop_high_missing: bool = True) -> pd.DataFrame:
        """Process dataframe to handle missing values."""
        # Drop columns with excessive missingness if specified
        if drop_high_missing and self.cols_to_drop:
            df = df.drop(columns=self.cols_to_drop)
            logger.info(f"Dropped {len(self.cols_to_drop)} columns with excessive missing values: {self.cols_to_drop}")
        
        # Create missing value indicators
        df = self.create_missing_indicators(df)
        
        return df


class FeatureEngineering:
    """Handles feature engineering processes."""
    
    def __init__(self, categorical_threshold: int = 15, text_threshold: int = 100,
                 use_onehot: bool = False, use_tfidf: bool = False, use_poly: bool = False):
        self.categorical_features = []
        self.numeric_features = []
        self.text_features = []
        self.date_features = []
        self.categorical_threshold = categorical_threshold
        self.text_threshold = text_threshold
        self.types_set = False  # Track if types have been set from training
        self.use_onehot = use_onehot
        self.use_tfidf = use_tfidf
        self.use_poly = use_poly
        self.onehot_encoder = None
        self.tfidf_vectorizers = {}
        self.poly_transformer = None
        logger.info(f"FeatureEngineering: onehot={use_onehot}, tfidf={use_tfidf}, poly={use_poly}")
    
    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify feature types in the dataframe. Only set if not already set."""
        if not self.types_set:
            self.categorical_features = []
            self.numeric_features = []
            self.text_features = []
            self.date_features = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].nunique() < self.categorical_threshold and df[col].nunique() / len(df) < 0.05:
                        self.categorical_features.append(col)
                    else:
                        self.numeric_features.append(col)
                elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    if df[col].nunique() < self.text_threshold:  # threshold for categorical
                        self.categorical_features.append(col)
                    else:
                        self.text_features.append(col)
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    self.date_features.append(col)
            self.types_set = True
        feature_types = {
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features,
            'text_features': self.text_features,
            'date_features': self.date_features
        }
        return feature_types
    
    def set_feature_types(self, feature_types: Dict[str, list]):
        """Set feature types from a dictionary (for inference)."""
        self.categorical_features = feature_types.get('categorical_features', [])
        self.numeric_features = feature_types.get('numeric_features', [])
        self.text_features = feature_types.get('text_features', [])
        self.date_features = feature_types.get('date_features', [])
        self.types_set = True
    
    def get_feature_types(self) -> Dict[str, list]:
        """Get current feature types as a dictionary."""
        return {
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features,
            'text_features': self.text_features,
            'date_features': self.date_features
        }
    
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from date columns."""
        df_processed = df.copy()
        
        for col in self.date_features:
            df_processed[f"{col}_year"] = df[col].dt.year
            df_processed[f"{col}_month"] = df[col].dt.month
            df_processed[f"{col}_day"] = df[col].dt.day
            df_processed[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df_processed[f"{col}_quarter"] = df[col].dt.quarter
            
            # Drop original date column as AutoGluon doesn't handle it well
            df_processed.drop(columns=[col], inplace=True)
        
        return df_processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all enabled feature engineering steps (fit on training data)."""
        logger.info(f"[FeatureEngineering] Starting fit_transform. Input shape: {df.shape}")
        start_time = time.time()
        df_out = df.copy()
        # One-hot encoding
        if self.use_onehot and self.categorical_features:
            try:
                self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                onehot = self.onehot_encoder.fit_transform(df_out[self.categorical_features])
                onehot_df = pd.DataFrame(onehot, columns=self.onehot_encoder.get_feature_names_out(self.categorical_features), index=df_out.index)
                df_out = df_out.drop(columns=self.categorical_features)
                df_out = pd.concat([df_out, onehot_df], axis=1)
                logger.info(f"Applied one-hot encoding to: {self.categorical_features}")
            except Exception as e:
                logger.error(f"Error in one-hot encoding: {e}")
                raise
        # TF-IDF for text features
        if self.use_tfidf and self.text_features:
            for col in self.text_features:
                try:
                    vect = TfidfVectorizer(max_features=20)
                    tfidf = vect.fit_transform(df_out[col].fillna("").astype(str))
                    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(tfidf.shape[1])], index=df_out.index)
                    df_out = df_out.drop(columns=[col])
                    df_out = pd.concat([df_out, tfidf_df], axis=1)
                    self.tfidf_vectorizers[col] = vect
                    logger.info(f"Applied TF-IDF to: {col}")
                except Exception as e:
                    logger.error(f"Error in TF-IDF for {col}: {e}")
                    raise
        # Polynomial features
        if self.use_poly and self.numeric_features:
            try:
                self.poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
                poly = self.poly_transformer.fit_transform(df_out[self.numeric_features])
                poly_cols = self.poly_transformer.get_feature_names_out(self.numeric_features)
                poly_df = pd.DataFrame(poly, columns=poly_cols, index=df_out.index)
                df_out = df_out.drop(columns=self.numeric_features)
                df_out = pd.concat([df_out, poly_df], axis=1)
                logger.info(f"Applied polynomial features to: {self.numeric_features}")
            except Exception as e:
                logger.error(f"Error in polynomial features: {e}")
                raise
        logger.info(f"[FeatureEngineering] fit_transform complete. Output shape: {df_out.shape}. Time: {time.time() - start_time:.2f}s")
        return df_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all enabled feature engineering steps (use fit params from training)."""
        logger.info(f"[FeatureEngineering] Starting transform. Input shape: {df.shape}")
        start_time = time.time()
        df_out = df.copy()
        # One-hot encoding
        if self.use_onehot and self.categorical_features and self.onehot_encoder is not None:
            try:
                onehot = self.onehot_encoder.transform(df_out[self.categorical_features])
                onehot_df = pd.DataFrame(onehot, columns=self.onehot_encoder.get_feature_names_out(self.categorical_features), index=df_out.index)
                df_out = df_out.drop(columns=self.categorical_features)
                df_out = pd.concat([df_out, onehot_df], axis=1)
            except Exception as e:
                logger.error(f"Error in one-hot encoding (transform): {e}")
                raise
        # TF-IDF for text features
        if self.use_tfidf and self.text_features:
            for col in self.text_features:
                vect = self.tfidf_vectorizers.get(col)
                if vect is not None:
                    try:
                        tfidf = vect.transform(df_out[col].fillna("").astype(str))
                        tfidf_df = pd.DataFrame(tfidf.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(tfidf.shape[1])], index=df_out.index)
                        df_out = df_out.drop(columns=[col])
                        df_out = pd.concat([df_out, tfidf_df], axis=1)
                    except Exception as e:
                        logger.error(f"Error in TF-IDF (transform) for {col}: {e}")
                        raise
        # Polynomial features
        if self.use_poly and self.numeric_features and self.poly_transformer is not None:
            try:
                poly = self.poly_transformer.transform(df_out[self.numeric_features])
                poly_cols = self.poly_transformer.get_feature_names_out(self.numeric_features)
                poly_df = pd.DataFrame(poly, columns=poly_cols, index=df_out.index)
                df_out = df_out.drop(columns=self.numeric_features)
                df_out = pd.concat([df_out, poly_df], axis=1)
            except Exception as e:
                logger.error(f"Error in polynomial features (transform): {e}")
                raise
        logger.info(f"[FeatureEngineering] transform complete. Output shape: {df_out.shape}. Time: {time.time() - start_time:.2f}s")
        return df_out


class ModelFeatureImportance:
    """Handles feature importance analysis including missing value importance."""
    
    def __init__(self, model_path: str, missing_handler: MissingDataHandler):
        """
        Args:
            model_path: Path to the trained AutoGluon model
            missing_handler: The MissingDataHandler instance with missing indicator info
        """
        self.model_path = model_path
        self.missing_handler = missing_handler
        self.predictor = TabularPredictor.load(model_path)
        self.feature_importance = None
        self.combined_importance = None
    
    def calculate_importance(self, df: pd.DataFrame, method: str = 'permutation') -> pd.DataFrame:
        """Calculate feature importance."""
        if method == 'permutation':
            importance = self.predictor.feature_importance(df, subsample_size=10000)
        else:
            importance = self.predictor.feature_importance(df, method='auto')
            
        self.feature_importance = importance
        return importance
    
    def combine_missing_importance(self) -> pd.DataFrame:
        """Combine original feature importance with its missing indicator importance."""
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Call calculate_importance first.")
        
        importance_df = self.feature_importance.copy()
        combined = {}
        
        # Initialize with known features
        for feature in importance_df.index:
            if not feature.endswith('_missing'):
                combined[feature] = importance_df.loc[feature, 'importance']
        
        # Add missing indicator importance to original feature
        for orig_feature, missing_feature in self.missing_handler.missing_indicators.items():
            if missing_feature in importance_df.index:
                if orig_feature in combined:
                    combined[orig_feature] += importance_df.loc[missing_feature, 'importance']
                else:
                    # Handle case where original feature might have been dropped
                    combined[orig_feature] = importance_df.loc[missing_feature, 'importance']
        
        # Create combined importance dataframe
        combined_df = pd.DataFrame({
            'feature': list(combined.keys()),
            'combined_importance': list(combined.values())
        }).sort_values('combined_importance', ascending=False)
        
        self.combined_importance = combined_df
        return combined_df
    
    def plot_feature_importance(self, top_n: int = 20) -> plt.Figure:
        """Plot top N features by combined importance."""
        if self.combined_importance is None:
            self.combine_missing_importance()
            
        plt.figure(figsize=(12, 8))
        data = self.combined_importance.head(top_n)
        sns.barplot(x='combined_importance', y='feature', data=data)
        plt.title(f'Top {top_n} Features by Combined Importance')
        plt.tight_layout()
        
        return plt.gcf()


class PartialDependenceAnalyzer:
    """Handles calculation and visualization of partial dependence plots."""
    
    def __init__(self, model_path: str, feature_engineering: FeatureEngineering):
        """
        Args:
            model_path: Path to the trained AutoGluon model
            feature_engineering: FeatureEngineering instance with feature type info
        """
        self.model_path = model_path
        self.predictor = TabularPredictor.load(model_path)
        self.feature_engineering = feature_engineering
        
    def generate_pdp(self, df: pd.DataFrame, features: List[str], target_class: Optional[int] = None) -> Dict[str, Any]:
        """Generate partial dependence data for specified features."""
        model = self.predictor.get_model_best()
        pdp_results = {}
        
        for feature in features:
            try:
                if feature in self.feature_engineering.numeric_features:
                    # Numeric feature PDP
                    pdp_isolate = pdpbox.pdp.PDPIsolate(
                        model=model,
                        dataset=df,
                        model_features=df.columns.tolist(),
                        feature=feature,
                        num_grid_points=20
                    )
                    
                    fig, axes = pdp_isolate.plot(center=True, plot_pts_dist=True)
                    pdp_results[feature] = {
                        'type': 'numeric',
                        'pdp_values': pdp_isolate.pdp,
                        'figure': fig
                    }
                    
                elif feature in self.feature_engineering.categorical_features:
                    # Categorical feature PDP
                    pdp_isolate = pdpbox.pdp.PDPIsolate(
                        model=model,
                        dataset=df,
                        model_features=df.columns.tolist(),
                        feature=feature
                    )
                    
                    fig, axes = pdp_isolate.plot(center=True, plot_pts_dist=True)
                    pdp_results[feature] = {
                        'type': 'categorical',
                        'pdp_values': pdp_isolate.pdp,
                        'figure': fig
                    }
            except Exception as e:
                logger.warning(f"Could not generate PDP for feature {feature}: {str(e)}")
                
        return pdp_results


class ClassificationPipeline:
    """End-to-end classification pipeline with missing value handling and model interpretation."""
    
    def __init__(self, output_path: str = '/opt/ml/model', missing_threshold: float = 0.75, categorical_threshold: int = 15, text_threshold: int = 100,
                 use_onehot: bool = False, use_tfidf: bool = False, use_poly: bool = False):
        self.output_path = output_path
        self.missing_handler = MissingDataHandler(threshold=missing_threshold)
        self.feature_engineering = FeatureEngineering(
            categorical_threshold=categorical_threshold,
            text_threshold=text_threshold,
            use_onehot=use_onehot,
            use_tfidf=use_tfidf,
            use_poly=use_poly
        )
        self.predictor = None
        self.feature_importance = None
        self.pdp_analyzer = None
        self.is_trained = False
        self.feature_types = None
        logger.info(f"Pipeline thresholds: missing={missing_threshold}, categorical={categorical_threshold}, text={text_threshold}, onehot={use_onehot}, tfidf={use_tfidf}, poly={use_poly}")
        
    def preprocess(self, df: pd.DataFrame, target_column: Optional[str] = None, is_training: bool = False) -> pd.DataFrame:
        """Preprocess data with missing value handling and feature engineering. Use training types if not training."""
        logger.info(f"[Pipeline] Preprocessing started. is_training={is_training}, input shape={df.shape}")
        start_time = time.time()
        # Analyze missing patterns (always recompute for current df)
        missing_analysis = self.missing_handler.analyze_missing(df)
        logger.info(f"Missing data analysis: {len(missing_analysis['cols_to_drop'])} columns with excessive missingness")
        # Handle missing values
        df_processed = self.missing_handler.handle_missing(df)
        # Feature type handling
        if is_training or not self.feature_engineering.types_set:
            feature_types = self.feature_engineering.identify_feature_types(df_processed)
            self.feature_types = feature_types
        else:
            self.feature_engineering.set_feature_types(self.feature_types)
        logger.info(f"Feature types: {self.feature_engineering.get_feature_types()}")
        # Create date features if any
        if self.feature_engineering.date_features:
            df_processed = self.feature_engineering.create_date_features(df_processed)
        # Feature engineering steps
        if is_training:
            df_processed = self.feature_engineering.fit_transform(df_processed)
        else:
            df_processed = self.feature_engineering.transform(df_processed)
        # Save preprocessing components only during training
        if is_training:
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'missing_handler.pkl'), 'wb') as f:
                pickle.dump(self.missing_handler, f)
            with open(os.path.join(self.output_path, 'feature_engineering.pkl'), 'wb') as f:
                pickle.dump(self.feature_engineering, f)
            with open(os.path.join(self.output_path, 'feature_types.pkl'), 'wb') as f:
                pickle.dump(self.feature_types, f)
        logger.info(f"[Pipeline] Preprocessing complete. Output shape: {df_processed.shape}. Time: {time.time() - start_time:.2f}s")
        return df_processed
    
    def train(self, 
              train_data: pd.DataFrame, 
              target_column: str,
              eval_data: Optional[pd.DataFrame] = None,
              time_limit: int = 3600,
              presets: str = 'best_quality',
              hyperparameters: Optional[Any] = 'multimodal') -> TabularPredictor:
        """Train the classification model. Logs data shapes and training time."""
        logger.info(f"[Pipeline] Training started. Train data shape: {train_data.shape}")
        start_time = time.time()
        y_train = train_data[target_column]
        X_train = train_data.drop(columns=[target_column])
        X_train_processed = self.preprocess(X_train, is_training=True)
        train_processed = X_train_processed.copy()
        train_processed[target_column] = y_train
        predictor = TabularPredictor(
            label=target_column,
            path=os.path.join(self.output_path, 'model'),
            problem_type='binary' if len(y_train.unique()) <= 2 else 'multiclass',
            eval_metric='roc_auc' if len(y_train.unique()) <= 2 else 'accuracy'
        )
        predictor.fit(
            train_data=train_processed,
            tuning_data=eval_data if eval_data is not None else None,
            time_limit=time_limit,
            presets=presets,
            hyperparameters=hyperparameters,
            num_cpus=os.cpu_count(),
            verbosity=2,
            use_bag_holdout=True if eval_data is not None else False
        )
        self.predictor = predictor
        self.is_trained = True
        logger.info(f"Model training completed. Leaderboard: {predictor.leaderboard()}")
        logger.info(f"[Pipeline] Training complete. Time: {time.time() - start_time:.2f}s")
        return predictor
    
    def _load_preprocessing_objects(self):
        """Load preprocessing objects if not already loaded."""
        if self.missing_handler is None:
            try:
                with open(os.path.join(self.output_path, 'missing_handler.pkl'), 'rb') as f:
                    self.missing_handler = pickle.load(f)
                logger.info("Loaded missing_handler from disk.")
            except Exception as e:
                logger.error(f"Could not load missing_handler: {e}")
                raise
        if self.feature_engineering is None:
            try:
                with open(os.path.join(self.output_path, 'feature_engineering.pkl'), 'rb') as f:
                    self.feature_engineering = pickle.load(f)
                logger.info("Loaded feature_engineering from disk.")
            except Exception as e:
                logger.error(f"Could not load feature_engineering: {e}")
                raise
        if self.feature_types is None:
            try:
                with open(os.path.join(self.output_path, 'feature_types.pkl'), 'rb') as f:
                    self.feature_types = pickle.load(f)
                logger.info("Loaded feature_types from disk.")
            except Exception as e:
                logger.error(f"Could not load feature_types: {e}")
                raise

    def evaluate(self, test_data: pd.DataFrame, target_column: str) -> Dict:
        if self.predictor is None:
            raise ValueError("Model not trained. Call train() method first.")
        y_test = test_data[target_column]
        X_test = test_data.drop(columns=[target_column])
        # Load preprocessing objects if not set
        if self.missing_handler is None or self.feature_engineering is None or self.feature_types is None:
            self._load_preprocessing_objects()
        X_test_processed = self.preprocess(X_test, is_training=False)
        test_processed = X_test_processed.copy()
        test_processed[target_column] = y_test
        performance = self.predictor.evaluate(test_processed)
        logger.info(f"Model evaluation: {performance}")
        # Calculate probabilities
        self.feature_importance = ModelFeatureImportance(
            model_path=os.path.join(self.output_path, 'model'),
            missing_handler=self.missing_handler
        )
        # Pass test_processed (with label) to feature_importance
        importance = self.feature_importance.calculate_importance(test_processed)
        combined_importance = self.feature_importance.combine_missing_importance()
        # Generate partial dependence plots for top features
        self.pdp_analyzer = PartialDependenceAnalyzer(
            model_path=os.path.join(self.output_path, 'model'),
            feature_engineering=self.feature_engineering
        )
        top_features = combined_importance['feature'].head(10).tolist()
        pdp_results = self.pdp_analyzer.generate_pdp(X_test_processed, top_features)
        # Save importance and pdp results
        combined_importance.to_csv(os.path.join(self.output_path, 'feature_importance.csv'), index=False)
        # Save PDPs as plots
        pdp_dir = os.path.join(self.output_path, 'pdp_plots')
        os.makedirs(pdp_dir, exist_ok=True)
        for feature, pdp_data in pdp_results.items():
            if 'figure' in pdp_data:
                pdp_data['figure'].savefig(os.path.join(pdp_dir, f"pdp_{feature}.png"))
        return {
            'performance': performance,
            'feature_importance': combined_importance,
            'top_features': top_features
        }
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.predictor is None:
            try:
                self.predictor = TabularPredictor.load(os.path.join(self.output_path, 'model'))
                logger.info("Loaded predictor from disk.")
            except Exception as e:
                logger.error(f"Could not load model: {e}")
                raise ValueError(f"Could not load model: {str(e)}")
        # Load preprocessing objects if not set
        if self.missing_handler is None or self.feature_engineering is None or self.feature_types is None:
            self._load_preprocessing_objects()
        df_processed = self.preprocess(df, is_training=False)
        predictions = self.predictor.predict(df_processed)
        probabilities = self.predictor.predict_proba(df_processed)
        results = pd.DataFrame(probabilities)
        results['prediction'] = predictions
        return results
    
    def explain_prediction(self, df: pd.DataFrame) -> Dict:
        if self.predictor is None:
            try:
                self.predictor = TabularPredictor.load(os.path.join(self.output_path, 'model'))
                logger.info("Loaded predictor from disk.")
            except Exception as e:
                logger.error(f"Could not load model: {e}")
                raise ValueError(f"Could not load model: {str(e)}")
        # Load preprocessing objects if not set
        if self.missing_handler is None or self.feature_engineering is None or self.feature_types is None:
            self._load_preprocessing_objects()
        df_processed = self.preprocess(df, is_training=False)
        # Get best model for explanation
        model_name = self.predictor.get_model_best()
        model = self.predictor._trainer.load_model(model_name)
        # Initialize explainer
        explainer = shap.Explainer(model.predict, df_processed)
        shap_values = explainer(df_processed)
        # Get predictions
        predictions = self.predictor.predict(df_processed)
        # Combine into explanation
        explanations = []
        for i in range(len(df)):
            explanation = {
                'prediction': predictions.iloc[i],
                'shap_values': shap_values[i].values.tolist(),
                'features': df_processed.columns.tolist()
            }
            explanations.append(explanation)
        return {
            'explanations': explanations,
            'base_value': shap_values[0].base_values
        }


class SagemakerEntrypoint:
    """Entry points for SageMaker training and inference."""
    
    @staticmethod
    def train():
        """SageMaker training entry point."""
        parser = argparse.ArgumentParser()
        
        # Hyperparameters
        parser.add_argument('--time-limit', type=int, default=3600)
        parser.add_argument('--presets', type=str, default='best_quality')
        
        # SageMaker parameters
        parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
        parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
        parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test'))
        parser.add_argument('--target-column', type=str, required=True)
        parser.add_argument('--missing-threshold', type=float, default=None, help='Max fraction of missing values allowed in a column (0-1)')
        parser.add_argument('--categorical-threshold', type=int, default=None, help='Max unique values for categorical features')
        parser.add_argument('--text-threshold', type=int, default=None, help='Min unique values for text features')
        parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
        parser.add_argument('--use-onehot', action='store_true', help='Enable one-hot encoding for categorical features')
        parser.add_argument('--use-tfidf', action='store_true', help='Enable TF-IDF vectorization for text features')
        parser.add_argument('--use-poly', action='store_true', help='Enable polynomial features for numeric features')
        
        args, _ = parser.parse_known_args()
        
        # Load config file if provided
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {args.config}: {config}")
        # Determine thresholds (CLI overrides config, which overrides default)
        missing_threshold = args.missing_threshold if args.missing_threshold is not None else config.get('missing_threshold', 0.75)
        categorical_threshold = args.categorical_threshold if args.categorical_threshold is not None else config.get('categorical_threshold', 15)
        text_threshold = args.text_threshold if args.text_threshold is not None else config.get('text_threshold', 100)
        use_onehot = args.use_onehot if 'use_onehot' in args else config.get('use_onehot', False)
        use_tfidf = args.use_tfidf if 'use_tfidf' in args else config.get('use_tfidf', False)
        use_poly = args.use_poly if 'use_poly' in args else config.get('use_poly', False)
        logger.info(f"Final thresholds: missing={missing_threshold}, categorical={categorical_threshold}, text={text_threshold}")
        
        # Initialize pipeline
        pipeline = ClassificationPipeline(
            output_path=args.model_dir,
            missing_threshold=missing_threshold,
            categorical_threshold=categorical_threshold,
            text_threshold=text_threshold,
            use_onehot=use_onehot,
            use_tfidf=use_tfidf,
            use_poly=use_poly
        )
        
        # Load data
        train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
        
        if os.path.exists(os.path.join(args.validation, 'validation.csv')):
            validation_data = pd.read_csv(os.path.join(args.validation, 'validation.csv'))
        else:
            validation_data = None
            
        if os.path.exists(os.path.join(args.test, 'test.csv')):
            test_data = pd.read_csv(os.path.join(args.test, 'test.csv'))
        else:
            test_data = None
            
        # Train model
        pipeline.train(
            train_data=train_data,
            target_column=args.target_column,
            eval_data=validation_data,
            time_limit=args.time_limit,
            presets=args.presets
        )
        
        # Evaluate if test data available
        if test_data is not None:
            evaluation = pipeline.evaluate(test_data, args.target_column)
            with open(os.path.join(args.model_dir, 'evaluation.json'), 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                clean_eval = {}
                for k, v in evaluation['performance'].items():
                    if isinstance(v, np.ndarray):
                        clean_eval[k] = v.tolist()
                    elif isinstance(v, np.number):
                        clean_eval[k] = v.item()
                    else:
                        clean_eval[k] = v
                json.dump(clean_eval, f)
        
        logger.info("Training complete!")
    
    @staticmethod
    def model_fn(model_dir):
        """SageMaker model loading function."""
        predictor = TabularPredictor.load(os.path.join(model_dir, 'model'))
        with open(os.path.join(model_dir, 'missing_handler.pkl'), 'rb') as f:
            missing_handler = pickle.load(f)
        with open(os.path.join(model_dir, 'feature_engineering.pkl'), 'rb') as f:
            feature_engineering = pickle.load(f)
            
        pipeline = ClassificationPipeline(output_path=model_dir)
        pipeline.predictor = predictor
        pipeline.missing_handler = missing_handler
        pipeline.feature_engineering = feature_engineering
        
        return pipeline
    
    @staticmethod
    def transform_fn(pipeline, request_body, content_type, accept_type):
        """SageMaker inference function."""
        if content_type == 'application/json':
            data = json.loads(request_body)
            df = pd.DataFrame(data['instances'])
        elif content_type == 'text/csv':
            df = pd.read_csv(io.StringIO(request_body))
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Make predictions
        predictions = pipeline.predict(df)
        
        # Return predictions in requested format
        if accept_type == 'application/json':
            response_body = json.dumps({
                'predictions': predictions.to_dict(orient='records')
            })
            content_type = 'application/json'
        elif accept_type == 'text/csv':
            response_body = predictions.to_csv(index=False)
            content_type = 'text/csv'
        else:
            response_body = json.dumps({
                'predictions': predictions.to_dict(orient='records')
            })
            content_type = 'application/json'
            
        return response_body, content_type


# For local testing
if __name__ == '__main__':
    # Example usage
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # Load sample data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Add some missing values to simulate real-world data
    np.random.seed(42)
    for col in X.columns[:5]:  # Add missing values to first 5 columns
        mask = np.random.random(len(X)) < 0.1  # 10% missing
        X.loc[mask, col] = np.nan
    
    # Create a combined dataframe
    df = X.copy()
    df['target'] = y
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize and run pipeline
    pipeline = ClassificationPipeline(output_path='./model_output')
    
    # Train model
    pipeline.train(
        train_data=train_df,
        target_column='target',
        time_limit=300,  # 5 minutes for testing
        presets='good_quality'  # Faster preset for testing
    )
    
    # Evaluate model
    evaluation = pipeline.evaluate(test_df, 'target')
    print(f"Model performance: {evaluation['performance']}")
    
    # Get predictions for a sample
    sample = test_df.drop(columns=['target']).iloc[:5]
    predictions = pipeline.predict(sample)
    print(f"Predictions: {predictions}")
    
    # Explain predictions
    explanations = pipeline.explain_prediction(sample)
    print(f"Explanation for first sample: {explanations['explanations'][0]}") 