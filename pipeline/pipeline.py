"""
Main pipeline orchestration for the classification pipeline.
"""
import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from autogluon.tabular import TabularPredictor
from pipeline.missing_data import MissingDataHandler
from pipeline.feature_engineering import FeatureEngineering
from pipeline.interpretation import ModelFeatureImportance
from pipeline.exceptions import DataValidationError, ModelTrainingError

logger = logging.getLogger(__name__)

class ClassificationPipeline:
    """End-to-end classification pipeline with missing value handling and model interpretation."""
    def __init__(self, output_path: str = '/opt/ml/model', missing_threshold: float = 0.75, categorical_threshold: int = 15, text_threshold: int = 100,
                 use_onehot: bool = False, use_tfidf: bool = False, use_poly: bool = False, shap_class: int = 1):
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
        self.is_trained = False
        self.feature_types = None
        self.shap_class = shap_class
        logger.info(f"Pipeline thresholds: missing={missing_threshold}, categorical={categorical_threshold}, text={text_threshold}, onehot={use_onehot}, tfidf={use_tfidf}, poly={use_poly}, shap_class={shap_class}")

    def preprocess(self, df: pd.DataFrame, target_column: Optional[str] = None, is_training: bool = False) -> pd.DataFrame:
        logger.info(f"[Pipeline] Preprocessing started. is_training={is_training}, input shape={df.shape}")
        start_time = time.time()
        missing_analysis = self.missing_handler.analyze_missing(df)
        logger.info(f"Missing data analysis: {len(missing_analysis['cols_to_drop'])} columns with excessive missingness")
        df_processed = self.missing_handler.handle_missing(df)
        if is_training or not self.feature_engineering.types_set:
            feature_types = self.feature_engineering.identify_feature_types(df_processed)
            self.feature_types = feature_types
        else:
            self.feature_engineering.set_feature_types(self.feature_types)
        logger.info(f"Feature types: {self.feature_engineering.get_feature_types()}")
        if self.feature_engineering.date_features:
            df_processed = self.feature_engineering.create_date_features(df_processed)
        if is_training:
            df_processed = self.feature_engineering.fit_transform(df_processed)
        else:
            df_processed = self.feature_engineering.transform(df_processed)
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
        if self.missing_handler is None or self.feature_engineering is None or self.feature_types is None:
            self._load_preprocessing_objects()
        X_test_processed = self.preprocess(X_test, is_training=False)
        test_processed = X_test_processed.copy()
        test_processed[target_column] = y_test
        performance = self.predictor.evaluate(test_processed)
        logger.info(f"Model evaluation: {performance}")
        self.feature_importance = ModelFeatureImportance(
            model_path=os.path.join(self.output_path, 'model'),
            missing_handler=self.missing_handler,
            shap_class=self.shap_class
        )
        # SHAP-based feature importance
        self.feature_importance.calculate_shap_values(X_test_processed)
        shap_importance = self.feature_importance.get_shap_importance(combine_missing=True)
        shap_importance.to_csv(os.path.join(self.output_path, 'feature_importance.csv'), index=False)
        return {
            'performance': performance,
            'feature_importance': shap_importance
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.predictor is None:
            try:
                self.predictor = TabularPredictor.load(os.path.join(self.output_path, 'model'))
                logger.info("Loaded predictor from disk.")
            except Exception as e:
                logger.error(f"Could not load model: {e}")
                raise ValueError(f"Could not load model: {str(e)}")
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
        if self.missing_handler is None or self.feature_engineering is None or self.feature_types is None:
            self._load_preprocessing_objects()
        df_processed = self.preprocess(df, is_training=False)
        model_name = self.predictor.get_model_best()
        model = self.predictor._trainer.load_model(model_name)
        import shap
        explainer = shap.Explainer(model.predict, df_processed)
        shap_values = explainer(df_processed)
        predictions = self.predictor.predict(df_processed)
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
            'base_value': shap_values[0].base_values,
            'explainer': explainer
        } 