"""
Feature engineering utilities for the classification pipeline.
"""
import time
import logging
import pandas as pd
from typing import Dict, List, Any
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

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
        self.types_set = False
        self.use_onehot = use_onehot
        self.use_tfidf = use_tfidf
        self.use_poly = use_poly
        self.onehot_encoder = None
        self.tfidf_vectorizers = {}
        self.poly_transformer = None
        logger.info(f"FeatureEngineering: onehot={use_onehot}, tfidf={use_tfidf}, poly={use_poly}")

    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
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
                    if df[col].nunique() < self.text_threshold:
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
        self.categorical_features = feature_types.get('categorical_features', [])
        self.numeric_features = feature_types.get('numeric_features', [])
        self.text_features = feature_types.get('text_features', [])
        self.date_features = feature_types.get('date_features', [])
        self.types_set = True

    def get_feature_types(self) -> Dict[str, list]:
        return {
            'categorical_features': self.categorical_features,
            'numeric_features': self.numeric_features,
            'text_features': self.text_features,
            'date_features': self.date_features
        }

    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        for col in self.date_features:
            df_processed[f"{col}_year"] = df[col].dt.year
            df_processed[f"{col}_month"] = df[col].dt.month
            df_processed[f"{col}_day"] = df[col].dt.day
            df_processed[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df_processed[f"{col}_quarter"] = df[col].dt.quarter
            df_processed.drop(columns=[col], inplace=True)
        return df_processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"[FeatureEngineering] Starting fit_transform. Input shape: {df.shape}")
        start_time = time.time()
        df_out = df.copy()
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
        logger.info(f"[FeatureEngineering] Starting transform. Input shape: {df.shape}")
        start_time = time.time()
        df_out = df.copy()
        if self.use_onehot and self.categorical_features and self.onehot_encoder is not None:
            try:
                onehot = self.onehot_encoder.transform(df_out[self.categorical_features])
                onehot_df = pd.DataFrame(onehot, columns=self.onehot_encoder.get_feature_names_out(self.categorical_features), index=df_out.index)
                df_out = df_out.drop(columns=self.categorical_features)
                df_out = pd.concat([df_out, onehot_df], axis=1)
            except Exception as e:
                logger.error(f"Error in one-hot encoding (transform): {e}")
                raise
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