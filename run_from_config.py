import argparse
import yaml
import pandas as pd
import boto3
import os
import shutil
import numpy as np
from pipeline.base_pipeline import BasePipeline
from pipeline.minimal_modular_pipeline import MinimalPipeline
from classification_pipeline import ClassificationPipeline
import joblib

def download_from_s3(s3_path, local_path):
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3.download_file(bucket, key, local_path)

def load_csv(path):
    if path.startswith("s3://"):
        local_path = os.path.basename(path)
        download_from_s3(path, local_path)
        return pd.read_csv(local_path), local_path
    else:
        return pd.read_csv(path), path

class ClassicPipelineWrapper(BasePipeline):
    def __init__(self, config=None):
        self.config = config
        self.pipeline = None
        self.target_column = None
        self.output_path = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, config=None):
        if config is not None:
            self.config = config
        self.target_column = self.config['target_column']
        self.output_path = self.config.get('output_path', './results')
        self.pipeline = ClassificationPipeline(output_path=self.output_path)
        df_train = X_train.copy()
        df_train[self.target_column] = y_train
        df_val = None
        if X_val is not None and y_val is not None:
            df_val = X_val.copy()
            df_val[self.target_column] = y_val
        self.pipeline.train(
            train_data=df_train,
            target_column=self.target_column,
            eval_data=df_val,
            time_limit=self.config.get('time_limit', 3600),
            presets=self.config.get('presets', 'best_quality'),
            hyperparameters=self.config.get('autogluon_hyperparameters', 'multimodal')
        )

    def predict(self, X):
        return self.pipeline.predict(X)['prediction']

    def predict_proba(self, X):
        preds = self.pipeline.predict(X)
        # If binary, return probability for class 1
        if 1 in preds.columns:
            return preds[1]
        else:
            return preds.iloc[:, 0]

    def evaluate(self, X, y):
        from sklearn.metrics import roc_auc_score
        proba = self.predict_proba(X)
        return {'roc_auc': roc_auc_score(y, proba)}

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

def get_pipeline_class(pipeline_type):
    if pipeline_type == 'classic':
        return ClassicPipelineWrapper
    elif pipeline_type == 'modular':
        return MinimalPipeline
    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")

def main(config_path):
    # Download config if on S3
    if config_path.startswith("s3://"):
        local_config = "config.yaml"
        download_from_s3(config_path, local_config)
        config_path = local_config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pipeline_type = config.get('pipeline_type', 'classic')
    PipelineClass = get_pipeline_class(pipeline_type)

    # Load training data
    train_df, train_path = load_csv(config["train_data"])  # required
    target = config["target_column"]  # required
    predictors = config.get("predictors", [col for col in train_df.columns if col != target])
    X = train_df[predictors]
    y = train_df[target]

    # Load optional validation and test data
    X_val = y_val = None
    if "validation_data" in config:
        val_df, _ = load_csv(config["validation_data"])
        X_val = val_df[predictors]
        y_val = val_df[target]
    X_test = y_test = None
    if "test_data" in config:
        test_df, _ = load_csv(config["test_data"])
        X_test = test_df[predictors]
        y_test = test_df[target]

    # Fit pipeline
    pipeline = PipelineClass(config)
    pipeline.fit(X, y, X_val, y_val, config=config)
    print("Training complete.")

    # Evaluate if test data available
    if X_test is not None and y_test is not None:
        evaluation = pipeline.evaluate(X_test, y_test)
        print(f"Test set evaluation: {evaluation}")
        preds = pipeline.predict(X_test)
        pd.DataFrame({'prediction': preds}).to_csv('test_predictions.csv', index=False)

    # Save pipeline
    pipeline.save('trained_pipeline.joblib')
    print("Pipeline saved as trained_pipeline.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file (local or S3)")
    args = parser.parse_args()
    main(args.config) 