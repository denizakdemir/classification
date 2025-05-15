# ML Classification Pipeline Usage Guide

This guide explains how to use the comprehensive ML classification pipeline for AWS SageMaker with AutoGluon, and also introduces a new modular, hyperparameter-optimizable pipeline for advanced users. Both are designed to handle missing data, provide probabilistic estimates, feature importance analysis, and partial dependence plots.

## Features

- **Missing Data Handling**: Automatically identifies and addresses both random and structural missingness in your data
- **Probabilistic Estimates**: Provides probability scores for each class, not just final classifications
- **Feature Importance Analysis**: Combines importance of original variables with their missingness patterns
- **Partial Dependence Analysis**: Shows how each feature affects predictions while accounting for missingness
- **Complete Prediction Pipeline**: Handles raw data with uncoded values and missing entries
- **AWS SageMaker Integration**: Ready for deployment in AWS SageMaker environment with AutoGluon

## Getting Started

### Prerequisites

- AWS account with SageMaker access
- Appropriate IAM roles configured
- Python 3.7+ with pandas, numpy
- Basic understanding of AWS SageMaker concepts

### File Structure

This package contains the following key files:

- `classification_pipeline.py`: The core pipeline code with all components
- `sagemaker_deployment.py`: Script to deploy the pipeline on AWS SageMaker (automatically generates the required entry point script for SageMaker)

## Using the Pipeline Locally

For local testing and development, you can use the pipeline without SageMaker:

```python
from classification_pipeline import ClassificationPipeline

# Initialize pipeline
pipeline = ClassificationPipeline(output_path='./model_output')

# Load your data (with missing values as np.nan)
# Your dataframe should have predictors and target column
import pandas as pd
df = pd.read_csv('your_data.csv')

# Split your data into train/test
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train the model
pipeline.train(
    train_data=train_df,
    target_column='your_target_column',  # Name of your target variable
    time_limit=3600,  # Time limit in seconds
    presets='best_quality'  # AutoGluon preset
)

# Evaluate the model
evaluation = pipeline.evaluate(test_df, 'your_target_column')
print(f"Model performance: {evaluation['performance']}")
print(f"Top features: {evaluation['top_features']}")

# Make predictions on new data
new_data = pd.read_csv('new_data.csv')  # Can have missing values
predictions = pipeline.predict(new_data)
```

## Deploying to AWS SageMaker

To deploy the pipeline to SageMaker, use the provided `sagemaker_deployment.py` script:

```bash
# Upload your code and data to SageMaker
python sagemaker_deployment.py \
  --train-data s3://your-bucket/path/to/train.csv \
  --test-data s3://your-bucket/path/to/test.csv \
  --target-column your_target_column \
  --time-limit 3600 \
  --instance-type ml.m5.2xlarge
```

> **Note:** The deployment script will automatically generate the minimal entry point required by SageMaker; you do not need to create or edit a `pipeline_entry_point.py` file yourself.

Alternatively, follow these steps manually:

1. Package your pipeline code
2. Upload your training data to S3
3. Use the SageMaker Python SDK to create an estimator
4. Deploy the model to an endpoint

## Data Format

Your data should be in a pandas DataFrame format or CSV with:

- Each row representing an observation
- Each column representing a feature
- Missing values represented as `np.nan`
- One column designated as the target/label column

Example:

```
feature1,feature2,feature3,target
23.1,category_a,,1
,12.3,category_b,0
45.2,,category_a,1
```

## Handling Missing Data

The pipeline handles missing data in two ways:

1. **Missing Indicators**: Creates binary flags for missingness patterns
2. **Imputation**: Automatically handled by AutoGluon's internal processing

No pre-imputation is required; the pipeline is designed to work with raw data containing `np.nan` values.

## Interpreting Results

### Feature Importance

After training, you'll get feature importance that combines:

- The importance of the variable itself
- The importance of its missingness pattern

The combined importance is stored in `feature_importance.csv` in your model output directory.

### Partial Dependence Plots

PDP plots showing how each feature affects predictions are saved in the `pdp_plots` directory. These help visualize:

- How numeric variables influence model output across their range
- How categorical variables affect predictions for each category
- How missingness in variables impacts predictions

## Configuration Options

The pipeline supports the following configuration parameters (see `example_config.yaml`):

- `train_data`, `validation_data`, `test_data`: Paths to your CSV data (local or S3)
- `target_column`: Name of the target variable
- `predictors`: List of feature columns to use (optional; if omitted, all except target)
- `model_name`, `run_name`, `output_path`: Used to organize output artifacts
- `missingness_threshold`: Max allowed missing fraction per column (default 0.75)
- `categorical_threshold`: Max unique values for a column to be considered categorical (default 15)
- `text_threshold`: Min unique values for a column to be considered text (default 100)
- `use_onehot`: Enable one-hot encoding for categorical features (default false)
- `use_tfidf`: Enable TF-IDF vectorization for text features (default false)
- `use_poly`: Enable polynomial features for numeric features (default false)
- `random_seed`: Random seed for reproducibility (numpy, random, torch)
- `autogluon_hyperparameters`: Custom hyperparameters for AutoGluon TabularPredictor.fit (optional)
- `save_leaderboard`, `save_predictions`, `save_config_copy`: Output toggles
- `time_limit`, `presets`: AutoGluon training options

Example config snippet:

```yaml
pipeline_type: classic
train_data: datasets/heart_train.csv
validation_data: datasets/heart_val.csv
test_data: datasets/heart_test.csv
target_column: target
output_path: ./results
missingness_threshold: 0.75
categorical_threshold: 15
text_threshold: 100
use_onehot: false  # Set to true to enable one-hot encoding for categorical features
use_tfidf: false   # Set to true to enable TF-IDF vectorization for text features
use_poly: false    # Set to true to enable polynomial features for numeric features
random_seed: 42
autogluon_hyperparameters:
  RF: {n_estimators: 100}
save_leaderboard: true
save_predictions: true
save_config_copy: true
time_limit: 60
presets: good_quality
```

### How these affect the pipeline
- `missingness_threshold`: Columns with a higher missing fraction are dropped.
- `categorical_threshold`/`text_threshold`: Control how features are classified as categorical or text.
- `random_seed`: Ensures reproducibility for numpy, random, and torch.
- `autogluon_hyperparameters`: Passed directly to AutoGluon for model customization.

## Running Tests
To run all tests (unit, integration, edge cases):
```bash
PYTHONPATH=. pytest -v tests/
```
If you use NumPy >=1.24, the test suite includes a monkey-patch for `np.bool` and `np.int` to ensure compatibility with SHAP and other libraries.

## Troubleshooting
- **NumPy/SHAP errors:** If you see errors about `np.bool` or `np.int`, ensure your test file includes:
  ```python
  import numpy as np
  if not hasattr(np, 'bool'):
      np.bool = np.bool_
  if not hasattr(np, 'int'):
      np.int = int
  ```
- **Ray errors:** The test suite uses only non-parallel models (`RF`, `XT`, `KNN`) to avoid Ray dependency.

## Advanced Configuration

### Customizing Missing Value Handling

```python
# Adjust the missing value threshold (default is 0.75 or 75%)
pipeline.missing_handler.threshold = 0.5  # Only keep features with < 50% missing

# Force keeping columns even with high missingness
pipeline.missing_handler.cols_to_drop = ['feature_to_drop']  # Override automatic detection
```

### Adjusting Model Training

```python
# Change time limit
pipeline.train(train_data=train_df, target_column='target', time_limit=7200)  # 2 hours

# Use different AutoGluon preset
pipeline.train(train_data=train_df, target_column='target', presets='medium_quality_faster_train')

# Pass custom hyperparameters from config
pipeline.train(
    train_data=train_df,
    target_column='target',
    hyperparameters={
        'GBM': {'extra_trees': True, 'ag_args': {'name_suffix': 'EXTRA'}}
    }
)
```

### Adjusting Feature Detection

```python
# Set thresholds for feature type detection
pipeline.feature_engineering.categorical_threshold = 20
pipeline.feature_engineering.text_threshold = 200
```

### Setting Random Seed

```python
# Set random seed for reproducibility
import numpy as np
np.random.seed(42)
import random
random.seed(42)
# If using torch:
# import torch
# torch.manual_seed(42)
```

## Best Practices

1. **Avoid Pre-processing**: Let the pipeline handle missing values directly
2. **Check Missing Patterns**: Review the `missing_stats` output to understand your data
3. **Examine Feature Importance**: Pay attention to both original features and their missingness
4. **Review PDPs**: Use partial dependence plots to understand feature effects
5. **Enough Training Time**: For complex datasets, provide adequate training time (1+ hours)

## Advanced Usage Examples

### Multi-class Classification

```python
# Pipeline automatically detects multi-class problems
pipeline = ClassificationPipeline()
pipeline.train(
    train_data=multi_class_df,
    target_column='multi_class_target',
    time_limit=3600
)
```

### Explaining Individual Predictions

```python
# Get explanations for specific predictions
sample = test_df.drop(columns=['target']).iloc[:5]
explanations = pipeline.explain_prediction(sample)

# Access SHAP values for feature attribution
for idx, exp in enumerate(explanations['explanations']):
    print(f"Sample {idx} prediction: {exp['prediction']}")
    # Top contributing features
    feature_impact = list(zip(exp['features'], exp['shap_values']))
    sorted_impact = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)
    print(f"Top features: {sorted_impact[:5]}")
```

### Saving and Loading Models

The model is automatically saved during training. To load a previously trained model:

```python
# Load an existing pipeline
pipeline = ClassificationPipeline(output_path='path/to/saved/model')
pipeline.predictor = TabularPredictor.load('path/to/saved/model/model')

# Now you can use it for predictions
predictions = pipeline.predict(new_data)
```

## Performance Considerations

- For large datasets (>100K rows), use more powerful instances
- For many features (>100), increase training time
- For high-dimensional categorical features, expect longer training times
- For very high missing rates, model quality may degrade

## Using the Modular, Hyperparameter-Optimizable Pipeline

For advanced users, the repository includes a minimal, modular pipeline with pluggable preprocessing, feature engineering, and model selection, and built-in hyperparameter optimization using Optuna.

- File: `pipeline/minimal_modular_pipeline.py`
- Features:
  - Modular design: swap preprocessing, feature engineering, and model components
  - Joint hyperparameter optimization across all stages
  - Easy to extend with new options

### Example Usage

```bash
python pipeline/minimal_modular_pipeline.py
```

This will:
- Load `datasets/heart_train.csv`
- Split into train/validation
- Search over imputation, scaling, polynomial features, and n_estimators
- Print the best hyperparameters and validation ROC AUC

You can modify the config in the script to add more options or use your own data.

## Unified Pipeline Interface and Config

This repository supports two interchangeable pipelines for tabular classification:

- **Classic pipeline**: AutoGluon-based, with SageMaker integration, feature importance, and explainability.
- **Modular pipeline**: Pluggable, hyperparameter-optimizable, and research-friendly (Optuna HPO).

Both pipelines implement a unified interface (`fit`, `predict`, `predict_proba`, `evaluate`, `save`, `load`) and can be run with the same runner script and config file.

### Switching Pipelines

Set the `pipeline_type` key in your YAML config:

```yaml
pipeline_type: classic   # Use the classic AutoGluon pipeline
# pipeline_type: modular  # Use the modular, HPO-enabled pipeline
```

### Minimal Unified Config Example

```yaml
pipeline_type: modular
train_data: datasets/heart_train.csv
validation_data: datasets/heart_val.csv
test_data: datasets/heart_test.csv
target_column: target
preprocessing:
  impute_strategy: [mean, median, knn]
  scaling: [standard, none]
feature_engineering:
  add_polynomial: [True, False]
model:
  n_estimators: [100, 200]
# autogluon_hyperparameters:  # Only used for classic pipeline
#   GBM: {extra_trees: True}
```

### Unified Usage Example

```bash
python run_from_config.py --config your_config.yaml
```

- The runner will select the correct pipeline based on `pipeline_type`.
- Both pipelines support the same interface for training, evaluation, and saving/loading.

---

**Choose the pipeline that best fits your needs:**
- Use the classic pipeline for full AutoGluon support, SageMaker integration, and feature importance/explainability.
- Use the modular pipeline for research, rapid prototyping, or custom HPO across all pipeline stages.