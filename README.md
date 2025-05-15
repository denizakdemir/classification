# ML Classification Pipeline

A robust, configurable pipeline for tabular classification tasks, supporting missing data, feature importance, partial dependence plots, and AWS SageMaker integration. Organizes results by model/data names and tracks all key artifacts.

## Quickstart

1. Clone the repository and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   pip install -r requirements.txt
   ```
2. Prepare your data and config (see `example_config.yaml`).
3. Run the pipeline:
   ```bash
   python main.py --config your_config.yaml
   ```
   - The runner will auto-detect the pipeline type and validate your config.
   - Results and models are saved to the output directory specified in your config.

## Features Table

| Area                | Improvement                                      |
|---------------------|-------------------------------------------------|
| Entry Point         | Unified `main.py` or CLI                        |
| Error Handling      | Config validation, clear errors                  |
| Installation        | `requirements.txt`, `Makefile` (optional)        |
| Documentation       | Quickstart, examples, comments                   |
| CLI                 | Helpful flags, print config at start             |
| Packaging           | Proper package, relative imports                 |
| Output              | Structured results, auto-create dirs             |
| Testing             | Add `tests/`, unit tests (see roadmap)           |
| Extensibility       | Easy to add new models/steps                     |
| User Experience     | Progress bars, result summary                    |
| Notebook            | Jupyter example (see `notebooks/`)               |
| Config              | Minimal configs, warn on typos                   |
| Deployment          | Add deployment guide (see `sagemaker_deployment.py`) |
| Web UI (optional)   | Streamlit/Gradio interface (see roadmap)         |

## Features
- Automated missing data handling
- Probabilistic and class predictions
- Feature importance and SHAP-based effect analysis
- Configurable via YAML (local or S3)
- Organized output directories by model/data names
- SageMaker-ready
- **Modular feature engineering**: Enable one-hot encoding, TF-IDF, or polynomial features via config toggles
- Modular pipeline code is now split into:
  - `pipeline/pipeline.py`: Main pipeline orchestration
  - `pipeline/missing_data.py`, `pipeline/feature_engineering.py`, `pipeline/interpretation.py`, etc.: Modular components
- SHAP-based feature importance and effect analysis (global and local)
- Missing value indicators are handled and can be combined or separated in importance analysis

## Usage
1. Prepare your data CSV(s) and a config YAML (see `example_config.yaml`).
2. Run:
   ```bash
   python main.py --config your_config.yaml
   ```
   - Or, for backward compatibility:
   ```bash
   python run_from_config.py --config your_config.yaml
   ```

## Configuration File
See `example_config.yaml` for all supported options. Example:
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

### Parameter Descriptions
- `missingness_threshold`: Maximum allowed fraction of missing values per column before dropping (default 0.75).
- `categorical_threshold`: Max unique values for a column to be considered categorical (default 15).
- `text_threshold`: Min unique values for a column to be considered text (default 100).
- `random_seed`: Sets the random seed for numpy, random, and torch for reproducibility.
- `autogluon_hyperparameters`: Dictionary of custom hyperparameters passed to AutoGluon TabularPredictor.fit (see AutoGluon docs for options).
- `use_onehot`: Enable one-hot encoding for categorical features (default false)
- `use_tfidf`: Enable TF-IDF vectorization for text features (default false)
- `use_poly`: Enable polynomial features for numeric features (default false)

All these parameters are passed to the pipeline and affect feature detection, reproducibility, and model training.

## Output Organization
Results are saved under the output directory specified in your config, organized by model and run name.

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
python main.py --config your_config.yaml
```

- The runner will select the correct pipeline based on `pipeline_type`.
- Both pipelines support the same interface for training, evaluation, and saving/loading.

## Example: Heart Disease Dataset

This repository includes two ready-to-use config files for the heart disease dataset:

- `heart_classic_config.yaml`: For the classic AutoGluon pipeline
- `heart_config.yaml`: For the modular (Optuna/scikit-learn) pipeline

### Running the Classic (AutoGluon) Pipeline

```bash
python main.py --config heart_classic_config.yaml
```

### Running the Modular (Optuna) Pipeline

```bash
python main.py --config heart_config.yaml
```

You can switch between pipelines by changing the `pipeline_type` and config structure. See the example YAML files for details.

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

---

For more, see `pipeline_usage_guide.md` and `run_locally.md`.

from pipeline.pipeline import ClassificationPipeline

### SHAP-based Feature Effect and Importance Analysis

The pipeline uses [SHAP](https://shap.readthedocs.io/) for all feature effect and importance analysis:

- **Global feature importance**: SHAP values are aggregated to show which features (including missing value indicators) most influence model predictions.
- **Combined/separate missingness**: You can view the importance of a variable and its missingness indicator either combined or separately.
- **Categorical support**: Categorical variables are handled appropriately in SHAP analysis.

SHAP summary and dependence plots are available for interpretation. See the usage guide for examples.