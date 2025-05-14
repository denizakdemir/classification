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
- Feature importance and partial dependence analysis
- Configurable via YAML (local or S3)
- Organized output directories by model/data names
- SageMaker-ready

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
train_data: s3://your-bucket/path/to/train.csv
validation_data: s3://your-bucket/path/to/validation.csv  # optional
test_data: s3://your-bucket/path/to/test.csv  # optional
target_column: target
predictors:
  - age
  - gender
  - cholesterol
  - blood_pressure
model_name: my_model
run_name: experiment1
output_path: ./results  # base directory for outputs
missingness_threshold: 0.75  # Max allowed missing fraction per column (default 0.75)
categorical_threshold: 15    # Max unique values for a column to be considered categorical (default 15)
text_threshold: 100          # Min unique values for a column to be considered text (default 100)
random_seed: 42              # Random seed for reproducibility (numpy, random, torch)
autogluon_hyperparameters:   # Custom hyperparameters for AutoGluon TabularPredictor.fit (optional)
  # Example:
  # GBM: {extra_trees: True, ag_args: {name_suffix: 'EXTRA'}}
save_leaderboard: true
save_predictions: true
save_config_copy: true
time_limit: 3600
presets: best_quality
```

### Parameter Descriptions
- `missingness_threshold`: Maximum allowed fraction of missing values per column before dropping (default 0.75).
- `categorical_threshold`: Max unique values for a column to be considered categorical (default 15).
- `text_threshold`: Min unique values for a column to be considered text (default 100).
- `random_seed`: Sets the random seed for numpy, random, and torch for reproducibility.
- `autogluon_hyperparameters`: Dictionary of custom hyperparameters passed to AutoGluon TabularPredictor.fit (see AutoGluon docs for options).

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

---

For more, see `pipeline_usage_guide.md` and `run_locally.md`.