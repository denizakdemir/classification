# Running the Classification Pipeline Locally

This guide explains how to set up a Python environment and run the ML classification pipeline on your local machine—no AWS or SageMaker required.

---

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

---

## 2. Set Up a Python Environment

You can use either **conda** or **venv**. Python 3.8+ is recommended.

### Using Conda (Recommended)
```bash
conda create -n ml-pipeline python=3.9 -y
conda activate ml-pipeline
```

### Using venv (Standard Library)
```bash
python3 -m venv ml-pipeline-env
source ml-pipeline-env/bin/activate  # On Windows: ml-pipeline-env\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Prepare Your Data and Config
- Place your CSV data file(s) in the project directory.
- Edit or copy `example_config.yaml` to match your data (set `train_data`, `target_column`, etc.).

---

## 5. Run the Pipeline Using a Config File

```bash
python run_from_config.py --config ./example_config.yaml
```
- Results, models, and artifacts will be saved to the output directory specified in your config.

---

## 6. (Optional) Run the Pipeline in a Python Script or Notebook

```python
from pipeline.pipeline import ClassificationPipeline
import pandas as pd

# Load your data
train_df = pd.read_csv('your_train.csv')
test_df = pd.read_csv('your_test.csv')

pipeline = ClassificationPipeline(output_path='./model_output')
pipeline.train(
    train_data=train_df,
    target_column='your_target_column',
    time_limit=600,
    presets='good_quality'
)
evaluation = pipeline.evaluate(test_df, 'your_target_column')
print(evaluation)
preds = pipeline.predict(test_df.drop(columns=['your_target_column']))
print(preds.head())
```

---

## 7. Troubleshooting
- **Missing packages?** Run `pip install -r requirements.txt` again.
- **Permission errors?** Try running as administrator or check file paths.
- **Data errors?** Ensure your CSV files and config match (column names, etc.).
- **AutoGluon errors?** See [AutoGluon documentation](https://auto.gluon.ai/stable/index.html).

---

## 8. Deactivating the Environment

- For conda: `conda deactivate`
- For venv: `deactivate`

---

## 9. Cleaning Up
- Remove the environment: `conda remove -n ml-pipeline --all` or delete the venv folder.
- Delete output/model directories if needed.

---

## 10. Unified Pipeline Interface and Config

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

---

## Need Help?
- Check the `

- SHAP is used for all feature effect and importance analysis (no PDPs)
- Missing value indicators can be combined or separated in importance analysis
- Categorical variables are handled in SHAP analysis