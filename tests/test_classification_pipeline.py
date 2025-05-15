import pytest
import numpy as np
import pandas as pd
from classification_pipeline import MissingDataHandler, FeatureEngineering, ClassificationPipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = int

# --- Unit tests: MissingDataHandler ---
def test_missing_data_handler_basic():
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [np.nan, np.nan, np.nan, 1],
        'c': [1, 1, 1, 1]
    })
    handler = MissingDataHandler(threshold=0.5)
    stats = handler.analyze_missing(df)
    assert 'b' in stats['cols_to_drop']
    df2 = handler.handle_missing(df)
    assert 'b' not in df2.columns
    assert 'a_missing' in df2.columns
    assert set(df2['a_missing']) == {0, 1}

def test_missing_data_handler_no_missing():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    handler = MissingDataHandler()
    stats = handler.analyze_missing(df)
    assert stats['cols_to_drop'] == []
    df2 = handler.handle_missing(df)
    assert all(col in df2.columns for col in ['a', 'b'])

# --- Unit tests: FeatureEngineering ---
def test_feature_engineering_types_and_date():
    df = pd.DataFrame({
        'cat': ['a', 'b', 'a'],
        'num': [1.0, 2.0, 3.0],
        'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
    })
    fe = FeatureEngineering()
    types = fe.identify_feature_types(df)
    assert 'cat' in types['categorical_features']
    assert 'num' in types['numeric_features']
    assert 'date' in types['date_features']
    df2 = fe.create_date_features(df)
    assert all(f'date_{suffix}' in df2.columns for suffix in ['year', 'month', 'day', 'dayofweek', 'quarter'])
    assert 'date' not in df2.columns

def test_feature_engineering_onehot():
    df = pd.DataFrame({'cat': ['a', 'b', 'a'], 'num': [1, 2, 3]})
    fe = FeatureEngineering(use_onehot=True)
    fe.identify_feature_types(df)
    df2 = fe.fit_transform(df)
    assert any('cat_' in c for c in df2.columns)
    df3 = fe.transform(df)
    assert df2.shape == df3.shape

def test_feature_engineering_tfidf():
    df = pd.DataFrame({'text': ['hello world', 'foo bar', 'hello foo']})
    fe = FeatureEngineering(use_tfidf=True, text_threshold=1)
    fe.identify_feature_types(df)
    df2 = fe.fit_transform(df)
    assert any('text_tfidf_' in c for c in df2.columns)
    df3 = fe.transform(df)
    assert df2.shape == df3.shape

def test_feature_engineering_poly():
    df = pd.DataFrame({'num1': [1, 2, 3], 'num2': [4, 5, 6]})
    fe = FeatureEngineering(use_poly=True)
    fe.identify_feature_types(df)
    df2 = fe.fit_transform(df)
    assert df2.shape[1] > 2  # Should have interaction terms
    df3 = fe.transform(df)
    assert df2.shape == df3.shape

# --- Integration test: ClassificationPipeline ---
@pytest.mark.slow
def test_classification_pipeline_end_to_end():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
    df['target'] = y
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    pipeline = ClassificationPipeline(output_path='./test_model_output')
    pipeline.train(train_data=train_df, target_column='target', time_limit=10, presets='good_quality', hyperparameters={"RF": {}, "XT": {}, "KNN": {}})
    eval_result = pipeline.evaluate(test_df, 'target')
    assert 'performance' in eval_result
    preds = pipeline.predict(test_df.drop(columns=['target']))
    assert len(preds) == len(test_df)
    # Check that probabilities are in [0, 1]
    for col in preds.columns:
        if col != 'prediction':
            assert ((preds[col] >= 0) & (preds[col] <= 1)).all()
    # Check explanations
    explanations = pipeline.explain_prediction(test_df.drop(columns=['target']).iloc[:5])
    assert 'explanations' in explanations

# --- Edge case tests ---
def test_all_missing_column():
    df = pd.DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [1, 2, 3]})
    handler = MissingDataHandler(threshold=0.5)
    stats = handler.analyze_missing(df)
    assert 'a' in stats['cols_to_drop']
    df2 = handler.handle_missing(df)
    assert 'a' not in df2.columns

def test_all_constant_column():
    df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 3, 4]})
    fe = FeatureEngineering()
    types = fe.identify_feature_types(df)
    # Should still identify 'a' as numeric or categorical
    assert 'a' in types['numeric_features'] or 'a' in types['categorical_features']

def test_all_unique_column():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    fe = FeatureEngineering()
    types = fe.identify_feature_types(df)
    # Should identify 'a' and 'b' as numeric
    assert set(['a', 'b']).issubset(set(types['numeric_features']))

def test_small_dataset():
    # Use at least two rows with different target values
    df = pd.DataFrame({'a': [1, 2], 'b': [2, 3], 'target': [0, 1]})
    pipeline = ClassificationPipeline(output_path='./test_model_output')
    pipeline.train(train_data=df, target_column='target', time_limit=5, presets='good_quality', hyperparameters={"RF": {}, "XT": {}, "KNN": {}})
    preds = pipeline.predict(df.drop(columns=['target']))
    assert len(preds) == 2

def test_only_categorical():
    df = pd.DataFrame({'cat': ['a', 'b', 'c'], 'target': [0, 1, 0]})
    pipeline = ClassificationPipeline(output_path='./test_model_output', use_onehot=True)
    pipeline.train(train_data=df, target_column='target', time_limit=5, presets='good_quality', hyperparameters={"RF": {}, "XT": {}, "KNN": {}})
    preds = pipeline.predict(df.drop(columns=['target']))
    assert len(preds) == 3

def test_only_numeric():
    df = pd.DataFrame({'num1': [1, 2, 3], 'num2': [4, 5, 6], 'target': [0, 1, 0]})
    pipeline = ClassificationPipeline(output_path='./test_model_output')
    pipeline.train(train_data=df, target_column='target', time_limit=5, presets='good_quality', hyperparameters={"RF": {}, "XT": {}, "KNN": {}})
    preds = pipeline.predict(df.drop(columns=['target']))
    assert len(preds) == 3

def test_only_text():
    df = pd.DataFrame({'text': ['foo', 'bar', 'baz'], 'target': [0, 1, 0]})
    pipeline = ClassificationPipeline(output_path='./test_model_output', use_tfidf=True, text_threshold=1)
    pipeline.train(train_data=df, target_column='target', time_limit=5, presets='good_quality', hyperparameters={"RF": {}, "XT": {}, "KNN": {}})
    preds = pipeline.predict(df.drop(columns=['target']))
    assert len(preds) == 3 