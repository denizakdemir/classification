import pytest
import yaml
from main import validate_config

def test_valid_config():
    config = {
        'pipeline_type': 'classic',
        'train_data': 'datasets/heart_train.csv',
        'target_column': 'target'
    }
    # Should not raise
    validate_config(config)

def test_missing_required_keys():
    config = {
        'pipeline_type': 'classic',
        'train_data': 'datasets/heart_train.csv',
        # 'target_column' missing
    }
    with pytest.raises(ValueError) as excinfo:
        validate_config(config)
    assert 'Missing required config keys' in str(excinfo.value) 