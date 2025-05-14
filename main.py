import argparse
import sys
import os
import yaml


def validate_config(config):
    required_keys = ['pipeline_type', 'train_data', 'target_column']
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    # Add more validation as needed


def main():
    parser = argparse.ArgumentParser(description='Unified ML Pipeline Runner')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    try:
        validate_config(config)
    except Exception as e:
        print(f"Config validation error: {e}")
        sys.exit(1)

    print(f"\n[INFO] Running pipeline: {config.get('pipeline_type', 'classic')}\n")
    print(f"[INFO] Config summary:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Dispatch to run_from_config.py
    import run_from_config
    run_from_config.run_from_config_main(config)

if __name__ == '__main__':
    main() 