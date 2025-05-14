install:
	pip install -r requirements.txt

test:
	pytest tests/

run:
	python main.py --config example_config.yaml 