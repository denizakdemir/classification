from setuptools import setup, find_packages

setup(
    name="classification-pipeline",
    version="0.1",
    description="A modular, extensible pipeline for tabular classification with AutoGluon and SageMaker integration.",
    packages=find_packages(),
    install_requires=[
        "pandas==1.5.3",
        "numpy==1.24.4",
        "pyyaml==6.0.1",
        "boto3==1.28.57",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "autogluon.tabular==0.8.2",
        "shap==0.43.0",
        "sagemaker==2.197.0",
        "scikit-learn==1.2.2"
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/classification-pipeline",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 