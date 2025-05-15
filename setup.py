from setuptools import setup, find_packages

setup(
    name="classification",
    version="0.1",
    packages=find_packages(),
    py_modules=["classification_pipeline", "sagemaker_deployment"],
    install_requires=[],
) 