# AWS SageMaker Deployment Script
# This script demonstrates how to deploy the classification pipeline to SageMaker

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Set up SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()  # Make sure your IAM role has necessary permissions
bucket = sagemaker_session.default_bucket()
prefix = 'autogluon-classification'

# Define the SageMaker code entry points
script_path = 'pipeline_entry_point.py'

# Write the entry point script
with open(script_path, 'w') as f:
    f.write("""
import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

# Make sure the script folder is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Classification Pipeline
from classification_pipeline import ClassificationPipeline, SagemakerEntrypoint

if __name__ == '__main__':
    SagemakerEntrypoint.train()
""")

# Prepare your data
def prepare_sample_data():
    """Prepare sample data for demonstration."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    # Load sample data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Add some missing values
    np.random.seed(42)
    for col in X.columns[:5]:
        mask = np.random.random(len(X)) < 0.1  # 10% missing
        X.loc[mask, col] = np.nan
    
    # Create a dataframe
    df = X.copy()
    df['target'] = y
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save to CSV
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    train_df.to_csv('data/train/train.csv', index=False)
    test_df.to_csv('data/test/test.csv', index=False)
    
    return 'data/train/train.csv', 'data/test/test.csv'

# Prepare data
train_path, test_path = prepare_sample_data()

# Upload data to S3
train_s3 = sagemaker_session.upload_data(
    path='data/train',
    bucket=bucket,
    key_prefix=f"{prefix}/train"
)

test_s3 = sagemaker_session.upload_data(
    path='data/test',
    bucket=bucket,
    key_prefix=f"{prefix}/test"
)

# Create SageMaker estimator
sagemaker_estimator = Estimator(
    entry_point='pipeline_entry_point.py',
    source_dir='.',  # Include all files in current directory
    role=role,
    instance_count=1,
    instance_type='ml.m5.2xlarge',  # Choose appropriate instance
    volume_size=30,
    max_run=3600,  # 1 hour
    output_path=f's3://{bucket}/{prefix}/output',
    sagemaker_session=sagemaker_session,
    base_job_name='autogluon-classification',
    hyperparameters={
        'target-column': 'target',
        'time-limit': 1800,  # 30 minutes
        'presets': 'best_quality'
    }
)

# Define training inputs
training_inputs = {
    'train': TrainingInput(
        s3_data=train_s3,
        content_type='text/csv'
    ),
    'test': TrainingInput(
        s3_data=test_s3,
        content_type='text/csv'
    )
}

# Start training job
print("Starting SageMaker training job...")
sagemaker_estimator.fit(inputs=training_inputs)

# Deploy model to endpoint
print("Deploying model to endpoint...")
predictor = sagemaker_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer()
)

# Example inference request
print("Making test prediction...")
sample_df = pd.read_csv('data/test/test.csv').drop(columns=['target']).iloc[:5]
prediction = predictor.predict(sample_df.values)
print(f"Prediction result: {prediction}")

# Clean up (optional)
print("Cleaning up endpoint...")
predictor.delete_endpoint()
print("Done!")

# Instructions for manual inference
print("\nTo use your model for inference after deployment:")
print(f"Endpoint name: {predictor.endpoint_name}")
print("Sample Python code for inference:")
print("""
import boto3
import pandas as pd
import numpy as np
import json

# Load your data
data = pd.read_csv('your_data.csv')

# Preprocess data if needed (the endpoint handles missing values)

# Create SageMaker runtime client
runtime = boto3.client('sagemaker-runtime')

# Make prediction
response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='text/csv',
    Body=data.to_csv(index=False, header=False).encode('utf-8')
)

# Parse results
result = json.loads(response['Body'].read().decode())
print(result)
""") 