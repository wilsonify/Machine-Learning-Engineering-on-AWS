import boto3
from sagemaker import get_execution_role

sagemaker_client = boto3.client('sagemaker-runtime')

role = get_execution_role()

cmd = ["curl", "http://169.254.169.254/latest/meta-data/identity-credentials/ec2/security-credentials/ec2-instance"]

cmd = ["curl", "169.254.170.2$AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"]
