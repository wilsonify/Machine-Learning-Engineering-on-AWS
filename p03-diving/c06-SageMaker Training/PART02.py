import random
import string

import sagemaker
from sagemaker.estimator import Estimator


def generate_random_string():
    return ''.join(random.sample(string.ascii_uppercase, 12))


image = "811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:1"
job_name = ""
role = sagemaker.get_execution_role()
s3_bucket = "064592191516-ml-engineering"
prefix = "ch06"


def glob_s3(glob_pattern):
    import boto3
    import fnmatch
    s3_client = boto3.client('s3')
    first_asterisk_index = glob_pattern.find('*')
    s3_prefix = glob_pattern[:first_asterisk_index] if first_asterisk_index >= 0 else glob_pattern
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    matched_objects = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        is_matched = fnmatch.fnmatch(key, glob_pattern)
        if is_matched:
            matched_objects.append(key)
    return matched_objects


session = sagemaker.Session()
previous = Estimator.attach(job_name)

previous.logs()

model_data = previous.model_data
print(f"model_data={model_data}")

base_job_name = generate_random_string()
print(f"base_job_name = {base_job_name}")

checkpoint_folder = "checkpoints"
checkpoint_s3_bucket = f"s3://{s3_bucket}/{base_job_name}/{checkpoint_folder}"
checkpoint_local_path = "/opt/ml/checkpoints"

training_samples = glob_s3(f"tmp2/train/*/*.png")
len(training_samples)


def map_input(source):
    path = f's3://{s3_bucket}/{prefix}/{source}'
    return sagemaker.inputs.TrainingInput(
        path,
        distribution='FullyReplicated',
        content_type='application/x-image',
        s3_data_type='S3Prefix'
    )


data_channels = {
    "train": map_input("train"),
    "validation": map_input("validation"),
    "train_lst": map_input("train_lst"),
    "validation_lst": map_input("validation_lst")
}

output_path = f's3://{s3_bucket}/{prefix}/output'
print(f"output_path = {output_path}")

estimator = sagemaker.estimator.Estimator(
    image_uri=image,
    role=role,
    instance_count=2,
    instance_type='ml.p2.xlarge',
    output_path=output_path,
    sagemaker_session=session,
    enable_network_isolation=True,
    model_uri=model_data,
    use_spot_instances=True,
    max_run=1800,
    max_wait=3600,
    base_job_name=base_job_name,
    checkpoint_s3_uri=checkpoint_s3_bucket,
    checkpoint_local_path=checkpoint_local_path
)

hyperparameters = {
    'num_training_samples': len(training_samples),
    'num_layers': 18,
    'image_shape': "1,28,28",
    'num_classes': 10,
    'mini_batch_size': 100,
    'epochs': 3,
    'learning_rate': 0.01,
    'top_k': 5,
    'precision_dtype': 'float32'
}

estimator.set_hyperparameters(**hyperparameters)

print(f"estimator.__dict__ = {estimator.__dict__}")

estimator.fit(inputs=data_channels, logs="All")

print(f"estimator.model_data = {estimator.model_data}")

f"checkpoint_s3_uri = {estimator.checkpoint_s3_uri}"
