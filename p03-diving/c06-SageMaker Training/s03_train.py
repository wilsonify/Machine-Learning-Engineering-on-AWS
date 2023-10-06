import boto3
import sagemaker
from sagemaker import image_uris


def map_input(source):
    return sagemaker.inputs.TrainingInput(
        s3_data=f's3://064592191516-ml-engineering/ch06/{source}',
        distribution='FullyReplicated',
        content_type='application/x-image',
        s3_data_type='S3Prefix'
    )


session = sagemaker.Session()
role = sagemaker.get_execution_role()
region_name = boto3.Session().region_name

image = image_uris.retrieve(
    framework="image-classification",
    region=region_name,
    version="1"
)

# -

data_channels = {
    "train": map_input("train"),
    "validation": map_input("validation"),
    "train_lst": map_input("train_lst"),
    "validation_lst": map_input("validation_lst")
}

output_path = f's3://064592191516-ml-engineering/ch06/output'

estimator = sagemaker.estimator.Estimator(
    image_uri=image,
    role=role,
    instance_count=2,
    instance_type='ml.p2.xlarge',
    output_path=output_path,
    sagemaker_session=session,
    enable_network_isolation=True
)

hyperparameters = {
    'num_training_samples': 100,
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
print(f"estimator.latest_training_job.name = {estimator.latest_training_job.name}")
