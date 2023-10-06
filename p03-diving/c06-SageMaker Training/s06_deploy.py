import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.serializers import IdentitySerializer


def map_path(source):
    s3_bucket = "064592191516-ml-engineering"
    prefix = "ch06"
    return f's3://{s3_bucket}/{prefix}/{source}'


def map_input(source):
    path = map_path(source)
    return sagemaker.inputs.TrainingInput(
        s3_data=path,
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

output_path = map_path("output")

estimator = sagemaker.estimator.Estimator(
    image_uri=image,
    role=role,
    instance_count=2,
    instance_type='ml.p2.xlarge',
    output_path=output_path,
    sagemaker_session=session,
    enable_network_isolation=True
)

endpoint = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')
endpoint.serializer = IdentitySerializer(content_type="application/x-image")
