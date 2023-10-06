import boto3
import sagemaker
from sagemaker.base_deserializers import BytesDeserializer
from sagemaker.base_serializers import IdentitySerializer

session = sagemaker.Session()
role = sagemaker.get_execution_role()
region_name = boto3.Session().region_name
registry = ""
hostname = ""
repository = ""
image = f"{registry}.dkr.{hostname}/{repository}"

estimator = sagemaker.estimator.Estimator(
    image_uri=image,
    role=role,
    instance_count=2,
    instance_type='ml.p2.xlarge',
    output_path='s3://064592191516-ml-engineering/ch06/output',
    sagemaker_session=session,
    enable_network_isolation=True
)

endpoint = sagemaker.predictor.Predictor(
    endpoint_name="",
    sagemaker_session=session,
    serializer=IdentitySerializer(content_type="application/x-image"),
    deserializer=BytesDeserializer()
)

endpoint.delete_endpoint()
