#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:081325390199:image/datascience-1.0
from sagemaker import get_execution_role
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.serverless import ServerlessInferenceConfig

model_data = ""
s3_bucket = ""
prefix = ""

role = get_execution_role()

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,
    max_concurrency=5,
)

model = PyTorchModel(
    model_data=model_data,
    role=role,
    source_dir="scripts",
    entry_point='inference.py',
    framework_version='1.6.0',
    py_version="py3"
)

predictor = model.deploy(
    instance_type='ml.m5.xlarge',
    initial_instance_count=1,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
    serverless_inference_config=serverless_config
)

predictor.predict({"text": "I love reading the book MLE on AWS!"})
predictor.predict({"text": "This is the worst spaghetti I've ever tasted"})
predictor.delete_endpoint()
