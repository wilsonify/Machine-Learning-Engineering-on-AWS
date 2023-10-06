#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:081325390199:image/datascience-1.0
# !aws s3 cp data/input.json {input_data}

from time import sleep

from sagemaker import get_execution_role
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.serializers import JSONSerializer

model_data = ""
s3_bucket = ""
prefix = ""
input_data = f"s3://{s3_bucket}/{prefix}/data/input.json"

role = get_execution_role()

output_path = f"s3://{s3_bucket}/{prefix}/output"
async_config = AsyncInferenceConfig(output_path=output_path)

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
    async_inference_config=async_config
)

response = predictor.predict_async(input_path=input_data)

sleep(40)

response.get_result()

output_path = response.output_path

# !aws s3 cp {output_path} sample.out

# !cat sample.out

# !rm sample.out

predictor.delete_endpoint()
