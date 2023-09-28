# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (Data Science)
#     language: python
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:081325390199:image/datascience-1.0
# ---

# %store -r model_data
# %store -r s3_bucket
# %store -r prefix

input_data = "s3://{}/{}/data/input.json".format(s3_bucket,prefix)
input_data

# !aws s3 cp data/input.json {input_data}

# +
from sagemaker import get_execution_role 

role = get_execution_role()

# +
from sagemaker.async_inference import AsyncInferenceConfig

output_path = f"s3://{s3_bucket}/{prefix}/output"
async_config = AsyncInferenceConfig(output_path=output_path)

# +
from sagemaker.pytorch.model import PyTorchModel

model = PyTorchModel(
    model_data=model_data, 
    role=role, 
    source_dir="scripts",
    entry_point='inference.py', 
    framework_version='1.6.0',
    py_version="py3"
)

# +
# %%time

from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = model.deploy(
    instance_type='ml.m5.xlarge', 
    initial_instance_count=1,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
    async_inference_config=async_config
)
# -

response = predictor.predict_async(input_path=input_data)

# +
from time import sleep

sleep(40)

response.get_result()
# -

output_path = response.output_path
output_path

# !aws s3 cp {output_path} sample.out

# !cat sample.out

# !rm sample.out

predictor.delete_endpoint()
