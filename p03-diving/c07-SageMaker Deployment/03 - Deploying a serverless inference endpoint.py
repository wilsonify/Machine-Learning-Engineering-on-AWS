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

# +
from sagemaker import get_execution_role 

role = get_execution_role()

# +
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
  memory_size_in_mb=4096,
  max_concurrency=5,
)

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
    serverless_inference_config=serverless_config
)

# +
payload = {
    "text": "I love reading the book MLE on AWS!"
}

predictor.predict(payload)

# +
payload = {
    "text": "This is the worst spaghetti I've ever tasted"
}

predictor.predict(payload)
# -

predictor.delete_endpoint()
