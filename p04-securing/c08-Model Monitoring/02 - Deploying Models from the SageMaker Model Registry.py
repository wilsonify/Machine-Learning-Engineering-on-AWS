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
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/datascience-1.0
# ---

# %store -r knn_package_arn
# %store -r ll_package_arn

# +
import sagemaker
from sagemaker import get_execution_role
from sagemaker import ModelPackage
from sagemaker.predictor import Predictor


session = sagemaker.Session()
role = get_execution_role()

model = ModelPackage(
    role=role,
    model_package_arn=knn_package_arn,
    sagemaker_session=session
)

model.predictor_cls = Predictor

# +
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = model.deploy(
    instance_type='ml.m5.xlarge', 
    initial_instance_count=1,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

# +
payload = {
    'instances': [
        {
          "features": [ 1.5, 2 ]
        },
    ]
}

predictor.predict(data=payload)


# -

def process_prediction_result(raw_result):
    first = raw_result['predictions'][0]
    return first['predicted_label']


def predict(x, y, predictor=predictor):
    payload = {
        'instances': [
            {
              "features": [ x, y ]
            },
        ]
    }
    
    raw_result = predictor.predict(
        data=payload
    )
    
    return process_prediction_result(raw_result)


predict(x=3, y=4)

# +
from time import sleep

def test_different_values(predictor=predictor):
    for x in range(-3, 3+1):
        for y in range(-3, 3+1):
            label = predict(x=x, y=y, predictor=predictor)
            print(f"x={x}, y={y}, label={label}")
            sleep(0.2)


# -

test_different_values()

# +
import boto3
client = boto3.client(service_name="sagemaker")


def create_model(model_package_arn, model_name, role=role, client=client):
    container_list = [
        {'ModelPackageName': model_package_arn}
    ]

    response = client.create_model(
        ModelName = model_name,
        ExecutionRoleArn = role,
        Containers = container_list
    )
    
    return response["ModelArn"]


# +
import string 
import random

def generate_random_string():
    return ''.join(
        random.sample(
        string.ascii_uppercase,12)
    )


model_name = f"ll-{generate_random_string()}"

model_arn = create_model(
    model_package_arn=ll_package_arn,
    model_name=model_name
)


# -

def create_endpoint_config(model_name, config_name, client=client):
    response = client.create_endpoint_config(
        EndpointConfigName = config_name,
        ProductionVariants=[{
            'InstanceType': "ml.m5.xlarge",
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'
        }]
    )

    return response["EndpointConfigArn"]


# +
config_name = f"config-{generate_random_string()}"

config_arn = create_endpoint_config(
    model_name=model_name,
    config_name=config_name
)
# -

response = client.update_endpoint(
    EndpointName=predictor.endpoint_name,
    EndpointConfigName=config_name
)

print('Wait for update operation to complete')
sleep(60*5)

predictor = Predictor(
    endpoint_name=predictor.endpoint_name,
    sagemaker_session=session,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

# +
payload = {
    'instances': [
        {
          "features": [ 1.5, 2 ]
        },
    ]
}

predictor.predict(data=payload)
# -

test_different_values(predictor=predictor)

# +
# predictor.delete_endpoint()
# -

endpoint_name = predictor.endpoint_name
# %store endpoint_name
