# name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:236514542706:image/datascience-1.0


import random
import string
from time import sleep

import boto3
import sagemaker
from sagemaker import ModelPackage
from sagemaker import get_execution_role
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

knn_package_arn = ""
ll_package_arn = ""


def create_model(model_package_arn, model_name, role, client):
    container_list = [{'ModelPackageName': model_package_arn}]
    response = client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        Containers=container_list
    )
    return response["ModelArn"]


session = sagemaker.Session()
role = get_execution_role()
client = boto3.client(service_name="sagemaker")
model = ModelPackage(
    role=role,
    model_package_arn=knn_package_arn,
    sagemaker_session=session
)

model.predictor_cls = Predictor

predictor = model.deploy(
    instance_type='ml.m5.xlarge',
    initial_instance_count=1,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

predictor.predict(data={'instances': [{"features": [1.5, 2]}, ]})


def predict(x_in, y_in):
    raw_result = predictor.predict(
        data={'instances': [{"features": [x_in, y_in]}, ]}
    )
    processed_result = raw_result['predictions'][0]['predicted_label']
    return processed_result


predict(3, 4)

for x in range(-3, 3 + 1):
    for y in range(-3, 3 + 1):
        label = predict(x, y)
        print(f"x={x}, y={y}, label={label}")
        sleep(0.2)

rand_str = ''.join(random.sample(string.ascii_uppercase, 12))
model_name = f"ll-{rand_str}"

model_arn = create_model(
    model_package_arn=ll_package_arn,
    model_name=model_name
)


def create_endpoint_config(model_name, config_name, client=client):
    response = client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'InstanceType': "ml.m5.xlarge",
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'
        }]
    )

    return response["EndpointConfigArn"]


rand_str = ''.join(random.sample(string.ascii_uppercase, 12))
config_name = f"config-{rand_str}"

config_arn = create_endpoint_config(
    model_name=model_name,
    config_name=config_name
)

response = client.update_endpoint(
    EndpointName=predictor.endpoint_name,
    EndpointConfigName=config_name
)

print('Wait for update operation to complete')
sleep(60 * 5)
print('Done Waiting for update operation to complete')

predictor = Predictor(
    endpoint_name=predictor.endpoint_name,
    sagemaker_session=session,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

predictor.predict(data={'instances': [{"features": [1.5, 2]}, ]})

for x in range(-3, 3 + 1):
    for y in range(-3, 3 + 1):
        label = predict(x, y)
        print(f"x={x}, y={y}, label={label}")
        sleep(0.2)

endpoint_name = predictor.endpoint_name
print(f"endpoint_name={endpoint_name}")
predictor.delete_endpoint()
