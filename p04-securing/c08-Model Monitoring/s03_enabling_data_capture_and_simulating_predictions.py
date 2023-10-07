# name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:236514542706:image/datascience-1.0

import random
from time import sleep

import sagemaker
from sagemaker import get_execution_role
from sagemaker.deserializers import CSVDeserializer
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer


def generate_random_payload():
    x = random.randint(-5, 5)
    y = random.randint(-5, 5)
    return f"{x},{y}"


s3_bucket = ""
prefix = ""
ll_package_arn = ""
endpoint_name = ""
base = f"s3://{s3_bucket}/{prefix}"
capture_upload_path = f"{base}/data-capture"

session = sagemaker.Session()
role = get_execution_role()

predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=session,
    role=role,
    serializer=CSVSerializer(),
    deserializer=CSVDeserializer()
)

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=capture_upload_path,
    capture_options=["REQUEST", "RESPONSE"],
    csv_content_types=["text/csv"],
    json_content_types=["application/json"]

)

predictor.update_data_capture_config(
    data_capture_config=data_capture_config
)

generate_random_payload()


def perform_good_input():
    print("> PERFORM REQUEST WITH GOOD INPUT")
    payload = generate_random_payload()
    result = predictor.predict(data=payload)
    print(result)


def perform_bad_input():
    print("> PERFORM REQUEST WITH BAD INPUT")
    payload = generate_random_payload() + ".50"
    result = predictor.predict(data=payload)
    print(result)


perform_good_input()

perform_bad_input()


def generate_sample_requests():
    for i in range(0, 2 * 240):
        print(f"ITERATION # {i}")
        perform_good_input()
        perform_bad_input()
        print("> SLEEPING FOR 30 SECONDS")
        sleep(30)


generate_sample_requests()
