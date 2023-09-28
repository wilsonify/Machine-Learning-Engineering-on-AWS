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

# %store -r s3_bucket
# %store -r prefix

# %store -r ll_package_arn
# %store -r endpoint_name

# +
import sagemaker
from sagemaker import get_execution_role
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer

session = sagemaker.Session()
role = get_execution_role()

predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=session,
    role=role,
    serializer=CSVSerializer(),
    deserializer=CSVDeserializer()
)

# +
from sagemaker.model_monitor import DataCaptureConfig

base = f"s3://{s3_bucket}/{prefix}"
capture_upload_path = f"{base}/data-capture"

capture_config_dict = {
    'enable_capture': True,
    'sampling_percentage': 100,
    'destination_s3_uri': capture_upload_path,
    'kms_key_id': None,
    'capture_options': ["REQUEST", "RESPONSE"],
    'csv_content_types': ["text/csv"],
    'json_content_types': ["application/json"]
}

data_capture_config = DataCaptureConfig(
    **capture_config_dict
)

# +
# %%time

predictor.update_data_capture_config(
    data_capture_config=data_capture_config
)
# -

# %store capture_upload_path

# +
import random

def generate_random_payload():
    x = random.randint(-5,5)
    y = random.randint(-5,5)
    
    return f"{x},{y}"


# -

generate_random_payload()


# +
def perform_good_input(predictor):
    print("> PERFORM REQUEST WITH GOOD INPUT")
    payload = generate_random_payload()
    result = predictor.predict(data=payload)
    print(result)


def perform_bad_input(predictor):
    print("> PERFORM REQUEST WITH BAD INPUT")
    payload = generate_random_payload() + ".50"
    result = predictor.predict(data=payload)
    print(result)


# -

perform_good_input(predictor)

perform_bad_input(predictor)

# +
from time import sleep

def generate_sample_requests(predictor):
    for i in range(0, 2 * 240):
        print(f"ITERATION # {i}")
        perform_good_input(predictor)
        perform_bad_input(predictor)
        
        print("> SLEEPING FOR 30 SECONDS")
        sleep(30)


# -

generate_sample_requests(predictor)
