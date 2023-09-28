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

# + language="bash"
#
# mkdir -p tmp
# wget -O tmp/knn.model.tar.gz https://bit.ly/3yZ6qHE
# wget -O tmp/ll.model.tar.gz https://bit.ly/3ahj1fd
# -

s3_bucket = "<INSERT S3 BUCKET HERE>"
prefix = "chapter08"

# !aws s3 mb s3://{s3_bucket}

ll_model_data = f's3://{s3_bucket}/{prefix}/models/ll.model.tar.gz'
knn_model_data = f's3://{s3_bucket}/{prefix}/models/knn.model.tar.gz'

# !aws s3 cp tmp/ll.model.tar.gz {ll_model_data}
# !aws s3 cp tmp/knn.model.tar.gz {knn_model_data}

# +
from sagemaker.image_uris import retrieve

ll_image_uri = retrieve(
    "linear-learner", 
    region="us-west-2", 
    version="1"
)

knn_image_uri = retrieve(
    "knn", 
    region="us-west-2", 
    version="1"
)
# -

import boto3
client = boto3.client(service_name="sagemaker")

# +
import string 
import random

def generate_random_string():
    return ''.join(
        random.sample(
        string.ascii_uppercase,12)
    )


# +
group_id = generate_random_string()
model_package_group_name = f"group-{group_id}"
model_package_group_desc = f"Model package group {group_id}"

response = client.create_model_package_group(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageGroupDescription=model_package_group_desc
)

package_group_arn = response['ModelPackageGroupArn']
package_group_arn


# -

def prepare_inference_specs(image_uri, model_data):
    return {
        "Containers": [
            {
                "Image": image_uri,
                "ModelDataUrl": model_data
            }
        ],
        "SupportedContentTypes": [ 
            "text/csv" 
        ],
        "SupportedResponseMIMETypes": [ 
            "application/json" 
        ],
    }


def create_model_package(package_group_arn, inference_specs, client=client):
    input_dict = {
        "ModelPackageGroupName" : package_group_arn,
        "ModelPackageDescription" : f"Description for {package_group_arn}",
        "ModelApprovalStatus" : "Approved",
        "InferenceSpecification" : inference_specs
    }
    
    response = client.create_model_package(**input_dict)
    return response["ModelPackageArn"]


# +
knn_inference_specs = prepare_inference_specs(
    image_uri=knn_image_uri,
    model_data=knn_model_data
)

ll_inference_specs = prepare_inference_specs(
    image_uri=ll_image_uri,
    model_data=ll_model_data
)

# +
knn_package_arn = create_model_package(
    package_group_arn=package_group_arn,
    inference_specs=knn_inference_specs
)

ll_package_arn = create_model_package(
    package_group_arn=package_group_arn,
    inference_specs=ll_inference_specs
)
# -

# %store knn_package_arn
# %store ll_package_arn

# %store s3_bucket
# %store prefix
