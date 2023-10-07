# name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:236514542706:image/datascience-1.0
from random import sample
from string import ascii_uppercase

import boto3
from sagemaker.image_uris import retrieve

s3_bucket = "064592191516-ml-engineering"
prefix = "c08"
ll_model_data = f's3://{s3_bucket}/{prefix}/models/ll.model.tar.gz'
knn_model_data = f's3://{s3_bucket}/{prefix}/models/knn.model.tar.gz'
ll_image_uri = retrieve(framework="linear-learner", region="us-east-1", version="1")
knn_image_uri = retrieve(framework="knn", region="us-east-1", version="1")
client = boto3.client(service_name="sagemaker")

group_id = ''.join(sample(ascii_uppercase, 12))
print(f"group_id = {group_id}")
response = client.create_model_package_group(
    ModelPackageGroupName=f"group-{group_id}",
    ModelPackageGroupDescription=f"Model package group {group_id}"
)

package_group_arn = response['ModelPackageGroupArn']
print(f"package_group_arn = {package_group_arn}")

knn_inference_specs = dict(
    Containers=[{"Image": knn_image_uri, "ModelDataUrl": knn_model_data}],
    SupportedContentTypes=["text/csv"],
    SupportedResponseMIMETypes=["application/json"],
)

ll_inference_specs = dict(
    Containers=[{"Image": ll_image_uri, "ModelDataUrl": ll_model_data}],
    SupportedContentTypes=["text/csv"],
    SupportedResponseMIMETypes=["application/json"],
)

knn_package_arn = client.create_model_package(
    ModelPackageGroupName=package_group_arn,
    ModelPackageDescription=f"Description for {package_group_arn}",
    ModelApprovalStatus="Approved",
    InferenceSpecification=knn_inference_specs
)

ll_package_arn = client.create_model_package(
    ModelPackageGroupName=package_group_arn,
    ModelPackageDescription=f"Description for {package_group_arn}",
    ModelApprovalStatus="Approved",
    InferenceSpecification=ll_inference_specs
)
