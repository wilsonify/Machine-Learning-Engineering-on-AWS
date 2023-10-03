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

# !wget -O processing.py https://bit.ly/3QiGDQO

# !mkdir -p tmp

# !wget -O tmp/bookings.all.csv https://bit.ly/3BUcMK4

s3_bucket = '<INSERT S3 BUCKET NAME HERE>'
prefix = 'pipeline'

# !aws s3 mb s3://{s3_bucket}

source_path = f's3://{s3_bucket}/{prefix}' + \
              '/source/dataset.all.csv'

# !aws s3 cp tmp/bookings.all.csv {source_path}

# +
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput
)
from sagemaker.sklearn.processing import (
    SKLearnProcessor
)
from sagemaker.workflow.parameters import (
    ParameterString
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep
)

# -

role = get_execution_role()

session = sagemaker.Session()

input_data = ParameterString(
    name="RawData",
    default_value=source_path,
)

# +
input_raw = ProcessingInput(
    source=input_data,
    destination='/opt/ml/processing/input/'
)

output_split = ProcessingOutput(
    output_name="split",
    source='/opt/ml/processing/output/',
    destination=f's3://{s3_bucket}/{prefix}/output/'
)

# +
processor = SKLearnProcessor(
    framework_version='0.20.0',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large'
)

step_process = ProcessingStep(
    name="PrepareData",
    processor=processor,
    inputs=[input_raw],
    outputs=[output_split],
    code="processing.py",
)
# -

model_path = f"s3://{s3_bucket}/{prefix}/model/"

model_id = "autogluon-classification-ensemble"

region_name = "us-west-2"

# +
from sagemaker import image_uris

train_image_uri = image_uris.retrieve(
    region=region_name,
    framework=None,
    model_id=model_id,
    model_version="*",
    image_scope="training",
    instance_type="ml.m5.xlarge",
)

# +
from sagemaker import script_uris

train_source_uri = script_uris.retrieve(
    model_id=model_id,
    model_version="*",
    script_scope="training"
)
# -

# !aws s3 cp {train_source_uri} tmp/sourcedir.tar.gz

# +
from sagemaker import model_uris

train_model_uri = model_uris.retrieve(
    model_id=model_id,
    model_version="*",
    model_scope="training"
)
# -

# !aws s3 cp {train_model_uri} tmp/ensemble.tar.gz

# +
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri=train_image_uri,
    source_dir=train_source_uri,
    model_uri=train_model_uri,
    entry_point="transfer_learning.py",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    max_run=900,
    output_path=model_path,
    session=session,
    role=role
)

# +
from sagemaker.hyperparameters import retrieve_default

hyperparameters = retrieve_default(
    model_id=model_id,
    model_version="*"
)
hyperparameters["verbosity"] = "3"
estimator.set_hyperparameters(**hyperparameters)

# +
s3_data = step_process.properties.ProcessingOutputConfig.Outputs["split"].S3Output.S3Uri

step_train = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "training": TrainingInput(
            s3_data=s3_data,
        )
    },
)

# +
s3_data = step_process.properties.ProcessingOutputConfig.Outputs["split"].S3Output.S3Uri

step_train = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "training": TrainingInput(
            s3_data=s3_data,
        )
    },
)

# +
deploy_image_uri = image_uris.retrieve(
    region=region_name,
    framework=None,
    image_scope="inference",
    model_id=model_id,
    model_version="*",
    instance_type="ml.m5.xlarge",
)

deploy_source_uri = script_uris.retrieve(
    model_id=model_id,
    model_version="*",
    script_scope="inference"
)
# -

# !aws s3 cp {deploy_source_uri} tmp/sourcedir.tar.gz

# +
updated_source_uri = f's3://{s3_bucket}/{prefix}/sourcedir/sourcedir.tar.gz'

# !aws s3 cp tmp/sourcedir.tar.gz {updated_source_uri}

# +
import uuid


def random_string():
    return uuid.uuid4().hex.upper()[0:6]


# +
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import \
    PipelineSession

pipeline_session = PipelineSession()

model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts
model = Model(image_uri=deploy_image_uri,
              source_dir=updated_source_uri,
              model_data=model_data,
              role=role,
              entry_point="inference.py",
              sagemaker_session=pipeline_session,
              name=random_string())

# +
from sagemaker.workflow.model_step import ModelStep

model_package_group_name = "AutoGluonModelGroup"

register_args = model.register(
    content_types=["text/csv"],
    response_types=["application/json"],
    inference_instances=["ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    approval_status="Approved",
)

step_model_create = ModelStep(
    name="CreateModel",
    step_args=register_args
)

# +
pipeline_name = f"PARTIAL-PIPELINE"

partial_pipeline = Pipeline(
    name=pipeline_name,
    parameters=[ input_data ],
    steps=[ step_process, step_train, step_model_create, ],
)
# -

partial_pipeline.upsert(role_arn=role)

execution = partial_pipeline.start()
execution.describe()

execution.wait()
