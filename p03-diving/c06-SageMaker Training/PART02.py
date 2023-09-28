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

s3_bucket = "<INSERT S3 BUCKET HERE>"
prefix = "ch06"

# %store -r role
# %store -r region_name
# %store -r job_name
# %store -r image
# %store -r analytics_df

job_name

# +
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session()
previous = Estimator.attach(job_name)
# -

previous.logs()

model_data = previous.model_data
model_data

# +
import string 
import random

def generate_random_string():
    return ''.join(
        random.sample(
        string.ascii_uppercase,12)
    )


# -

base_job_name = generate_random_string()
base_job_name

checkpoint_folder="checkpoints"
checkpoint_s3_bucket="s3://{}/{}/{}".format(s3_bucket, base_job_name, checkpoint_folder)
checkpoint_local_path="/opt/ml/checkpoints"

# !rm -rf tmp2 && mkdir -p tmp2

# +
# %%time

# !wget -O tmp2/batch2.zip https://bit.ly/3KyonQE

# +
# %%time

# !cd tmp2 && unzip batch2.zip && rm batch2.zip

# +
import glob

training_samples = glob.glob(f"tmp2/train/*/*.png")

len(training_samples)


# -

# !aws s3 mb s3://{s3_bucket}

# +
# %%time

# !aws s3 rm s3://{s3_bucket} --recursive

# +
# %%time

# !aws s3 cp tmp2/.  s3://{s3_bucket}/{prefix}/ --recursive

# +
def map_path(source):
    return 's3://{}/{}/{}'.format(
        s3_bucket, 
        prefix, 
        source
    )

def map_input(source):
    path = map_path(source)
    
    return sagemaker.inputs.TrainingInput(
        path, 
        distribution='FullyReplicated', 
        content_type='application/x-image', 
        s3_data_type='S3Prefix'
    )


# -

data_channels = {}

channels = ["train", 
            "validation",
            "train_lst",
            "validation_lst"]

for channel in channels:
    data_channels[channel] = map_input(channel)

output_path = map_path("output")
output_path

estimator = sagemaker.estimator.Estimator(
    image,
    role, 
    instance_count=2, 
    instance_type='ml.p2.xlarge',
    output_path=output_path,
    sagemaker_session=session,
    enable_network_isolation=True,
    model_uri=model_data,
    use_spot_instances=True,
    max_run=1800,
    max_wait=3600,
    base_job_name=base_job_name,
    checkpoint_s3_uri=checkpoint_s3_bucket,
    checkpoint_local_path=checkpoint_local_path
)

hyperparameters = {
    'num_training_samples': len(training_samples),
    'num_layers': 18,
    'image_shape': "1,28,28",
    'num_classes': 10,
    'mini_batch_size': 100,
    'epochs': 3,
    'learning_rate': 0.01,
    'top_k': 5,
    'precision_dtype': 'float32'    
}

estimator.set_hyperparameters(**hyperparameters)

estimator.__dict__

# +
# %%time

estimator.fit(inputs=data_channels, logs=True)
# -

estimator.model_data

# !aws s3 ls {estimator.checkpoint_s3_uri} --recursive
