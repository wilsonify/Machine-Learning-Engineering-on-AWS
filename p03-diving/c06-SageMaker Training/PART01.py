
# !rm -rf tmp && mkdir -p tmp

# !wget -O tmp/batch1.zip https://bit.ly/37zmQeb

# %%time

# !cd tmp && unzip batch1.zip && rm batch1.zip

# !ls -RF

# !pip3 install ipyplot

import glob

# +
import ipyplot

for i in range(0, 10):
    image_files = glob.glob(f"tmp/train/{i}/*.png")
    print(f'---{i}---')
    ipyplot.plot_images(image_files,
                        max_images=5,
                        img_width=128)
# -

s3_bucket = "<INSERT S3 BUCKET HERE>"
prefix = "ch06"

training_samples = glob.glob(f"tmp/train/*/*.png")
len(training_samples)

# !aws s3 mb s3://{s3_bucket}

# %%time
# !aws s3 rm s3://{s3_bucket} --recursive

# %%time
# !aws s3 cp tmp/.  s3://{s3_bucket}/{prefix}/ --recursive

# +
import sagemaker
import boto3

session = sagemaker.Session()
role = sagemaker.get_execution_role()
region_name = boto3.Session().region_name

# +
image = sagemaker.image_uris.retrieve(
    "image-classification",
    region_name,
    "1"
)

image


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
    enable_network_isolation=True
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

# %%time
estimator.fit(inputs=data_channels, logs=True)

estimator.model_data

model_data = estimator.model_data
job_name = estimator.latest_training_job.name

# %store model_data
# %store job_name
# %store role
# %store region_name
# %store image

endpoint = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)

# +
from sagemaker.serializers import IdentitySerializer

endpoint.serializer = IdentitySerializer(
    content_type="application/x-image"
)

# +
import json


def get_class_from_results(results):
    results_prob_list = json.loads(results)
    best_index = results_prob_list.index(
        max(results_prob_list)
    )

    return {
        0: "ZERO",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE",
        6: "SIX",
        7: "SEVEN",
        8: "EIGHT",
        9: "NINE"
    }[best_index]


# +
from IPython.display import Image, display


def predict(filename, endpoint=endpoint):
    byte_array_input = None

    with open(filename, 'rb') as image:
        f = image.read()
        byte_array_input = bytearray(f)

    display(Image(filename))

    results = endpoint.predict(byte_array_input)
    return get_class_from_results(results)


# -

# !ls tmp/test

# results = !ls -1
for filename in glob.glob("tmp/test"):
    print(predict(f"tmp/test/{filename}"))

endpoint.delete_endpoint()