# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (PyTorch 1.10 Python 3.8 CPU Optimized)
#     language: python
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:081325390199:image/pytorch-1.10-cpu-py38
# ---

# **Image**: PyTorch 1.10 Python 3.8 CPU Optimized Image

# !pip3 install transformers==4.4.2

from transformers import AutoModelForSequenceClassification

pretrained = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(pretrained)

model.save_pretrained(save_directory=".")

import tarfile
tar = tarfile.open("model.tar.gz", "w:gz")
tar.add("pytorch_model.bin")
tar.add("config.json")
tar.close()

# + language="bash"
#
# rm pytorch_model.bin
# rm config.json
# -

s3_bucket = "<INSERT BUCKET HERE>"
prefix = "chapter07"

# !aws s3 mb s3://{s3_bucket}

model_data = "s3://{}/{}/model/model.tar.gz".format(s3_bucket, prefix)

model_data

# !aws s3 cp model.tar.gz {model_data}

# %store model_data
# %store s3_bucket
# %store prefix
