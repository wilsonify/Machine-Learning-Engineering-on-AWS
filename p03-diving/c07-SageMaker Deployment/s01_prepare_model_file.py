# **Image**: PyTorch 1.10 Python 3.8 CPU Optimized Image pytorch-1.10-cpu-py38
import io
import tarfile

import boto3
from transformers import AutoModelForSequenceClassification

s3_bucket = "064592191516-ml-engineering"
prefix = "c07"
model_data = f"s3://{s3_bucket}/{prefix}/model/model.tar.gz"
pretrained = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(pretrained)

# Create a BytesIO buffer to hold the tar.gz file in-memory
tar_buffer = io.BytesIO()
# Create the tar.gz file in-memory
with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
    # Add model files to the tarball
    tar.addfile(tarfile.TarInfo(name="pytorch_model.bin"), model.save_to_buffer())
    tar.addfile(tarfile.TarInfo(name="config.json"), io.BytesIO(model.config.to_json_string().encode()))

# Upload the in-memory tar.gz file to S3
s3_client = boto3.client("s3")
tar_buffer.seek(0)  # Reset the buffer position to the beginning
s3_client.upload_fileobj(tar_buffer, s3_bucket, f"{prefix}/model/model.tar.gz")
# Close the tar buffer (optional)
tar_buffer.close()
print(f"Model has been uploaded to {model_data}")
