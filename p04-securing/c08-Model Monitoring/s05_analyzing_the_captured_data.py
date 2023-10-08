# name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:236514542706:image/datascience-1.0
import fnmatch
import json

import boto3
import pandas as pd
from flatten_dict import flatten

s3_bucket = "064592191516-ml-engineering"
prefix = "c08"
capture_upload_path = f"s3://{s3_bucket}/{prefix}/data-capture"


def glob_s3(glob_pattern):
    s3_client = boto3.client('s3')
    first_asterisk_index = glob_pattern.find('*')
    s3_prefix = glob_pattern[:first_asterisk_index] if first_asterisk_index >= 0 else glob_pattern
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    matched_objects = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        is_matched = fnmatch.fnmatch(key, glob_pattern)
        if is_matched:
            matched_objects.append(key)
    return matched_objects


results = glob_s3(capture_upload_path)
processed = []

for result in results:
    partial = result.split()[-1]
    path = f"s3://{s3_bucket}/{partial}"
    processed.append(path)

print(f"processed = {processed}")

# !mkdir -p captured

for index, path in enumerate(processed):
    print(index, path)
    s3_client = boto3.client('s3')
    s3_client.copy_object(
        Bucket=s3_bucket,
        CopySource={'Bucket': s3_bucket, 'Key': path},
        Key=f"captured/{index}.jsonl"
    )


def load_json_file(path):
    output = []
    with open(path) as f:
        for line in f:
            output += [json.loads(line)]

    return output


all_json = []

for index, _ in enumerate(processed):
    print(f"INDEX: {index}")
    new_records = load_json_file(f"captured/{index}.jsonl")
    all_json = all_json + new_records

print(f"all_json = {all_json}")

first = flatten(all_json[0], reducer='dot')
print(f"first = {first}")

flattened_json = []

for entry in all_json:
    result = flatten(entry, reducer='dot')
    flattened_json.append(result)

print(f"flattened_json = {flattened_json}")

df = pd.DataFrame(flattened_json)
print(f"df.head(5) = {df.head(5)}")

df[['x', 'y']] = df['captureData.endpointInput.data'].str.split(',', 1, expand=True)

print(f"df.head(5) = {df.head(5)}")

df['predicted_label'] = df['captureData.endpointOutput.data'].str.strip()

print(f"df.head(5) = {df.head(5)}")

clean_df = df[['predicted_label', 'x', 'y']]

print(f"clean_df.head(5) = {clean_df.head(5)}")

clean_df = clean_df.astype({
    'predicted_label': 'int',
    'x': 'float',
    'y': 'float',
})

print(f"clean_df.head(5) = {clean_df.head(5)}")
