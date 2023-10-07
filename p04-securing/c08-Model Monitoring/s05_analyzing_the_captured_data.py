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
# %store -r capture_upload_path

# +
# results = !aws s3 ls {capture_upload_path} --recursive
processed = []

for result in results:
    partial = result.split()[-1]
    path = f"s3://{s3_bucket}/{partial}"
    processed.append(path)
    
processed
# -

# !mkdir -p captured

for index, path in enumerate(processed):
    print(index, path)
    # !aws s3 cp {path} captured/{index}.jsonl

# +
import json

def load_json_file(path):
    output = []
    
    with open(path) as f:
        output = [json.loads(line) for line in f]
        
    return output


# +
all_json = []

for index, _ in enumerate(processed):
    print(f"INDEX: {index}")
    new_records = load_json_file(f"captured/{index}.jsonl")
    all_json = all_json + new_records
    
    
all_json
# -

# !pip3 install flatten-dict

from flatten_dict import flatten

first = flatten(all_json[0], reducer='dot')
first

# +
flattened_json = []

for entry in all_json:
    result = flatten(entry, reducer='dot')
    flattened_json.append(result)
    
flattened_json
# -

import pandas as pd
df = pd.DataFrame(flattened_json)
df

df[['x', 'y']] = df['captureData.endpointInput.data'].str.split(',', 1, expand=True)

df

df['predicted_label'] = df['captureData.endpointOutput.data'].str.strip()

df

clean_df = df[['predicted_label', 'x', 'y']]

clean_df.head()

clean_df = clean_df.astype({
    'predicted_label': 'int',
    'x': 'float',
    'y': 'float',
})

clean_df.head()
