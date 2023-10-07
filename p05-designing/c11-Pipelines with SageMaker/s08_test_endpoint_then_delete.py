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

s3_bucket = '<INSERT S3 BUCKET HERE>'

prefix = 'pipeline'

endpoint_name = "AutoGluonEndpoint"

# +
from sagemaker.predictor import Predictor

import sagemaker
# -

session = sagemaker.Session()

# !mkdir -p tmp

test_data_path = f"s3://{s3_bucket}/{prefix}/output/test/data.csv"

# !aws s3 cp {test_data_path} tmp/test_data.csv

# +
import pandas as pd

test_df = pd.read_csv("tmp/test_data.csv", header=None)
test_df
# -

test_df.rename(
    columns={ 
        test_df.columns[0]: "is_cancelled" 
    }, 
    inplace = True
)

test_df

predictor = Predictor(endpoint_name, session)

# +
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()

# +
import csv

def get_test_payload(index, test_df=test_df):
    test_data = test_df.drop(['is_cancelled'], axis=1)
    target_record = test_data.iloc[index]
    predictor_values = target_record.to_csv(
        header=None,
        index=False,
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC
    ).split()
    csv_string = ','.join(predictor_values)
    return csv_string


# -

def get_test_actual_result(index, test_df=test_df):
    result = test_df.iloc[index]['is_cancelled']
    return result


def predict(index, predictor=predictor):
    payload = get_test_payload(index)
    prediction = predictor.predict([payload])
    print(prediction)
    [[prob_0, prob_1]] = prediction['probabilities']
    
    if prob_0 > prob_1:
        return 0
    else:
        return 1


predict(5)

# +
from time import sleep

actual_list = []
predicted_list = []

for i in range(0, 100):
    actual = get_test_actual_result(i)
    predicted = predict(i)
    print(f"[iteration # {i}]")
    print(f"actual = {actual}; predicted = {predicted}")
    
    actual_list.append(actual)
    predicted_list.append(predicted)

    sleep(0.05)

# +
from sklearn.metrics import classification_report

target_names = ['not cancelled', 'cancelled']
print(classification_report(actual_list, predicted_list, target_names=target_names))

# +
from sklearn.metrics import accuracy_score

accuracy_score(actual_list, predicted_list)
# -

predictor.delete_endpoint()
