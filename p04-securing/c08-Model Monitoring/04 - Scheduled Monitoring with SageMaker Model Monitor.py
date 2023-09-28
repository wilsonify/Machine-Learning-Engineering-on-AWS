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
# %store -r prefix

# %store -r ll_package_arn
# %store -r endpoint_name

# %store -r ll_package_arn

# +
import sagemaker
from sagemaker import get_execution_role
from sagemaker.predictor import Predictor

session = sagemaker.Session()
role = get_execution_role()

predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=session,
    role=role
)

# + language="bash"
#
# mkdir -p tmp
# wget -O tmp/baseline.csv https://bit.ly/3td5vjx
# -

base = f's3://{s3_bucket}/{prefix}'
baseline_source_uri = f'{base}/baseline.csv'
baseline_output_uri = f"{base}/baseline-output"

# !aws s3 cp tmp/baseline.csv {baseline_source_uri}

# +
from sagemaker.model_monitor import DefaultModelMonitor

monitor_dict = {
    'role': role,
    'instance_count': 1,
    'instance_type': 'ml.m5.large',
    'volume_size_in_gb': 10,
    'max_runtime_in_seconds': 1800,
}

default_monitor = DefaultModelMonitor(
    **monitor_dict
)

# +
# %%time

from sagemaker.model_monitor import dataset_format

dataset_format = dataset_format.DatasetFormat.csv(header=True)

baseline_dict = {
    'baseline_dataset': baseline_source_uri,
    'dataset_format': dataset_format,
    'output_s3_uri': baseline_output_uri,
    'wait': True
}

default_monitor.suggest_baseline(
    **baseline_dict
)

# +
import pandas as pd

def flatten(input_dict):
    df = pd.json_normalize(input_dict)
    return df.head()


# +
baseline_job = default_monitor.latest_baselining_job
stats = baseline_job.baseline_statistics()
schema_dict = stats.body_dict["features"]

flatten(schema_dict)

# +
constraints = baseline_job.suggested_constraints()
constraints_dict = constraints.body_dict["features"]

flatten(constraints_dict)
# -

constraints.body_dict['features'][1]['inferred_type'] = 'Integral'
constraints.body_dict['features'][2]['inferred_type'] = 'Integral'
constraints.save()

# +
from sagemaker.model_monitor import CronExpressionGenerator
from string import ascii_uppercase
import random


def generate_label():
    chars = random.choices(ascii_uppercase, k=5)
    output = 'monitor-' + ''.join(chars)
    return output


# -

s3_report_path = f'{base}/report'
baseline_statistics = default_monitor.baseline_statistics()
constraints = default_monitor.suggested_constraints()

cron_expression = CronExpressionGenerator.hourly()

# +
schedule_dict = {
    'monitor_schedule_name': generate_label(),
    'endpoint_input': predictor.endpoint,
    'output_s3_uri': s3_report_path,
    'statistics': baseline_statistics,
    'constraints': constraints,
    'schedule_cron_expression': cron_expression,
    'enable_cloudwatch_metrics': True
}

default_monitor.create_monitoring_schedule(
    **schedule_dict
)
# -

flatten(default_monitor.describe_schedule())

from time import sleep
sleep(300)

dm = default_monitor
monitoring_violations = dm.latest_monitoring_constraint_violations()
monitoring_statistics = dm.latest_monitoring_statistics()

# +
# %%time

from time import sleep


def get_violations():
    return dm.latest_monitoring_constraint_violations()


def loop_and_load_violations():
    for i in range(0, 2 * 120):
        print(f"ITERATION # {i}")
        print("> SLEEPING FOR 60 SECONDS")
        sleep(60)
        
        try:
            v = get_violations()
            violations = v
            
            if violations:
                return violations
        except:
            pass
    
    print("> DONE!")
    return None              
              
          
loop_and_load_violations()
# -

violations = dm.latest_monitoring_constraint_violations()
violations.__dict__

monitoring_statistics = dm.latest_monitoring_statistics()
monitoring_statistics.__dict__
