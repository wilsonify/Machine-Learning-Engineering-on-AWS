# name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:236514542706:image/datascience-1.0
import random
from string import ascii_uppercase
from time import sleep

import pandas as pd
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor import dataset_format
from sagemaker.predictor import Predictor


def generate_label():
    chars = random.choices(ascii_uppercase, k=5)
    output = 'monitor-' + ''.join(chars)
    return output


def flatten(input_dict):
    df = pd.json_normalize(input_dict)
    return df.head()


s3_bucket = ""
prefix = ""
ll_package_arn = ""
endpoint_name = ""

base = f's3://{s3_bucket}/{prefix}'
baseline_source_uri = f'{base}/baseline.csv'
baseline_output_uri = f"{base}/baseline-output"

session = sagemaker.Session()
role = get_execution_role()

predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=session,
    role=role
)

default_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size_in_gb=10,
    max_runtime_in_seconds=1800,
)

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

baseline_job = default_monitor.latest_baselining_job
stats = baseline_job.baseline_statistics()
schema_dict = stats.body_dict["features"]

flatten(schema_dict)

constraints = baseline_job.suggested_constraints()
constraints_dict = constraints.body_dict["features"]

flatten(constraints_dict)

constraints.body_dict['features'][1]['inferred_type'] = 'Integral'
constraints.body_dict['features'][2]['inferred_type'] = 'Integral'
constraints.save()

s3_report_path = f'{base}/report'
baseline_statistics = default_monitor.baseline_statistics()
constraints = default_monitor.suggested_constraints()

cron_expression = CronExpressionGenerator.hourly()

default_monitor.create_monitoring_schedule(
    monitor_schedule_name=generate_label(),
    endpoint_input=predictor.endpoint,
    output_s3_uri=s3_report_path,
    statistics=baseline_statistics,
    constraints=constraints,
    schedule_cron_expression=cron_expression,
    enable_cloudwatch_metrics=True
)

flatten(default_monitor.describe_schedule())

sleep(300)

dm = default_monitor
monitoring_violations = dm.latest_monitoring_constraint_violations()
monitoring_statistics1 = dm.latest_monitoring_statistics()
print(f"monitoring_statistics1 = {monitoring_statistics1}")


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

violations2 = dm.latest_monitoring_constraint_violations()
print(f"violations2.__dict__ = {violations2.__dict__}")

monitoring_statistics = dm.latest_monitoring_statistics()
print(f"monitoring_statistics.__dict__ = {monitoring_statistics.__dict__}")
