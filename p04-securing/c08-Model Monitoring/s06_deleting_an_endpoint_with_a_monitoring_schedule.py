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

# %store -r endpoint_name

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

# +
monitors = predictor.list_monitors()

for monitor in monitors:
    print(monitor.__dict__)
# -

for monitor in monitors:
    monitor.delete_monitoring_schedule()

predictor.delete_endpoint()
