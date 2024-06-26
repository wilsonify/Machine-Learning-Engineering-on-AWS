# name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-1:236514542706:image/datascience-1.0
import sagemaker
from sagemaker import get_execution_role
from sagemaker.predictor import Predictor

endpoint_name = ""

session = sagemaker.Session()
role = get_execution_role()

predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=session,
    role=role
)

monitors = predictor.list_monitors()

for monitor in monitors:
    print(monitor.__dict__)

for monitor in monitors:
    monitor.delete_monitoring_schedule()

predictor.delete_endpoint()
