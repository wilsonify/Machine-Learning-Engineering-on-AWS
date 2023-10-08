from sagemaker import ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.network import NetworkConfig

image = ""
volume_kms_key = ""
output_kms_key = ""

estimator = Estimator(
    image=image,
    volume_kms_key=volume_kms_key,
    output_kms_key=output_kms_key
)

estimator.deploy(kms_key=volume_kms_key)

estimator = Estimator(image=image, encrypt_inter_container_traffic=True)

config = NetworkConfig(
    enable_network_isolation=True,
    encrypt_inter_container_traffic=True
)

processor = ScriptProcessor(network_config=config)

processor.run()

cmd = ["ssh", "<user>@<IP address of instance>", "-NL", "14344:localhost:8888"]
