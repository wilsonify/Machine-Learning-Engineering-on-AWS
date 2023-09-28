# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import tensorflow as tf
model = tf.keras.models.load_model('model')
model.summary()

# +
import numpy as np

def load_data(training_data_location):
    result = np.loadtxt(open(training_data_location, "rb"), delimiter=",")
    
    y = result[:, 0]
    x = result[:, 1]
    
    return (x, y)


# -

x, y = load_data("data/test_data.csv")

x[0:5]

predictions = model.predict(x[0:5])
predictions

results = model.evaluate(x, y, batch_size=128)
results
