# Tensorflow and Python
This project contains code that checks your systems
access to GPU.  Machine Learning frameworks like Tensorflow
and keras benefit from GPU performance.

# Getting Started
First, Install the python packages in the requirements.txt file.
```bash
pip install -r requirements.txt
```
This will install Tensforflow and Keras.

Next, open the main.py file and look at the imports.

```python
from tensorflow.python.client import device_lib
import tensorflow as tf
```
In order to check the device we use the device_lib.

```python
local_device = device_lib.list_local_devices()
```
The local_device variable will have the local device(s).
The following code get the number of GPUs and prints to the console.
```python
gpu_info["count"] = len([d for d in local_device
                         if d.device_type == "GPU"])

print("count", gpu_info)
```
The rest of the code compiles, trains and evaluates a ml model.

