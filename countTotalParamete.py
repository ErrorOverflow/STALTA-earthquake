import tensorflow as tf
import os
import numpy as np

#from tensorflow ckpt bakup files
model_dir = "/home/wangmulan/Code/PhaseNet-master/model/190703-214543"
checkpoint_path = os.path.join(model_dir, "model_95.ckpt")
reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
total_parameters = 0
for key in var_to_shape_map:  # list the keys of the model
    # print(key)
    # print(reader.get_tensor(key))
    shape = np.shape(reader.get_tensor(key))  # get the shape of the tensor in the model
    shape = list(shape)
    # print(shape)
    # print(len(shape))
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim
    # print(variable_parameters)
    total_parameters += variable_parameters

print(total_parameters)
