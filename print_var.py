# coding=utf-8
# 打印模型变量
import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = os.path.join("../wave.ckpt-done")
# 从checkpoint中读取数据
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# 得到变量的名字和形状
var_to_shape_map = reader.get_variable_to_shape_map()
# 打印变量的名字和形状
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
