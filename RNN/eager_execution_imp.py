# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
x = [[2.]]
m = tf.matmul(x, x)