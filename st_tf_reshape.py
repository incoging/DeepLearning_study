#coding:utf-8


import tensorflow as tf

# reshape(tensor, shape, name=None):
# 其中如果shape[]中有-1,则-1会根据shape中的其他数字进行适当的改变.
# 若只有-1,则相当于铺开t为一维的
t = tf.Variable(tf.random_normal([3, 2, 3],dtype=tf.float32),name="t")

t1 = tf.reshape(t, [-1, 6])
print(t1)   #Tensor("Reshape:0", shape=(3, 6), dtype=float32)

t2 = tf.reshape(t, [-1, 9])
print(t2)   #Tensor("Reshape_1:0", shape=(2, 9), dtype=float32)

t3 = tf.reshape(t, [-1])
print(t3)   #Tensor("Reshape_2:0", shape=(18,), dtype=float32)