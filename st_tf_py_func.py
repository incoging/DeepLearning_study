#coding:utf-8
#这个文档主要讲述tf.py_func,但是还顺便介绍palceholder


import numpy as np
import tensorflow as tf

#tf.py_func(func, inp, Tout, stateful=True, name=None)
#func：为一个python函数
#inp：为输入函数的参数，Tensor列表
#Tout： 指定func返回的输出的数据类型，是一个列表
def my_func(x):
  # x will be a numpy array with the contents of the placeholder below
  return np.sinh(x)
inp = tf.placeholder(tf.float32, [3, 4])

#placeholder
'''
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)
with tf.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.
    
    rand_array = np.random.rand(1024, 1024)           #产生shape为(1024, 1024)的矩阵,值为随机的[0,1]之间
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
'''


y = tf.py_func(my_func, [inp], [tf.float32])
with tf.Session() as sess:
    rand_arr = np.random.rand(3, 4)
    print rand_arr
    print ("----------------------------------------------")
    print sess.run(y,feed_dict={inp : rand_arr})