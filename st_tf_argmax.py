#coding:utf-8
'''
这个文档主要讲述tf.argmax, 它是返回一个tensor对应与axis维度的最大值的下标.
如果tensor是一个向量,那么axis是0,即返回这个向量最大值的下标,即一个数字,
如果tensor是一个2x2的矩阵,如果axis是0,那么求出每一列中最大值的下表.
如果tensor是一个2x2的矩阵,如果axis是1,那么求出每一行中最大值的下表.
依次类推
对于高维的tensor来说,输入的Tensor的Rank是未知的，所以不能说按行和列计算最大值。是取最大值所在的索引。
'''

import tensorflow as tf

a = tf.get_variable(name='a',
                    shape=[3, 4],
                    dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
b = tf.argmax(input=a, axis=0)
c = tf.argmax(input=a, dimension=1)  # 此处用dimesion或用axis是一样的
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print(sess.run(a))
# [[ 0.04261756 -0.34297419 -0.87816691 -0.15430689]
# [ 0.18663144  0.86972666 -0.06103253  0.38307118]
# [ 0.84588599 -0.45432305 -0.39736366  0.38526249]]
print(sess.run(b))
# [2 1 1 2]
print(sess.run(c))
# [0 1 0]