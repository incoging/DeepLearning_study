#coding:utf-8
#这个文档主要讲述梯度运算

import tensorflow as tf

w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])

# res = tf.matmul(w1, [[2],[1]])
res = tf.matmul(w1, [[2],[2]])

grads = tf.gradients(res,[w1])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print sess.run(res)
    print("-----------------------------")
    re = sess.run(grads)
    print(re)
#  [array([[2, 1]], dtype=int32)]