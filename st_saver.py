#coding=utf-8
#tf.train.Saver()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#创建一个生成模型
tf.reset_default_graph()  #清除当前正在运行的图,避免变量重复

X = tf.placeholder("float")
Y = tf.placeholder("float")

h_est = tf.Variable(0.0, name="hor_estimate")
v_est = tf.Variable(0.0, name="ver_estimate")

y_est = tf.square(X - h_est) + v_est

#定义一个损失函数
cost = (tf.pow(Y - y_est, 2))

trainop = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

h = 1
v = -2

x_train = np.linspace(-2, 4, 201)
noise = np.random.randn(*x_train.shape) * 0.4
y_train = (x_train -h) ** 2 + v + noise

plt.rcParams['figure.figsize'] = (10, 6)
plt.scatter(x_train, y_train)
plt.xlabel('x_train')
plt.ylabel('y_train')





# 创建一个Saver对象
saver = tf.train.Saver()
init = tf.global_variables_initializer()

def train_graph():
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            for(x, y) in zip(x_train, y_train):
                sess.run(trainop, feed_dict={X: x, Y: y})
            saver.save(sess, "model_iter", global_step=i)
