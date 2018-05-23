#coding:utf-8
import tensorflow as tf

#less_equal(x, y, name=None) 输出真值情况,如果x小于y则为True,否则为False
x1 = tf.constant([1,2,3,4,6,8,9])
x2 = tf.constant([1,3,6,2,5,9,10])

with tf.Session() as sess:
    t = tf.less_equal(x1, x2)
    # print sess.run(t)    #[ True False  True False]


    #tf.where()
    #where(condition, x=None, y=None, name=None)
    '''
    如果x,y都为None,则此操作将返回“condition”的真实元素的坐标。坐标以二维张量返回，其中第一维（行）表示真元素的数量，第二维（列）表示真元素的坐标。
    ?????这句话还不太懂,但是对于一维的张量,tf.where函数返回里面为true的位置索引.
    '''
    wherev = tf.where(t)
    # print sess.run(wherev)

    a = tf.constant([1, 2, 3, 1, 2])
    b = tf.where(tf.equal(a, 1), x=a, y=tf.zeros_like(a))
    print sess.run(b)

    #
    # indices = tf.squeeze(wherev, 1)
    # print sess.run(indices)
    #
    #
    # value = tf.gather(x1, indices)
    # print sess.run(value)