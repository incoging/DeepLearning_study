#coding:utf-8

import tensorflow as tf

#reduce_mean(input_tensor,axis=None,keep_dims=False,name=None,reduction_indices=None)
'''
axis为None表示所有求均值,
为0表示按列求均值,即每一列求均值;
为1表示按行求均值,即每一行求均值.
keep_dims 保持维度,不太明白,也有一个例子
'''


t1 = tf.constant([[1,1], [2,2]],dtype=tf.float32)

with tf.Session() as sess:
    print sess.run(t1)
    '''
    output:
    [[ 1.  1.]
     [ 2.  2.]]
    '''

    #对整个t1求均值,得到所有元素的均值1.5
    r1 = tf.reduce_mean(t1)
    # print sess.run(r1)    #1.5

    #按列求均值,即每一列求均值
    r2 = tf.reduce_mean(t1, 0)
    # print sess.run(r2)   #[ 1.5  1.5]


    #按行求均值,即每一行求均值
    r3 = tf.reduce_mean(t1, 1)
    # print sess.run(r3)    #[ 1.  2.]

    r4 = tf.reduce_mean(t1, 0, keep_dims = True)
    print sess.run(r4)      #[[ 1.5  1.5]]   比着上面的多了一层中括号