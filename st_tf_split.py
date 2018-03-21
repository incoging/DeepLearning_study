#coding:utf-8
#这个文档讲述split用法
import tensorflow as tf

'''
split(value, num_or_size_splits, axis=0, num=None, name="split")
value是要分割的张量,num_orsize_splits是要分割多少维,axis是对value的第几维进行分割
eg:
'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
tf.shape(split0) ==> [5, 4]
tf.shape(split1) ==> [5, 15]
tf.shape(split2) ==> [5, 11]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
tf.shape(split0) ==> [5, 10]
'''

t = tf.Variable(tf.constant(1,dtype=tf.int32,shape=[9,6,30,60]))
print(t)
t1 = tf.split(t,3,0)
print(t1)