##### tf.expand_dims
```
expand_dims(input, axis=None, name=None, dim=None):
```
它给定一个张量input,给它增加一个维度,
例如:
原来t的shape是[2,3]如果
expand_dims(t, axis=0)则执行后的结果是:
shape为[1,2,3]
axis是决定在这个张量中第几个索引处插入维度,若执行
expand_dims(t, axis=1),则结果是:
shape为[2,1,3]

接下来学习python中多维向量的写法：
一个2x3的矩阵分别转化为以下两个维度：
[[1,1,1]
 [1,1,1]]

shape为[1,2,3]的：
```
1.先写最后一维[1,1,1]
2.再写倒数第二维[[1,1,1]
               [1,1,1]]
3.再写最外层一维[[[1,1,1]
                [1,1,1]]]
```
shape为[2,1,3]的：
```
1.先写最后一维[1,1,1]
2.再写倒数第二维[[1,1,1]] 是一个1x3的
3.再写最外层的一维[[[1,1,1]]
                [[1,1,1]]]
```

* expand_dims举例
```python
import tensorflow as tf

# 建立一个2x3的全1矩阵
t = tf.constant(1,tf.float32,shape=[2,3],name="t")
print(t)   #<tf.Variable 't:0' shape=(2, 3) dtype=float32_ref>
# 输出t的内容
with tf.Session() as sess:
    print(sess.run(t))
"""
[[1. 1. 1.]
 [1. 1. 1.]]
"""
print("=========================")


# 第0维上加了一个维度，并显示结果
t1= tf.expand_dims(t, axis=0)
print(t1)     #Tensor("ExpandDims:0", shape=(1, 2, 3), dtype=float32)
with tf.Session() as sess:
    print(sess.run(t1))
"""
[[[1. 1. 1.]
  [1. 1. 1.]]]
"""
print("=========================")


# 在第1维上加了一个维度，并显示结果
t2 = tf.expand_dims(t, axis=1)
print(t2)     #Tensor("ExpandDims_1:0", shape=(2, 1, 3), dtype=float32)
with tf.Session() as sess:
    print(sess.run(t2))
"""
[[[1. 1. 1.]]

 [[1. 1. 1.]]]
"""
```