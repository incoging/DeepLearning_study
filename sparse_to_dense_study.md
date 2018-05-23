##### sparse_to_dense

该函数就是建立一个output_shape的矩阵，并在sparse_indices所指定的索引位置放上sparse_values这样的值，
其他的值设定为default_value，返回这个矩阵。
```
tf.sparse_to_dense(sparse_indices,
                    output_shape,
                    sparse_values,
                    default_value=0,
                    validate_indices=True,
                    name=None):
```

* sparse_indices：稀疏矩阵中那些个别元素对应的索引值。
> sparse_indices是个数，那么它只能指定一维矩阵的某一个元素
  sparse_indices是个向量，那么它可以指定一维矩阵的多个元素
  sparse_indices是个矩阵，那么它可以指定二维矩阵的多个元素
  
* output_shape：输出的稀疏矩阵的shape

* sparse_values：索引指定位子元素的值
> sparse_values是个数：所有索引指定的位置都用这个数
  sparse_values是个向量：输出矩阵的某一行向量里某一行对应的数
  （所以这里向量的长度应该和输出矩阵的行数对应，不然报错）
  
* validate_indices: 若此项为true，则前面的索引必须按照字典顺序排列，且不能重复，
  若为false，则索引可以按照任意的方式排列着写
  > 举个例子：若为True，sparse_indices只能为：[[0,0],[1,3],[2, 3]]
    表示第一行的第一列，第二行第4列，第三行第4列。
    不能为：[[0,0],[3,3],[2, 3]]，即第一位的必须按照顺序。
    如果将validate_indices设为False，则这些索引可以打乱顺序随便写
    
```
concated = tf.constant([[0,0],[1,4],[2, 3], [3, 6], [4, 7], [5, 9]])
onehot_labels = tf.sparse_to_dense(concated, tf.stack([BATCHSIZE, 10]), 1.0, 0.0, validate_indices=True)
with tf.Session() as sess:
    result = sess.run(onehot_labels)
    print(result)
```
output:
```
[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
```

另附上一个用上tf.concat, tf.stack, tf.sparse_to_dense 把数字标签转化成onehot的例子：
```python
import tensorflow as tf
import numpy

BATCHSIZE = 6
label = tf.expand_dims(tf.constant([0, 2, 3, 6, 7, 9]), 1)
index = tf.expand_dims(tf.range(0, BATCHSIZE), 1)
# use a matrix
# concated = tf.concat([index, label], 1)
concated = tf.constant([[0,0],[1,4],[2, 3], [3, 6], [4, 7], [5, 9]])
onehot_labels = tf.sparse_to_dense(concated, tf.stack([BATCHSIZE, 10]), 1.0, 0.0, validate_indices=True)

# use a vector
concated2 = tf.constant([1, 3, 4])
# onehot_labels2 = tf.sparse_to_dense(concated2, tf.pack([BATCHSIZE,10]), 1.0, 0.0)#cant use ,because output_shape is not a vector
onehot_labels2 = tf.sparse_to_dense(concated2, tf.stack([10]), 1.0, 0.0)  # can use

# use a scalar
concated3 = tf.constant(5)
onehot_labels3 = tf.sparse_to_dense(concated3, tf.stack([10]), 1.0, 0.0)

with tf.Session() as sess:
    result1 = sess.run(onehot_labels)
    result2 = sess.run(onehot_labels2)
    result3 = sess.run(onehot_labels3)
    print("This is result1:")
    print(result1)
    print("This is result2:")
    print(result2)
    print("This is result3:")
    print(result3)

```