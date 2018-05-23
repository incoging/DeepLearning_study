##### tf.concat

tf.concat(values, axis, name="concat"):
将向量按指定维度连接起来，其余维度不变
values: 为要连接的张量，
axis：指定从哪个维度进行连接
```
t1 = tf.constant([[1, 2, 3], [4, 5, 6]])
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])
# 从第0维度进行连接，即对此例为按行连接，即第一个矩阵的每一行，在后面拼上第二个矩阵的每一行
a = tf.concat([t1, t2], 0)
with tf.Session() as sess:
    print(sess.run(a))
```
输出结果为：
```
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
```
若从第二维连接：
```
# 从第1维度进行连接，即对此例为按列连接,即第一个矩阵的每一列，在后面拼上第二个矩阵的每一列
b = tf.concat([t1, t2], 1)
with tf.Session() as sess:
    print(sess.run(b))
```
输出结果为：
```
[[ 1  2  3  7  8  9]
 [ 4  5  6 10 11 12]]
```

>Notice: 如果是两个向量，它们是无法调用

```
t1=tf.constant([1,2,3])  
t2=tf.constant([4,5,6])  
#concated = tf.concat([t1,t2], 1)这样会报错  

```
因为t1,t2,它们对应的shape只有一个维度，当然不能在第二维上连了，
虽然实际中两个向量可以在行上连，但是放在程序里是会报错的
需要再加上下面的：
```
t1=tf.expand_dims(tf.constant([1,2,3]),1)  
t2=tf.expand_dims(tf.constant([4,5,6]),1)  
concated = tf.concat(1, [t1,t2])#这样就是正确的  
```