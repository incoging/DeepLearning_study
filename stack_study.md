##### tf.stack

tf.stack(values, axis=0, name="stack"):
将一组R维张量变为R+1维张量。
注意与tf.concat的区别，concat是按照要连接的矩阵里面指定维度连接起来，
而stack是将这一组张量的某一个维度直接放到一起。看下面的例子
```
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x,y,z],axis=1) ==> [[1, 2, 3], [4, 5, 6]]
```
此时x在计算机中的shape为[2]
而tf.concat需要先把一维的向量添加成2维的，才能在第二维度上连接，所以：
```
x1 = tf.expand_dims(x,0)
y1 = tf.expand_dims(y,0)
z1 = tf.expand_dims(z,0)
# 此时x1为1x2的矩阵,可以按照第二维来进行连接
bb = tf.concat([x1,y1,z1],1)

with tf.Session() as sess:
    # 打印aa的内容
    print(sess.run(aa))
    # 打印aa的形状
    print("aa shape is:", aa.shape.as_list())
    print("---------------------------")
    # 打印bb的内容
    print(sess.run(bb))
    # 打印bb的形状
    print("bb shape is:", bb.shape.as_list())
```
output:
```
[[1 2 3]
 [4 5 6]]
aa shape is: [2, 3]
---------------------------
[[1 4 2 5 3 6]]
bb shape is: [1, 6]
```