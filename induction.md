##### 入门教程


##### tf.cast

```
tf.cast(correct_prediction, "float")
```
tf.cast(x, dtype) 将张量x转换成dype类型。

##### eval()
eval() 其实就是tf.Tensor的Session.run() 的另外一种写法。下面两个例子：
```
如果有一个tensor t,
t.eval() 等价于 tf.get_default_session().run(t)
```
另一个例子：
```
with tf.Session() as sess:
    # keep_prob,是用来控制dropout比例
    print(accuracy.eval({x:batch[0], y_: batch[1], keep_prob: 1.0}))
```
其效果和下面的代码是等价的：
```
with tf.Session() as sess:
  print(sess.run(accuracy, {x:batch[0], y_: batch[1], keep_prob: 1.0}))
```
> Notice: eval()只能用于tf.Tensor类对象，也就是有输出的Operation。对于没有输出的Operation, 可以用.run()或者Session.run()。
Session.run()没有这个限制。

##### tf.placeholder

```
x = tf.placeholder("float", [None, 784])
```

x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
（这里的None表示此张量的第一个维度可以是任何长度的。）

##### tf.Variable

```
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。
它们可以用于计算输入值，也可以在计算中被修改。

##### tf.matmul

```
y = tf.nn.softmax(tf.matmul(x,W) + b)
```
tf.matmul(​​x, W)表示x乘以W

##### tf.reduce_sum

```
cross_entropy = -tf.reduce_sum(y_#####tf.log(y))
```
tf.reduce_sum 计算张量的所有元素的总和

##### tf.train.GradientDescentOptimizer

```
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```
TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。

##### tf.initialize_all_variables

```
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
```
初始化我们建立的变量

##### tf.argmax

```
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```
tf.argmax 返回某个tensor对象在某一维上其数据最大值所在的索引，tf.argmax(y,1)表示在y中
等于1这个维度的最大值的索引

##### tf.reduce_mean(input_tensor, axis)

```
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```
获得输入张量在axis维度上的平均值，tf.reduce_mean(aa,0),获取张量aa上第一维度的平均值，
（即，如果aa是二维的话，计算出每一行的平均值）


##### tf.InteractiveSession

```
import tensorflow as tf
sess = tf.InteractiveSession()
```
Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。
一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。
这里，我们使用更加方便的InteractiveSession类。如果你没有使用InteractiveSession，
那么你需要在启动session之前构建整个计算图，然后启动该计算图。

##### tf.truncated_normal

```
var1 =tf.Variable(tf.truncated_normal(shape, stddev=0.1)) 
```
这个函数产生正态分布，均值和标准差自己设定。
truncated_normal(shape,mean=0.0,stddev=1.0,dtype=dtypes.float32,seed=None,name=None)
shape表示生成张量的维度，mean是均值，stddev是标准差。
这是一个截断的产生正态分布的函数，就是说产生正态分布的值如果与均值的差值大于两倍的标准差，那就重新生成。

##### tf.constant

```
var1 =tf.Variable(tf.constant(0.1, shape=shape))
```
返回一个常量tensor
constant(value, dtype=None, shape=None, name="Const", verify_shape=False)

##### tf.nn.conv2d

```
tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
```

W是卷积的参数，eg:[5, 5, 1, 32] 前面两个数字代表卷积核的尺寸;第三个数字代表有多少个channer。
因为我们只有灰度单色，所以是1，如果是彩色的RGB图片，这里应该是3
最后一个数字代表卷积核的数量，也就是这个 卷积层会提取多少类的特征。

##### tf.nn.dropout
```
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```
为了减少过拟合，我们在输出层之前加入dropout。通过一个placeholder传入keep_prob比率来控制，
在训练时随机的丢弃一部分节点数据来减轻过拟合。这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。

##### tf.get_variable()和tf.Variable()
1.tf.get_variable()用于创建变量或获取变量;
2.tf.Variable()用于创建变量。
```
tf.get_variable(name,  shape, initializer)
```
初始化方式：
* tf.constant_initializer：常量初始化函数
* tf.random_normal_initializer：正态分布
* tf.truncated_normal_initializer：截取的正态分布
* tf.random_uniform_initializer：均匀分布
* tf.zeros_initializer：全部是0
* tf.ones_initializer：全是1
* tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值

例子：
```
v = tf.get_variable("v",shape=[1],initializer.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0,shape=[1]),name="v")
```
区别：
>1.两函数指定变量名称的参数不同，对于tf.Variable函数，变量名称是一个可选的参数，通过name="v"的形式给出
2.而tf.get_variable函数，变量名称是一个必填的参数，它会根据变量名称去创建或者获取变量。


##### tf.nn.in_top_k

tf.nn.in_top_k组要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量，
tf.nn.in_top_k(prediction, target, K):prediction就是表示你预测的结果，大小就是预测样本的数量乘以输出的维度，
类型是tf.float32等。target就是实际样本类别的标签，大小就是样本数量的个数。
K表示每个样本的预测结果的前K个最大的数的索引是否和target中的值匹配。一般都是取1。
```
import tensorflow as tf;

A = [[0.8,0.6,0.3], [0.1,0.6,0.4]]
B = [1, 1]
out = tf.nn.in_top_k(A, B, 1)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(out)

output:
[False  True]
```
> 解析：<br/>
因为A张量里面的第一个元素的最大值的标签是0，第二个元素的最大值的标签是1.。但是实际的确是1和1.所以输出就是False 和True。
如果把K改成2，那么第一个元素的前面2个最大的元素的位置是0，1，第二个的就是1，2。
实际结果是1和1。包含在里面，所以输出结果就是True 和True.如果K的值大于张量A的列，那就表示输出结果都是true

##### tf.nn.max_pool

```
tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```
执行最大化池化操作.   
x即要池化的tensor，格式若为NHWC,则表示[batch, height, width, channels]   
ksize,池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，
所以这两个维度设为了1   
窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]

##### tf.nn.softmax_cross_entropy_with_logits
下面的例子展示它与tf.nn.sparse_softmax_cross_entry_with_logits的区别
```
# 假设词汇表的大小为3， 语料包含两个单词"2 0"
word_labels = tf.constant([2, 0])

# 假设模型对两个单词预测时，产生的logit分别是[2.0, -1.0, 3.0]和[1.0, 0.0, -0.5]
# 注意这里的logit不是概率，因此它们不是0.0~1.0范围之间的数字，如果需要计算概率，
# 则需要调用prob = tf.nn.softmax(logits).但这里计算交叉熵的函数直接输入logits即可。
predict_logits = tf.constant([[2.0, -1.0, 3.0], [1.0, 0.0, -0.5]])

# 使用sparse_softmax_cross_entropy_with_logits计算交叉熵
loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels, logits=predict_logits)

# 运行程序，计算loss的结果是[0.32656264 0.4643688 ], 这对应两个预测的perplexity损失


# softmax_cross_entropy_with_logits与上面相似，但是需要将预测目标以概率分布的形式给出
# 即这种形式需要使用one_hot
word_prob_distribution = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_distribution, logits=predict_logits)

# 由于softmax_cross_entropy_with_logits允许提供一个概率分布，因此在使用时有更大的自由度。
# 举个例子，一种叫label smoothing的技巧是将正确数据的概率设为一个比1.0略小的值，
# 将错误数据的概率设为比0.0略大的值，这样可以避免模型与数据过拟合，在某些时候可以提高训练效果
word_prob_smooth = tf.constant([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01]])
loss3 = tf.nn.softmax_cross_entropy_with_logits(labels=word_prob_smooth, logits=predict_logits)

one_hot = tf.one_hot(word_labels,depth=3)
loss4 = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=predict_logits)
with tf.Session() as sess:
    print((sess.run(loss1)))
    print((sess.run(loss2)))
    print((sess.run(loss3)))
    print((sess.run(loss4)))
```



##### tf.train.Saver

保存结果的名字格式是：
> saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
...
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'

```
# Create a saver.
saver = tf.train.Saver(...variables...)
# Launch the graph and train, saving the model every 1,000 steps.
sess = tf.Session()
for step in range(1000000):   #python3没有xrange,在对大序列进行迭代的时候,因为xrange特性,会比较节约内存
    sess.run(...training_op...)
    if step % 1000 == 0:
        # Append the step number to the checkpoint name:
        saver.save(sess, 'my-model', global_step=step)
```

##### tf.reshape
```
x_image = tf.reshape(x, [-1,28,28,1])
```
前面的-1代表样本数量不固定，最后的1代表颜色通道数量

##### tf.reset_default_graph()

清除当前正在运行的图,避免变量重复

##### tf.squeeze()

给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。
 如果不想删除所有尺寸1尺寸，可以通过指定squeeze_dims来删除特定尺寸1尺寸。squeeze_dims维度从0开始。
```
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
Or, to remove specific size 1 dimensions:

# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
```

##### tf.train.AdamOptimizer
```
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```
ADAM优化器  adaptive moment estimation，自适应矩估计

##### tf.name_scope

```
with tf.name_scope('hidden1') as scope:
```
创建于该作用域之下的所有元素都将带有其前缀,例如，当一些层是在hidden1作用域下生成时，赋予权重变量的独特名称将会是"hidden1/weights"。

##### logits

未归一化的概率，一般也就是softmax层的输入

##### var.shape.as_list()

以列表的形式显示变量var的形状


##### tf.python.platform.gfile

随着 tensorflow 版本升级，gfile 已进一步提升至 tf.gfile 下
常用API：
```
tensorflow.python.platform.gfile.Exists
tensorflow.python.platform.gfile.Glob
tensorflow.python.platform.gfile.IsDirectory
tensorflow.python.platform.gfile.FastGFile
tensorflow.python.platform.gfile.DeleteRecursively
tensorflow.python.platform.gfile.Open
tensorflow.python.platform.gfile.ListDirectory
tensorflow.python.platform.gfile.MakeDirs
tensorflow.python.platform.gfile.GFile
tensorflow.python.platform.gfile.Open.write
tensorflow.python.platform.gfile.Stat.length
tensorflow.python.platform.gfile.MkDir
tensorflow.python.platform.gfile.Walk
tensorflow.python.platform.gfile.FastGFile.read
```

