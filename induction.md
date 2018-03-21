##### 入门教程

* tf.placeholder

```
x = tf.placeholder("float", [None, 784])
```

x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
（这里的None表示此张量的第一个维度可以是任何长度的。）

* tf.Variable

```
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。
它们可以用于计算输入值，也可以在计算中被修改。

* tf.matmul

```
y = tf.nn.softmax(tf.matmul(x,W) + b)
```
tf.matmul(​​x, W)表示x乘以W

* tf.reduce_sum

```
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
```
tf.reduce_sum 计算张量的所有元素的总和

* tf.train.GradientDescentOptimizer

```
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```
TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。

* tf.initialize_all_variables

```
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
```
初始化我们建立的变量

* tf.argmax

```
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```
tf.argmax 返回某个tensor对象在某一维上其数据最大值所在的索引，tf.argmax(y,1)表示在y中
等于1这个最大值的索引

* tf.reduce_mean(input_tensor, axis)

```
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```
获得输入张量在axis维度上的平均值，tf.reduce_mean(aa,0),获取张量aa上第一维度的平均值，
（即，如果aa是二维的话，计算出每一行的平均值）

* tf.cast

```
tf.cast(correct_prediction, "float")
```
tf.cast(x, dtype) 将张量x转换成dype类型。

* tf.InteractiveSession

```
import tensorflow as tf
sess = tf.InteractiveSession()
```
Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。
一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。
这里，我们使用更加方便的InteractiveSession类。如果你没有使用InteractiveSession，
那么你需要在启动session之前构建整个计算图，然后启动该计算图。

* tf.truncated_normal

```
var1 =tf.Variable(tf.truncated_normal(shape, stddev=0.1)) 
```
这个函数产生正太分布，均值和标准差自己设定。
truncated_normal(shape,mean=0.0,stddev=1.0,dtype=dtypes.float32,seed=None,name=None)
shape表示生成张量的维度，mean是均值，stddev是标准差。
这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。

* tf.constant

```
var1 =tf.Variable(tf.constant(0.1, shape=shape))
```
返回一个常量tensor
constant(value, dtype=None, shape=None, name="Const", verify_shape=False)

* tf.nn.conv2d

```
tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
```

W是卷积的参数，eg:[5, 5, 1, 32] 前面两个数字代表卷积核的尺寸;第三个数字代表有多少个channer。
因为我们只有灰度单色，所以是1，如果是彩色的RGB图片，这里应该是3
最后一个数字代表卷积核的数量，也就是这个 卷积层会提取多少类的特征。

* tf.nn.max_pool

```
tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```
执行最大化池化操作.   
x即要池化的tensor，格式若为NHWC,则表示[batch, height, width, channels]   
ksize,池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，
所以这两个维度设为了1   
窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]



* tf.train.Saver

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

* tf.reshape
```
x_image = tf.reshape(x, [-1,28,28,1])
```
前面的-1代表样本数量不固定，最后的1代表颜色通道数量

* tf.nn.dropout
```
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

为了减少过拟合，我们在输出层之前加入dropout。通过一个placeholder传入keep_prob比率来控制，
在训练时随机的丢弃一部分节点数据来减轻过拟合。这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。

* tf.train.AdamOptimizer
```
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```
ADAM优化器  adaptive moment estimation，自适应矩估计
 
* eval()
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

* tf.name_scope

```
with tf.name_scope('hidden1') as scope:
```
创建于该作用域之下的所有元素都将带有其前缀,例如，当一些层是在hidden1作用域下生成时，赋予权重变量的独特名称将会是"hidden1/weights"。

* logits

未归一化的概率，一般也就是softmax层的输入

* var.shape.as_list()

以列表的形式显示变量var的形状
