## slim模块
1.简介
2.创建变量


### 简介
TF-Slim是tensorflow中定义、训练和评估复杂模型的轻量级库。
```
import tensorflow.contrib.slim as slim
```

### 创建变量
创建一个权值变量，并且用truncated_normal初始化，用L2损失正则化，放置于CPU中，
我们只需要定义如下：
```
weights = slim.variable("weights",
                        shape=[10, 10, 3, 3],
                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                        regularizer=slim.l2_regularizer(0.05),
                        device="/CUP:0")
```
