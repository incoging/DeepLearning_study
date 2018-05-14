##### sparse_softmax_cross_entropy

用于求取交叉熵的loss损失
函数形式：
```
tf.losses.sparse_softmax_cross_entropy(
    labels, logits, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
```
实例：
```
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
```
参数：
* logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，
  单样本的话，大小就是num_classes
  
* labels：实际的标签，大小同上

函数执行过程：
1.第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，对于单样本而言，
输出就是一个num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率）


