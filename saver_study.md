### 保存检查点(checkpoint)
为了得到可以用来后续恢复模型以进一步训练或评估的检查点文件（checkpoint file），我们实例化一个tf.train.Saver。
```
saver = tf.train.Saver()
```

在训练循环中，将定期调用saver.save()方法，向训练文件夹中写入包含了当前所有可训练变量值得检查点文件。
```
saver.save(sess, FLAGS.train_dir, global_step=step)
```

这样，我们以后就可以使用saver.restore()方法，重载模型的参数，继续训练。
```
saver.restore(sess, FLAGS.train_dir)
```