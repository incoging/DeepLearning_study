## saver 学习
### 保存的文件
```
saver.save(sess, "/path/model/model.ckpt")
```
运行上面的代码会生成3个文件：<br>
1.model.ckpt.meta  保存了当前计算图的结构<br>
2.model.ckpt.index 保存当前的参数名<br>
3.model.ckpt.data-00000-of-00001  保存了每一个变量的取值<br>
最后有一个checkpoint文件，这个文件保存了一个目录下所有的模型文件列表。<br>


### 保存检查点(checkpoint)
为了得到可以用来后续恢复模型以进一步训练或评估的检查点文件（checkpoint file），我们实例化一个tf.train.Saver。
```
saver = tf.train.Saver()
```

在训练循环中，将定期调用saver.save()方法，向训练文件夹中写入包含了当前所有可训练变量值的检查点文件。
```
saver.save(sess, FLAGS.train_dir, global_step=step)
```

这样，我们以后就可以使用saver.restore()方法，重载模型的参数，继续训练。
```
saver.restore(sess, FLAGS.train_dir)
```