# coding=utf-8
"""
TensorBoard会自动读取最新的TensorFlow日志文件，并呈现当前TensorFlow程序运行的最新状态，
"""
import tensorflow as tf

# # 实现日志输出功能
# input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
# input2 = tf.Variable(tf.random_uniform([3]), name="input2")
# output = tf.add_n([input1, input2], name="add")
#
# # 生成一个写日志的writer，并将当前的TensorFlow计算图写入日志,引号为要存储的文件夹。
# writer = tf.summary.FileWriter("./data/log/log1", tf.get_default_graph())
# writer.close()

# 上面程序的改进版，使用命名空间
# with tf.name_scope("input1"):
#     input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
# with tf.name_scope("input2"):
#     input2 = tf.Variable(tf.random_uniform([3]), name="input2")
# output = tf.add_n([input1, input2], name="add")
# writer = tf.summary.FileWriter("./data/log/log2", tf.get_default_graph())
# writer.close()

# # 使用手写体识别的程序展示神经网络结构
# from tensorflow.examples.tutorials.mnist import input_data
# import sys
#
# sys.path.append("./data/")
# import mnist_inference
#
# # 定义神经网络的参数。
#
# BATCH_SIZE = 100
# LEARNING_RATE_BASE = 0.8
# LEARNING_RATE_DECAY = 0.99
# REGULARIZATION_RATE = 0.0001
# TRAINING_STEPS = 3000
# MOVING_AVERAGE_DECAY = 0.99
#
#
# # 定义训练的过程并保存TensorBoard的log文件。
#
#
# def train(mnist):
#     #  输入数据的命名空间。
#     with tf.name_scope('input'):
#         x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
#         y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
#     regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
#     y = mnist_inference.inference(x, regularizer)
#     global_step = tf.Variable(0, trainable=False)
#
#     # 处理滑动平均的命名空间。
#     with tf.name_scope("moving_average"):
#         variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#         variables_averages_op = variable_averages.apply(tf.trainable_variables())
#
#     # 计算损失函数的命名空间。
#     with tf.name_scope("loss_function"):
#         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
#         cross_entropy_mean = tf.reduce_mean(cross_entropy)
#         loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
#
#     # 定义学习率、优化方法及每一轮执行训练的操作的命名空间。
#     with tf.name_scope("train_step"):
#         learning_rate = tf.train.exponential_decay(
#             LEARNING_RATE_BASE,
#             global_step,
#             mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
#             staircase=True)
#
#         train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#
#         with tf.control_dependencies([train_step, variables_averages_op]):
#             train_op = tf.no_op(name='train')
#
#     writer = tf.summary.FileWriter("log", tf.get_default_graph())
#
#     # 训练模型。
#     with tf.Session() as sess:
#         tf.global_variables_initializer().run()
#         for i in range(TRAINING_STEPS):
#             xs, ys = mnist.train.next_batch(BATCH_SIZE)
#
#             if i % 1000 == 0:
#                 # 配置运行时需要记录的信息。
#                 run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#                 # 运行时记录运行信息的proto。
#                 run_metadata = tf.RunMetadata()
#                 _, loss_value, step = sess.run(
#                     [train_op, loss, global_step], feed_dict={x: xs, y_: ys},
#                     options=run_options, run_metadata=run_metadata)
#                 writer.add_run_metadata(run_metadata=run_metadata, tag=("tag%d" % i), global_step=i)
#                 print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
#             else:
#                 _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
#
#     writer.close()
#
#
# def main(argv=None):
#     mnist = input_data.read_data_sets("X:\jet_python\deepLearning\\tensorFlow\MNIST_data", one_hot=True)
#     train(mnist)
#
#
# if __name__ == '__main__':
#     main()


# 监控指标可视化

from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = "./data/log/log3"
BATCH_SIZE = 128
TRAIN_STEPS = 3000


# 生成变量监控信息并定义生成监控信息日志的操作，其中var给出了需要记录的张量，
# name给出了在可视化结果中显示的图表名称，这个名称一般与变量名一致
def variable_summaries(var, name):
    with tf.name_scope("summaries"):
        # 通过tf.summary.histogram函数记录张量中的元素的取值分布，对于给出的图表名称和张量，
        # tf.summary.histogram函数会生成一个Summary protocol buffer. 将Summary写入TensorBoard日志文件，
        # 在HISTOGRAMS栏和DISTRIBUTION栏都会出现对应名称的图表
        tf.summary.histogram(name, var)

        # 计算变量的平均值，并定义生成平均值信息日志的操作， "mean/" + name，其中mean为命名空间，/ 是命名的分隔符
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean/" + name, mean)
        # 计算变量的标准差，并定义生成其日志的操作
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev/" + name, stddev)


# 生成一层全链接层神经网络
def nn_layer(input_tensor, input_dim, output_dim,
             layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + "/weights")

        # 声明神经网络的偏置项，并调用生成偏置项监控信息日志的函数
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + "biases")

        # 记录神经网络输出节点在经过激活函数之前的分布
        with tf.name_scope("Wx_plus_b"):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + "/pre_activations", preactivate)
        activations = act(preactivate, name="activation")

        # 记录激活之后的分布
        tf.summary.histogram(layer_name + "/activations", activations)

        return activations
    pass


def main(_):
    mnist = input_data.read_data_sets("X:\jet_python\deepLearning\\tensorFlow\MNIST_data", one_hot=True)
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 10], name="y-label")

    # 将输入向量还原成图片的像素矩阵，并通过tf.summary.image函数定义将当前的图片信息写入日志的操作
    with tf.name_scope("input_reshape"):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image("input", image_shaped_input, 10)

    hidden1 = nn_layer(x, 784, 500, "layer1")
    hidden2 = nn_layer(hidden1, 500, 256, "layer-hidden2")
    y = nn_layer(hidden2, 256, 10, "layer2", act=tf.identity)

    # 计算交叉熵并定义生成交叉熵监控日志的操作
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar("cross entropy", cross_entropy)

    with tf.name_scope("train"):
        train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)

    # 计算模型在当前给定数据上的正确率
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)
        pass
    pass
    # 上面定义的tf.summary.scalar, histogram, image等都需要sess.run来明确调用，所以
    # tf.summary.merge_all函数来整理所有的日志生成操作，运行这个操作就可以将代码中定义的所有日志生成操作
    # 执行一次
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # 初始化写日志的writer，并将当前Tensorflow计算图写入日志
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 运行训练步骤以及所有的日志生成操作，得到这次运行的日志
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys})
            # 将所有日志写入文件，TensorBoard程序就可以拿到这次运行所有对应的运行信息
            summary_writer.add_summary(summary, i)
        pass
    pass

    summary_writer.close()


if __name__ == '__main__':
    tf.app.run()