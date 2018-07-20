#coding:utf-8
#tf.train_done.Coordinator()
'''
Tensorflow提供了两个类来帮助多线程的实现
tf.train_done.Coordinator()和tf.QueueRunner
Coordinator类可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常.
QueueRunner类用来协调多个工作线程同时将多个张量推入同一个队列中


Coordinator类用来帮助多个线程协同工作，多个线程同步终止。 其主要方法有：

should_stop():如果线程应该停止则返回True。
request_stop(<exception>): 请求该线程停止。
join(<list of threads>):等待被指定的线程终止。

首先创建一个Coordinator对象，然后建立一些使用Coordinator对象的线程。这些线程通常一直循环运行，
一直到should_stop()返回True时停止。 任何线程都可以决定计算什么时候应该停止。它只需要调用request_stop()，同时其他线程的should_stop()将会返回True，然后都停下来。
'''
import tensorflow as tf

# Create the graph, etc.
init_op = tf.initialize_all_variables()
# Create a session for running operations in the Graph.
sess = tf.Session()
# Initialize the variables (like the epoch counter).
sess.run(init_op)
# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    while not coord.should_stop():
        # Run training steps or whatever
        sess.run("train_op")
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()


#coord.join
'''
这个调用阻塞，直到一组线程终止。 线程集合是通过调用`Coordinator.register_thread（）`在`threads`参数中传递的线程与通过协调器注册的线程列表的联合。

在线程停止之后，如果`exc_info`被传递给`request_stop`，那么这个异常将被重新引发。

宽限期处理：当调用request_stop（）时，线程被赋予“stop_grace_period_secs”秒终止。 
如果其中的任何一个在这段时间过后仍然存在，就会产生一个`RuntimeError`。 请注意，如果`exc_info`被传递给`request_stop（）`，那么它会被引发而不是`RuntimeError`。

线程：`threading.Threads`的列表。 除了已注册的线程之外，还开始加入线程。
'''