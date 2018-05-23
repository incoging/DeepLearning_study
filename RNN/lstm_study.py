# coding=utf-8
# long short-term memory 长短时记忆网络

import tensorflow as tf

# # 定义一个LSTM结构
# lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
#
# # 将LSTM中的养成初始化为全0数组，BasicLSTMCell类提供了zero_state函数来生成
# # 全零的初始状态。state是一个包含两个张量的LSTMStateTuple类， 其中state.c和
# # state.h分别对应了图8-7中的c状态和h状态
# # 和其他神经网络类似，在优化循环神经网络 时，每次也会使用一个batch的训练样本。
# state = lstm.zero_state(batch_size=batch_size, tf.float32)
#
# # 定义损失函数
# loss = 0.0
# # 虽然在测试时循环神经网络可以处理任意长度的序列，但是在训练中为了将循环网络展开成
# # 前馈神经网络，我们需要知道训练数据的序列长度。
# for i in range(num_steps):
#     if i > 0: tf.get_variable_scope().reuse_variables()
#
#     # 每一步处理时间序列中的一个时刻，将当前输入current_input和前一时刻状态state传入
#     # 定义 的LSTM结构可以得到当前LSTM的输出lstm_output和更新后状态state，lstm_output
#     # 用于输出给其他层，state用于输出给下一时刻，它们在drouput等方面可以有不同的处理方式
#     lstm_output, state = lstm(current_input, state)
#     # 将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出
#     final_output = fully_connected(lstm_output)
#     loss += calc_loss(final_output, expected_output)


