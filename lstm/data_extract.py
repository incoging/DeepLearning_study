# coding=utf-8
from tensorFlow.models.tutorials.rnn.ptb import reader

DATA_PATH = "X:\jet_python\deepLearning\\tensorFlow_study\lstm\data\simple-examples\data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

# 将训练数据组织成batch大小为4,截断长度为5的数据组。
result = reader.ptb_producer(train_data, 4, 5)
# 读取第一个batch中的数据，其中包括每个时刻的输入和对应的正确输出
x, y = result.next()
print("X:", x)
print("Y:", y)