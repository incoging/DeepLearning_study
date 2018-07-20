# coding=utf-8
"""
最后一层全连接层输出V=[x1,x2,x3],真实标签是[1,0,0].那么假设V=[x1,x2,x3]是[3.1,3,3],
那么softmax的公式使得其只需要V的模长增加倍数即可以降低loss损失.这太容易(只需要增大参数即可)使得网络往往就是这样做的.
而不是我们通常想要的那样去努力降低x2,x3的相对于x1的值如[3.1,1,1]这样.这也是所以L2正则会缓解过拟合的一个原因.
"""

import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


# 测试结果
scores = [9.3, 9.0, 9.0]
dou_scores = [2 * x for x in scores]
print(softmax(scores))
print("--------separatrix-----------")
print(softmax(dou_scores))