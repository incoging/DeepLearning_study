### 记录tensorflow函数之外的相关问题

##### 1. Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX
```
高级矢量扩展（AVX）是英特尔在2008年3月提出的英特尔和AMD微处理器的x86指令集体系结构的扩展，
英特尔首先通过Sandy Bridge处理器在2011年第一季度推出，
随后由AMD推出Bulldozer处理器在2011年第三季度.AVX提供了新功能，新指令和新编码方案。
特别是，AVX引入了融合乘法累加（FMA）操作，加速了线性代数计算，即点积，矩阵乘法，卷积等。
几乎所有机器学习训练都涉及大量这些操作，因此将会支持AVX和FMA的CPU（最高达300％）更快。
该警告指出您的CPU确实支持AVX（hooray！）。
```
解决办法 ：
1.编译Tensorflow源码进行安装,详看：
```
question about this: http://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions
TensorFlow guide to build from source: https://www.tensorflow.org/install/install_sources
```
2.在代码中加入：
```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```