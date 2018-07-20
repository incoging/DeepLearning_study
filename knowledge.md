## 记录与深度学习相关的知识点:
1. 卷积尺寸计算
2. 感受野计算


##### 1.卷积尺寸计算
输出尺寸=(输入尺寸-filter尺寸+2*padding）/stride + 1 <br>
卷积向下取整，池化向上取整。

##### 2.感受野计算
Notice:
1. 第一层卷积层的输出特征图像素的感受野的大小等于滤波器的大小
2. 深层卷积层的感受野大小和它之前所有层的滤波器大小和步长有关系
3. 计算感受野大小时，忽略了图像边缘的影响，即不考虑padding的大小

计算公式：(从最后一层往前算)
```
RF = 2 # 待计算的feature map上的感受野大小,比如VGG16最后一层为2x2的核，步长为2的池化，那么这时RF=2
　　for layer in （top layer To down layer）:
　　　　RF = ((RF -1)* stride) + fsize

stride 表示卷积的步长； fsize表示卷积层滤波器的大小
```
代码计算：
```
#!/usr/bin/env python

net_struct = {
    'alexnet': {'net': [[11, 4, 0], [3, 2, 0], [5, 1, 2], [3, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0]],
                'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5']},
    'vgg16': {'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                      [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                      [2, 2, 0]],
              'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                       'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3',
                       'pool5']},
    'zf-5': {'net': [[7, 2, 3], [3, 2, 1], [5, 2, 2], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
             'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5']}}

imsize = 800


def outFromIn(isz, net, layernum):
    totstride = 1
    insize = isz
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2 * pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride


def inFromOut(net, layernum):
    RF = 2
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF - 1) * stride) + fsize
    return RF


if __name__ == '__main__':
    print("layer output sizes given image = %dx%d" % (imsize, imsize))

    for net in net_struct.keys():
        print('************net structrue name is %s**************' % net)
        for i in range(len(net_struct[net]['net'])):
            p = outFromIn(imsize, net_struct[net]['net'], i + 1)
            rf = inFromOut(net_struct[net]['net'], i + 1)
            print("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (
            net_struct[net]['name'][i], p[0], p[1], rf))
123

```

