### ROI pooling

Roi pooling层也是pooling层的一种，只是是针对于Rois的pooling操作而已。

Roi pooling层的过程就是为了将proposal抠出来的过程，然后resize到统一的大小。

#### Fast RCNN 整体结构

![20180119150020632](Z:\study\deepLearning\notes\ROI pooling.assets/20180119150020632.png)

#### ROI Pooling输入

1. 特征图：指的是图1中所示的特征图，在Fast RCNN中，它位于RoI Pooling之前，在Faster RCNN中，它是与RPN共享那个特征图，通常我们常常称之为“share_conv”；
2. rois：在Fast RCNN中，指的是Selective Search的输出；在Faster RCNN中指的是RPN的输出，一堆矩形候选框框，形状为1x5x1x1（4个坐标+索引index），其中值得注意的是：坐标的参考系不是针对feature map这张图的，而是针对**原图**的（神经网络最开始的输入） 

#### 步骤：

1. 根据输入image，将ROI映射到feature map对应位置；
2. 将映射后的区域划分为相同大小的sections（sections数量与输出的维度相同）；
3. 对每个sections进行max pooling操作；



eg:

**考虑一个8\*8大小的feature map，一个ROI，以及输出大小为2*2.** 

**（1）输入的固定大小的feature map**  

![20171112193549325](Z:\study\deepLearning\notes\ROI pooling.assets/20171112193549325-1533027333848.jpg)

**（2）region proposal 投影之后位置（左上角，右下角坐标）：（0，3），（7，8）。** 

![20171112193608750](Z:\study\deepLearning\notes\ROI pooling.assets/20171112193608750.jpg)

**（3）将其划分为（2\*2）个sections（因为输出大小为2*2），我们可以得到：** 

![20171112193628721](Z:\study\deepLearning\notes\ROI pooling.assets/20171112193628721.jpg)

**（4）对每个section做max pooling，可以得到：** 

![20171112193709781](Z:\study\deepLearning\notes\ROI pooling.assets/20171112193709781.jpg)

