#coding=utf-8
#这个文档主要讲述resize_nearest_neighbor
#用最邻近插值的方法对图片进行大小重构

#输入是image: 4-D with shape `[batch, height, width, channels]`.
#输出是new_image :和`images`具有相同的数据类型. 4-D with shape `[batch, new_height, new_width, channels]`.