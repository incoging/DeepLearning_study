#coding=utf-8

import os

image_dir = "/home/hp-z840/yjq/lkf/data/valid/valid_test"
files = sorted(os.listdir(image_dir))

for idx, file in enumerate(files):
    # print idx
    # print("-----------------------")
    # print file
    '''
    output:
    0
    -----------------------
    00a2aeec4c5d7a7e3ffd9fced147f638a061cf4c.jpg
    即:file是文件的全名,idx是文件的索引.
    
    '''

    if (idx + 1) % 1000 == 0 or (idx + 1) == len(files):
        print(str(idx + 1) + ' / ' + str(len(files)))