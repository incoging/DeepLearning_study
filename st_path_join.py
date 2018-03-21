#coding:utf-8
#这个文档主要介绍os.path

import os

path = "/home/hp-z840/yjq/study/test"
newpath1 = os.path.join(path, "test1/")
newpath2 = os.path.join(path, "test2")
os.system("mkdir " + newpath1)
os.system("mkdir " + newpath2)
