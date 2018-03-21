##### one_hot

直接先上代码
```python
from sklearn.preprocessing import OneHotEncoder   
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1],[1, 0, 2]])
print ("enc.n_values_ is:",enc.n_values_)
print ("enc.feature_indices_ is:",enc.feature_indices_)
print (enc.transform([[0, 1, 3]]).toarray())
```
输出结果：
```
enc.n_values_ is: [2 3 4]
enc.feature_indices_ is: [0 2 5 9]
[[1. 0. 0. 1. 0. 0. 0. 0. 1.]]
```

解释：
fit内是自己写的一个一个样例，然后onehot编码会根据这些样例进行编码，
如上，样例中有三个维度（也就是三个属性），样例中每一个维度最大的数字表示这个类别中含有（数字+1）类，（因为从0开始）
样例中第一维最大1，说明2类
样例中第一维最大2，说明3类
样例中第一维最大3，说明4类
则，将这些编码到一行会是，0,0 | 0,0,0 | 0,0,0,0

所以最后一行的试例[0, 1, 1] 对应的就是第一维的第一类，第二维度的第二类，第三维度的第二类，
所以结果就是该类所在的位置为1，其他为0   
[[1, 0, 0, 1, 0, 0, 1, 0, 0]]   

若样例中含有[5, 0,3] 则说明第一维度含有6类，则排成一列会是；   
0,0,0,0,0,0 | 0,0,0 | 0,0,0,0    
试例[0, 1, 1]的结果将变为：   
[[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]] 

接着解释：
n_values:是一个数组，表示每一个维度含有多少个类别
feature_indices_：则是对n_values的一个累加。 



