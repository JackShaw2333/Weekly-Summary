# Week #1
## 简介
### [网易云课堂 吴恩达 卷积神经网络](https://mooc.study.163.com/learn/2001281004?tid=2001392030#/learn/content)
#### 2.5 网络中的网络以及1x1卷积
![](pics/TIM%E6%88%AA%E5%9B%BE20180914092814.png)
1x1卷积也称为`Network in network`，因为每一次卷积相当于一个全连接层，对输入层的1x1x信道数切片进行运算。
它给神经网络添加了一个非线性函数，并能够在保持输入层长宽的基础上改变信道的数量。
输出结果的信道数等于使用的卷积核的数量。
1x1卷积在`Inception网络`中得到应用

#### 2.6 谷歌Inception网络简介
Inception网络可以代替人工确定卷积层中的卷积核类型，确定是否需要创建卷积层或池化层。
![](pics/TIM%E6%88%AA%E5%9B%BE20180914094011.png)
以上图为例，输入层为`28x28x192`。
1. 先使用64个1x1的卷积核，输出大小为28x28x64
2. 再使用128个3x3的`Same`卷积核，输出大小为28x28x128，将这一层叠到1x1卷积
3. 然后

通过1x1卷积构建“瓶颈层”可降低运算量

#### 2.7 Inception网络
构建卷积层时，通常要选择卷积核的大小，并考虑是否要加池化层，而Inception网络的作用就是帮助开发者决定。

初始的激活层分别经过5x5卷积核、3X3卷积、1x1卷积、最大值池化（不改变长宽）后按信道连接在一起，即组成一个Inception模块。

论文：[[Szegedy et al.,2014, Going Deeper with Convolutions]](https://arxiv.org/abs/1409.4842)

#### 2.8 使用开源的实现方案
如题

#### 2.9 迁移学习
做计算机视觉项目时，可以下载开源项目，利用他人已经训练好的网络结构的权重，将之作为预训练，然后转换到自己感兴趣的任务上。

若训练数据少，可冻结之前的隐层，只改动最后的Softmax层。

训练数据越多，可适当改动之前的隐层

总之，可用下载的权重代替随机初始化的权重。

#### 2.10 数据扩充
计算机视觉的一个问题：数据不够

1. 垂直镜像对称
2. 随机修剪（不完美）
3. 旋转、局部扭曲（不推荐）
4. 色彩转换，修改RGB值（可使用PCA颜色增强算法）

数据的扩充和训练过程可并行执行

#### 2.11 计算机视觉现状

数据过少，从事的更多是手工工程（精心设计特征、网络结构等）

在benchmarks训练和竞赛上的技巧
1. 综合模型，结果取平均值
2. Multi-crop，如10-crop，即将一张图剪切出10份，增加数据量

### 斯坦福CS231N课程

#### 2.1 图像分类 - 数据驱动方法
 关于K近邻算法，要防止过拟合现象的产生，K需要适当取大于1的数。
 K近邻算法的缺点在于训练快而预测慢，与实际需求冲突

#### 2.2 图像分类 - K最近邻算法
L1(Manhattan)距离
![$$d_1(I_1,I_2)=\sum_p|I_1^p-I_2^p|$$](http://latex.codecogs.com/gif.latex?\\$$d_1(I_1,I_2)=\sum_p|I_1^p-I_2^p|$$)​

L2(Euclidean)距离
![$$d_1(I_1,I_2)=\sqrt{\sum_p(I_1^p-I_2^p)^2}$$](http://latex.codecogs.com/gif.latex?\\$$d_1(I_1,I_2)=\sqrt{\sum_p(I_1^p-I_2^p)^2}$$)​

将数据分为`训练集`、`验证集`、`测试集`是较好的**寻找最佳超参数**的方法。
当数据较小时，可用**交叉验证**的方法，将数据集平分几份，每份轮流作验证集

K近邻算法的缺点：
1. 训练快，预测慢。
2. L1和L2距离算法不适合表示图像之间的相似度。
3. 需要训练点分布得尽可能密集，这样导致训练数据成倍增加，尤其在高维时所需的数据很大。

### Numpy
From [莫烦python](https://morvanzhou.github.io/tutorials/data-manipulation/np-pd/)

#### 2.2 创建array
```python
import numpy as np
#创建一个array
a = np.array([1,2,3])

#指明array元素的数据类型
a = np.array([1,2,3], dtype=np.int64)

#0元素矩阵
a = np.zeros((3,4))

#单位矩阵
a = np.ones((3,4))

#数据接近0的矩阵
a = np.empty()

#按指定范围创建数据
a = np.arange(12).reshape(3,4)

#创建线段
a = np.linspace(start, end, numbers).reshape(3,4)


```

#### 2.3&2.4 矩阵的基本运算
```python
#a和b对应元素相乘
c=a*b

#a于b矩阵相乘
c_dot = np.dot(a, b)
c_dot = a.dot(b)

#max(), min(), sum()
print(np.sum(a, axis = 0)
#axis = 0时以列为查找单元，axis = 1时以行为查找单元
#总之，axis=0代表跨行，=1代表跨列
```

#### 2.5 numpy的索引
```python
np.mean(A) 
np.average(A) 
A.mean() #求平均值

A.median() #求中位数

np.cumsum(A) #累加函数

np.diff(A) #累差

np.sort(A) #排序

np.transpose(A) #转置

A.T #转置

#把矩阵展开成1行
A.flatten()

for item in A.flat:
  print(item)
```

#### 2.6 numpy的array合并
```python
A = np.array([1,1,1])
B = np.array([2,2,2])

C = np.vstack((A,B)) #竖直合并
D = np.hstack((A,B)) #水平合并

#将行序列转化为列序列
A = np.array([1,1,1]) #shape = (3,)
print(A[np.newaxis,:]) #shape = (3,1)

#多个array合并
C = np.concatenate((A,B,B,A),axis=0) #竖向合并



```

#### 2.7 array的分割
```python
A = np.arange(12).reshape((3,4))
print(np.split(A, 2, axis=1)

np.hsplit(A, 2) #竖向分割
np.vsplit(A, 3) #横向分割
```

#### 2.8 copy&deep copy
```python
a = np.arange(4)
b = a #b为a的引用，b就是a

b = a.copy() #赋值
```

#### P.S. numpy.random
```python
import numpy as np
np.random.rand(d1,d2,...,dn) #根据指定维度生成[0,1)之间的数据

np.random.randn(d1,d2,...,dn) #根据指定维度生成按标准正态分布的数据

np.random.randint(low, high=None, size=None, dtype='l')  #返回在[low, high)随机整数，默认数据类型np.int。若high没填写，默认生成随机数[0, low)

np.random.random(size=None) #生成[0, 1)之间的浮点数

np.random.choice(a, size=None, replace=True, p=None) #从给定的一位数组中生成随机数
#当replace=False时，生成的随机数不能重复
#参数p为概率，p的长度与a一致，且p的数据之和为1

np.random.seed() #使随机数可预测。相同的seed，每次生成的随机数相同。


```

### Pandas
#### 3.1 pandas基本介绍
```python
import pandas as pd
import numpy as np
#创建了一个pandas序列
s = pd.Series([1,3,6,np.nan,44,1])

#创建DataFrame
df = pd.DataFrame(np.random.randn(3,4),index=X,columns=['a','b','c','d'])

df.dtypes #数据类型
df.index #df的行索引
df.columns #df的列索引
df.describe() #简单计算数据（如平均值，方差 etc）
df.T #DataFrame转置

df.sort_index(axis=1,ascending=False) #按索引排降序

df.sort_values(by='索引') #对值排序

```
#### 3.2 pandas选择数据
```python
#初始化代操作数据
dates = pd.date_range('20130101',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])

df['A'] or df.A #取A索引的数据


```


## Programs


## Summary
