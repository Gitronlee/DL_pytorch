## 1 深度学习介绍
#### 1.1 人工智能
#### 1.2 数据挖掘、机器学习、深度学习
###### 1.2.1 数据挖掘
从数据中挖掘到有意义的信息，寻找数据之间的特性。处理数据、预处理数据、分析和挖掘、可视化、可用信息。
###### 1.2.2 机器学习
监督学习、无监督学习、半监督学习、迁移学习、增强学习（通过观察周围环境学习）
###### 1.2.3 深度学习  
目前比较流行的网络结构有：深度神经网络DNN、卷积神经网络CNN、循环递归神经网络RNN、生成对抗网络GAN等等
#### 1.3 学习资源与建议
理论和工程相结合。python、微积分、线性代数基础。机器学习基础 吴恩达、林轩田、udacity、周志华、李航。深度学习course NNFML，231,224。

## 2 深度学习框架
#### 2.1 框架介绍：
##### tensorflow：
 Google、c++，使用人数最多最庞大社区，有着python和c++接口，基于tensorflow的几个第三方库有 keras、tflearn、tfslim、tensorlayer等
##### Theano：
tensorflow更像是Theano的孩子。
##### Torch：
facebook、pytorch的前身、更加灵活，支持动态图、python接口
##### MXNet：
李沐、亚马逊各种语言接口、教程不完善社区小。
#### 2.2 pytorch介绍
##### 2.2.1 什么是
python开发、强大的GPU加速、动态神经网络、tensorflow（原不支持）
可以看作是加入了GPU支持的numpy，拥有自动求导的强大的深度神经网络。
##### 2.2.2 为何
多学一个框架，以备不时之需。（所以并非不学tensorflow）
动态的，可以让你零延迟的任意改变神经网络的行为。灵活性
设计思路线性，易于使用。
相比于tensorflow更加直观简洁。轻松扩展。
#### 2.3 配置环境
anaconda、本书 py3.6 设置清华源修改channels：https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
暂时不配置了 先用Google的哪个colab https://colab.research.google.com/drive/ 记得修改笔记本设置为GPU
## 3 多层全连接神经网络
多层全连接神经网络是现在深度学习中各种复杂网络的基础。本章从pytorch基础介绍处理对象，运算操作，自动求导，数据处理，分类，多层全连接神经网络，反向传播算法，各种基于梯度的优化算法，预处理训练技巧，以pytorch实现。
#### 3.1 pytorch基础 
##### 3.1.1 张量 Tensor
是pytorch中最基本的操作对象tensor，即多维矩阵。可以与numpy中的ndarray相互转换。
```python
import torch
import numpy as np
torch.cuda.is_available()
a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
print("a is :{}".format(a))#farmat 为格式化替换｛｝ 默认是float类型
print('a size is {}'.format(a.size()))
b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])
print("b is :{}".format(b))
print('b size is {}'.format(b.size()))
c = torch.zeros((3, 2))
print("c is :{}".format(c))
print('c size is {}'.format(c.size()))
d = torch.randn((3, 2))
print("d is :{}".format(d))
print('d size is {}'.format(d.size()))
a[0,1] = 100
print("a is :{}".format(a))
numpy_b = b.numpy()
print('conver to numpy is\n {}'.format(numpy_b))
e = np.array([[2,3],[4,5]])
torch_e = torch.from_numpy(e)
print('from numpy to torch.Tensor is {}'.format(torch_e))
f_torch_e = torch_e.float()
print('change type is {}'.format(f_torch_e))#多了个点
if torch.cuda.is_available():
    a_cuda = a.cuda()
    print(a_cuda)
```
##### 3.1.2 变量 Variable
是神经网络计算图里特有的一个概念。Variable提供了自动求导的功能。本质上Variable和Tensor无区别。不过Variable被放入一个计算图里 然后进行前向传播、反向传播、自动求导。
Variable有三个比较重要的属性：data 可以取出其中的tensor数值，grad是这个Variable的反向传播梯度。grad_fn 表示得到这个Variable的操作（如加减乘除等操作）。
```python
import torch
import numpy as np
from torch.autograd import Variable

x = Variable(torch.Tensor([5]), requires_grad = True)
w = Variable(torch.Tensor([6]), requires_grad = True)
b = Variable(torch.Tensor([7]), requires_grad = True)

y = w * x + b
y.backward()

print(x.grad)
print(w.grad)
print(b.grad)
x = torch.randn(3)
x = Variable(x,requires_grad = True)
y = x * 2
print(y)
y.backward(torch.FloatTensor([1, 0.1, 0.01]))#每个分量的梯度乘上1,0.1,0.01
print(x.grad)
```
##### 3.1.3 数据集 Dataset
pytorch提供了很多工具使得数据的读取和预处理变得容易
torch.utils.data.Dataset 是代表这一数据的抽象类，你可以继承和重写。
