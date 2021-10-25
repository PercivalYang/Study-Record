# 基础网络架构

## AlexNet

- 文献读完感觉像在做技术报告，没有与当时的其他网络架构进行比较，只是强调自身网络架构在ImageNet比赛上的表现；
- 在non-linearity activation function上没用tanh、sigmoid，而是使用ReLU，原因是在Gradient Descent上ReLU的计算速度要比前两个快；
- 使用了Local Response Normalization 的归一化方法，作用个人理解为防止梯度消失和梯度爆炸；
- 引入DropOut避免Overfitting，以及将神经网络切割为两部分，放在两个GPU上进行计算；
- 本文有提到"若在硬件上能实现更大的计算能力，建立更深的网络层结构，则会达到更好的效果"，这也是之后在探索神经网络的一段时间内，学者的注意力都集中在建立**更深的网络层结构**

## VGGNet

- 整个系统框架都大量采用3$\times$3的Conv cell，这样做的效果：
  - 使得VGG的网络层数可以更深，例如VGG16，VGG19；
  - 小型的Conv cell相比直接使用大型Conv cell，更加节省参数数量（类似之后的bottleneck，先用1\*1的Conv cell降维-->再用3\*3的Conv cell提取feature-->最后用1\*1的进行升维还原H,W）
- 系统框架图：![image-20211016210341847](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211016210341847.png)
- 在高层的网络中不再只采用一个非线性激活函数，而是采用3个（如上图中VGG16~19中加粗conv层后都紧随一个非线性激活函数），这样做的效果可以使决策函数具有更强的区分能力；

## ResNet

- 在Introduction中说到网络架构的深度的重要性，在以往十几层的网络架构中出现了梯度消失/爆炸的问题，但可以通过normalization initialization解决，但是当建立到更高的层数（如文中56和20层的比较），网络的性能又会大幅度的降低（且这种降低不是因为Overfitting所引起），本文中称之为**退化问题**

- **Residual block**：为了让高层的Conv Layer保存浅层的特性，不会出现较大的偏差，在原先前向传播的网络架构上加入了"**shortcut**"（如下图所示），这样高层的网络同样可以使用SGD的BP算法进行优化，而不会产生退化。

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211016215112481.png" alt="image-20211016215112481" style="zoom:67%;" />

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017075601879.png" alt="image-20211017075601879" style="zoom:80%;" />

- shortcut的引入没有增加额外的参数和计算复杂度，同时在引入shortcut的block中的卷积层数量是不固定的，如下图所示

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017082047753.png" alt="image-20211017082047753" style="zoom:80%;" />

  - 上图右边bottleneck结构的目的是为了减少计算的参数，若直接使用3$\times$3,256的conv cell提取特征矩阵，则其参数数量为
    $$
    3\times3\times256\times256=589,824
    $$
    而使用bottleneck结构的参数数量为：
    $$
    1\times1\times64\times256+3\times3\times64\times64+1\times1\times256\times64=69,632
    $$
    可见参数数量减少了约9倍

- 在观察ResNet 34层的整体网络架构时，可以发现shortcut有实线和虚线两种类别，其中实线和上图中一样只是identity map的作用；而虚线（如下图）的作用，此处我的理解认为其应该是起到downsample的作用

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017083827285.png" alt="image-20211017083827285" style="zoom: 67%;" />

# MobileNet

## MobileNet-V1

- 作为Mobilenet的一个版本，文中比较重要的有两点：
  - 引入**Depthwise separable convolution **来构造轻量级深度神经网络
  - 引入两个Hyper-parameter：**width multiplier** & **resolution multiplier** 来定义小且效率的MobileNet

### Depthwise separable convolution

- 这其实是两种卷积结构的结合：**DW(Depthwise) Conv + PW(Pointwise) Conv**，他们的结构如下图所示

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017085145804.png" alt="image-20211017085145804" style="zoom:80%;" />

### Width multiplier & Resolution multiplier

- **Width multiplier**：假设为$\alpha$($\alpha\in(0,1]$)，并假设Network每一层输入的Channel数量为M

  - 当设置了$\alpha$后，则每一层的output channel会减少为$\alpha M$

- **Resolution multiplier**：减少输入矩阵的边长$D_F$，假设参数为$\rho$($\rho\in(0,1]$)

  - 当设置了$\rho$后，每一层输入的feature map(或 input image)的边长变为$\rho D_F$

- 在设置了上述两个参数后，计算所需的参数量为：
  $$
  D_K\times D_K\times\alpha M\times\rho D_F\times\rho D_F+\alpha M\times\alpha N\times\rho D_F\times\rho D_F
  $$
  ​	其中$D_K$为DW卷积核的边长，N为PW卷积核的个数(如下图所示)

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017090654037.png" alt="image-20211017090654037" style="zoom:80%;" />

### Architecture

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017090810322.png" alt="image-20211017090810322" style="zoom:80%;" />

- 从表中可发现，在第一层使用了一层standard conv对输入图像的特征进行提取，后面的卷积层都是DW+PW的卷积

- 每层卷积层后，都会紧随BatchNormal+ReLU的Layer(如下图所示)

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017091100632.png" alt="image-20211017091100632" style="zoom:67%;" />

## MobileNet-V2

- 第二个版本的MobileNet中比较重要的两点：
  - **Linear Bottleneck**：防止非线性激活函数，破坏数据的完整性
  - **Inverted Residual**：提高梯度在Multiplier Layer之间的传播能力

### Linear Bottleneck

- 这一部分的理解不够深入，对文中部分地方仍存在疑问；
- 起因：由于Network中每一层Conv后都有一个激活张量，这些激活张量集合会形成一个“**manifold of interest**”，并且学者们认为这种manifold可以通过空间操作将它转入低维空间。
  - 在MobileNet-V1中的width multiplier就可以通过减少Layer的数量，来降低空间操作的难度
  - 但在实验中发现，**ReLU函数会对低维的特征信息造成大量的损失**，因此需要替换成Linear Activation来保留完整信息

### Inverted Residual

- 借鉴了ResNet的残差结构，如下图所示：

![image-20211017101213906](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017101213906.png)

- 须注意的是：

  - 倒残差使用的激活函数为ReLU6，其表达式：
    $$
    ReLU6 = min(max(0,x),6)
    $$

  - 中间提取特征层采用的Dwise Conv，如下图：

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017101910095.png" alt="image-20211017101910095" style="zoom:67%;" />

  - 由上图也可注意当stride=1时，会引入shortcut；当stride$\ne$1时，则没有shortcut

### Architecture & Some Details

- 系统整体框架图（图中bottleneck 即inverted residual block）：

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017102040385.png" alt="image-20211017102040385" style="zoom:67%;" />

- 上图各参数：

  - t：扩展因子（表示input经过bottleneck第一层的conv 1$\times$1后，channel会扩展为原来的t倍：
    $$
    Input:h\times w\times c\Rightarrow Output:h\times w\times (tc)
    $$

  - c：输出的channel数量

  - n：bottleneck的重复次数

  - s：stride数，**对于n大于1的bottleneck，s只针对第一层bottleneck，其它后续bottleneck的s=1**

- 另外表中倒数第三层在编程中可用`flatten`进行展平降维，最后一层相当于是线性激活层，然后送入`softmax`中

## MobileNet-V3

- 第三个版本相比于第二版本在精确度参数数量和MAdds的优化上更进一步，比较值得注意的有：
  - 更新了Inverted Residual Block，加入了SE模块
  - 通过NAS和NetAdapt 搜索最优的Network Architecture：**这一部分还未深入研究**
  - 重新设计了耗时层结构：再通过NAS & NetAdapt搜索到的最优框架中发现，在最后基层Layers的计算量要比其他层大许多，于是手动对其做了调整

### New Inverted Residual Block with Squeeze + Excite

- 更新后的模块图如下：

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017103744818.png" alt="image-20211017103744818" style="zoom:80%;" />

- 其中SE模块放大图如下：

  <img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017103904500.png" alt="image-20211017103904500" style="zoom:80%;" />

  - SE模块的想法：对提取出来的feature block的每一层channel进行计算，赋予每一层channel一个权重，然后再返回其中
  - SE模块的计算步骤：
    - 先从单独一层channel 的feature matric，将它通过AvgPooling缩小为原来的**四分之一**；
    - 再通过两个全连接层计算其权重，最后返回到Inverted residual block中

###hard-swish & hard-sigmoid

- swish函数：
  $$
  swish(x)=x\times\sigma(x)
  \ \&\ \sigma(x)=\frac{1}{1+e^{-x}}
  $$

- 由于sigmoid函数在求导过程中计算量大，于是推出**h-sigmoid**函数：
  $$
  h-sigmoid(x)=\frac{ReLU6(x+3)}{6}
  $$

- 同样，**h-swish**函数：
  $$
  h-swish(x)=x\frac{ReLU6(x+3)}{6}
  $$

### Network Architeture

#### MobileNet-V3-Large

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017105532996.png" alt="image-20211017105532996" style="zoom:67%;" />

- 表中须注意的有：
  - **exp size**(expand size)：同MobileNet-V2里的扩展因子，即第一层1$\times$1的Conv Cell中对channel进行扩展
  - **\#out**：输出的channel数量
  - **SE**：是否引入SE模块
  - **NL**：采用哪种非线性激活函数（HS：h-siwsh；RE：ReLU）
  - **s**：stride
  - **NBN**：不采用BatchNormalization
  - **k**：No. of classes

#### MobileNet-V3-Small

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211017110527176.png" alt="image-20211017110527176" style="zoom:67%;" />

# Code Study

## 初始化权重

```python
"""
放在net的def __init__(self,...):里，如这里前面可以是
def __init__(self,...):
...此处为架构程序
"""
for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

"""
def forward(self,x):
...
"""
```

## Data Augmentation

```python
data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
```

## MobileNet V2~V3

### 直观易懂版

- [MobileNet-V2](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test6_mobilenet/model_v2.py)
- [MobileNet-V3](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test6_mobilenet/model_v3.py)

### 官方版

- [Tensorflow_MobileNet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)

- [Pytorch_MobileNet](https://github.com/rwightman/pytorch-image-models)

# 总结

- 本周的学习收获：
  - **网络架构方面**：AlexNet、VGG、ResNet、NASNet-2017、MobileNetV1~V3
  - **深度学习方面**：迁移学习、Meta-learning

- 以上的收获大都停留在基础的了解范畴内，对其中跟自己方向相关的内容（如MobileNet）还需在细节上更加完善。同时缺乏项目的实践，在之后的学习安排中，计划花费更多的时间在Github上寻找优质的动手项目，通过自己动手完整的做一遍MobileNet的项目实战来完善其中的细节；
- 下周计划学习内容：
  - **MobileNetV2~V3的项目实战**；
  - **网络架构方面**：ShuffleNet、EfficientNet
  - **深度学习方面**：根据学习网络架构需要的掌握的知识决定（如NASNet中使用了迁移学习的知识）

