# 论文篇

## ShuffleNet-V2

### GuideLines

- 在以往的Network 论文中，**FLOPs** 是常被用来评测计算成本的重要指标，但本文作者指出在网络的所有计算成本中， 还有其它指标占据相当一部分运算的时间，例如：**Element-wise operation、Shuffle、Data process**。于是作者在FLOPs 的基础上提出了新的评价指标：**MAC(Memory Access Cost)**

  - 假设参数：
    - $c_1$：输入channel 数量
    - $c_2$：输出channel 数量
    - $h,w$：输入 feature map 的高宽
  - 目前许多网络采用Group Conv，其中1$\times$1的卷积核占据了大部分的计算成本，根据**FLOPs** 计算，一个1$\times$1卷积核的计算成本$B=h\times w\times c_1\times c_2$ ；**MAC** 则为$MAC=hw(c_1+c_2)+c_1c_2$ ；

- **Guideline 1**：Equal channel width minimizes memory access cost (MAC).

  - 根据**均值不等式**：$\sqrt{c_1c_2}\le\frac{c_1+c_2}{2}$，可以得到：
    $$
    MAC\ge 2\sqrt{hwB}+\frac{B}{hw}
    $$

  - 根据上式可知，FLOPs 给定了MAC下限，当输入channel$c_1=$ 输出通道$c_2$时，MAC达到最小值。

- **Guideline 2**：Excessive group convolution increases MAC.

  - 在Group Conv 下，$B=\frac{hwc_1c_2}{g}$，其中$g$ 是groups 的数量，则可得MAC如下：
    $$
    MAC=hw(c_1+c_2)+\frac{c_1c_2}{g}=hwc_1+\frac{Bg}{c_1}+\frac{B}{hw}
    $$

  - 化简成最右边的形式是为了**消除等式中的$c_2$**，这样可以看出当给定 input feature map 的$h,w,c_1$以及计算成本$B$ 时，MAC随$g$ 的增大而减小。

- **Guideline 3**：Network fragmentation reduces degree of parallelism.

  - 网络架构中的分叉会降低并行计算速度，例如：GoogleNet 中的Inception 模块，ResNet 中的Residual Block等

- **Guideline 4**：Element-wise operations are non-negligible.

  - ReLU、AddTensor、AddBias等，都是有着较小的FLOPs，但却有很大的MAC

- 结论：

  - 采用相同channel 输入输入出结构的Conv Layer
  - 谨慎使用Group Conv
  - 降低网络分支的层次
  - 减少使用Element-wise operation

### Architecture

- 标准ShuffleNet-V2 block：

  ![image-20211025162628611](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211025162628611.png)

  - 加入Channel Split，将通道数分为$c'$ 和$c_1-c'$ (论文中采用$c'=c_1/2$)
  - 将$1\times1\ Group\ Conv$ 换成 $1\times1\ Conv$ (遵循**G2**)
  - 改变ReLU 位置

- Down Sample ：

  ![image-20211025163337060](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20211025163337060.png)

  - 没有Channel Split，因此输出的channels 是输入的两倍；
  - 更换了V1左边分支的AvgPool。

