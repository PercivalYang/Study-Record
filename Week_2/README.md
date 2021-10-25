[TOC]

# 论文篇

## 一、血管分割_OctaveUNet

### 1 Introduction

- Octave Convolution$\Rightarrow$参考文献[49]
- Octave Convolution 构造feature encoder block，然后通过网络中不同的层，来学习图像中不同层次的频率信息；
- 上述学习到的高频和低频信息具有不同的特征，相比于标准卷积提高了一定性能
- Octave Transposed convolution$\Rightarrow$用来解码前面学习到的features
- UNet$\Rightarrow$参考文献[36]

### 2 Method

#### 2.1 Tranpose Conv

- 目的：用作`decoder`中，充当upsample ，还原图片中 pixel 之间的关联性；
- 和`bottleneck`中的1$\times$1 升/降维conv不同，bottleneck 中改变的是 channel，而`transpose conv` 是将特征图(H/n, W/n) 还原为原始图片大小(H,W)，且保留pixel之间的关联性

#### 2.2 Octave UNet

##### 2.2.1 Octave Feature Representation<a name='X'></a>

- 起因：对于输入进Conv Cell的 Feature maps，其中可能有一部分特征图代表着输入图像的低频分量，并且这些低频分量在空间上是冗余的，可以进一步的压缩。因此作者提出**Octave Feature Representation**，通过 scale-space theory 对空间信息划分为高频、低频分量，并用$\alpha$ 来表示低频分量所占的比率
  $$
  输入Feature\ maps\ X\in\R^{c\times h\times w},将X=\{X^{H},X^{L}\}；\\其中X^{H}\in\R^{(1-\alpha)c\times h\times w},X^{L}\in\R^{c\alpha\times \frac{h}{2}\times\frac{w}{2}}
  $$

- 在血管分割论文中：

  - **low frequency**：主要描述平滑变化的结构，例如：主血管
  - **high frequency**：主要描述突变的细节结构，例如：细小血管

##### 2.2.2 Octave Conv

- 起因：**vanilla conv**不能直接作用在前面提到的**Octave Feature Representation**，为了改进并且不增加计算成本，提出了**Octave Conv**

- 设计目标：

  - 有效处理高低频分量
  - 实现有效的频间通信

- 设输入、输出、权重变量为$X,Y,W$，根据[前文](#X)可知将$X$进行了高低频划分，我们也将$Y,W$进行划分如下：
  $$
  Y=\{Y^H,Y^L\},\ W=\{W^H,W^L\}
  $$
  为了增加频间通信，令
  $$
  Y^H = Y^{L\rightarrow H}+Y^{H\rightarrow H},\ Y^L=Y^{L\rightarrow L}+Y^{H\rightarrow L}\\W^H=[W^{L\rightarrow H},W^{H\rightarrow H}],\ W^L=[W^{L\rightarrow L},W^{H\rightarrow L}]
  $$
  **Octave Conv**实现图如下：

  <div align=center><img src="images\image-20211024170433143.png" alt="image-20211024170433143" style="zoom:50%;" /></div>

  - 其中绿色箭头表示信息更新，红色箭头表示频间通信
  - 其中卷积核的图片如下：

  <div align=center><img src="images\image-20211024171054252.png" alt="image-20211024171054252" style="zoom:50%;" /></div>

#### 2.3 Details

- 在initial layer中，忽视了$X^L$，只用$X^H$作为输入进行计算，即：
  $$
  Y^H = f^{H\rightarrow H}(X^H)\\
  Y^L=f^{H\rightarrow L}(X^L)
  $$

- 在last layer中，忽视了$Y^L$，只计算$Y^H$(因为$Y^H$是Octave UNet生成的血管概率图，此时输入有$X^H \& X^L$)

- **kernel_size**：3；**stride**：1

- 最后一层用sigmoid激活（为了classification），其他层都用ReLU；所有层后面都有跟BatchNormalization

### 3 Experiment

#### 3.1 采用的评判方法及其计算

<div align=center><img src="images\image-20211017212724555.png" alt="image-20211017212724555" style="zoom: 50%;" /></div>

- 精确率：
  $$
  \frac{TP}{TP+TN}
  $$

- 查全率（召回率）：
  $$
  \frac{TP}{TP+FN}
  $$

- **SE**(灵敏度/真阳率)：
  $$
  \frac{TP}{TP+FN}
  $$
  
- **SP**(特异度/假阳率)：
  $$
  \frac{TN}{TN+FP}
  $$
  
- **F1**：（用来权衡精确率和查全率）
  $$
  \frac{2\times精确率\times查全率}{精确率+查全率}=\frac{2\times TP}{2\times TP+FP+FN}
  $$

- **ROC**：（横坐标为SP，纵坐标为SE，也是用来权衡SP和SE）

<div align=center><img src="images\image-20211021205516430.png" alt="image-20211021205516430" style="zoom: 50%;" /></div>

- **AUROC**(AUC)：(ROC下方的面积)
- **MCC**：马修斯系数，取值在[-1,1]之间，1时表示完美预测，0时表示预测的结果不如随机预测，-1是指预测分类和实际分类完全不一致

#### 3.2 复现与结果

- 本次复现中，通过`DRIVE`数据集作为训练集，测试了$\alpha=0$和$\alpha=0.5$的效果，其中$\alpha=0.5$是论文中推荐的最佳设置(即高低频率各一半的OctaveUNet网络架构)，$\alpha=0$为只有高频的标准UNet网络，由于作者在程序中设置了随机数种子，最后得出的实验结果与精度与论文中一致，评优参数表如下：

<div align=center><img src="images\image-20211024171440837.png" alt="image-20211024171440837" style="zoom:50%;" /></div>



## 二、ShuffleNet-V1

- 此网络目的和**MobileNet**一致，都在于减少网络计算复杂度，帮助网络能在移动设备等计算能力有限的设备上，获得更高的精确度；
- 文中提出在以前具有bottleneck 结构的Conv Layer 中，1$\times$1的Conv cell 占据了计算量中的大部分，本文中将1$\times$1 Conv Cell改进为 1$\times$1 GConv Cell来减少计算量；
- 利用**Group Conv** 减少计算的参数量， 通过**Channel Shuffle** 增强Channels 之间的相关性，提升网络的性能，提升模型精确度。

### Channel Shuffle

<div align=center><img src="images\image-20211023205351567.png" alt="image-20211023205351567" style="zoom:50%;" /></div>

- 图(a)是原始的GroupConv，可以看出channels 之间没有相关性
- 图(b)将GConv 提取的feature 进行分组, 进而到图(c)通过Channel Shuffle 将分好组的feature 重新排序

### Group Conv

- 回忆[MobileNet](https://github.com/PercivalYang/Study-Record/tree/main/Week_1#depthwise-separable-convolution)中的DW卷积，Group Conv 与 标准 Conv的比较图如下：

<div align=center><img src="images\image-20211023210908863.png" alt="image-20211023210908863" style="zoom: 33%;" /></div>

- 可见Group Conv 先比标准卷积，参数缩少至其 $\frac{1}{g}$

### ShuffleNet Unit<a name="Shuffle Unit"></a>

<div align=center><img src="images\image-20211023211409600.png" alt="image-20211023211409600" style="zoom:50%;" /></div>

- 图(a)：原始的带有DW卷积的Bottleneck；
- 图(b)<a name="图b"></a>：加入channel shuffle 和 GConv 的Bottleneck；
- 图(c)<a name="图c"></a>：在图(b)的基础上，引入downsample；

### ShuffleNet Architecture

<div align=center><img src="images\image-20211023212044546.png" alt="image-20211023212044546" style="zoom: 50%;" /></div>

- **Stride**：
  - Stage 中Stride=1 对应[Shuffle Unit](#Shuffle Unit)中的图(b)
  - Stage 中Stride=2 对应[Shuffle Unit](#Shuffle Unit)中的图(c)
- 网络第一层 Conv1 是标准的1$\times$1 Conv，因为此时`input_channel = 3 & output_channel=24`，Group Conv 在较大的channels 里作用才明显
- 论文中的实验基本都采用`g=3`

## EfficienNet-V1

- 在**MobileNet-V3**之后发布，其**创新点**为同时探索输入的：分辨率(Resolution)、网络深度(Depth)、网络宽度(Width/Channels)

  - **Resolution**：
    - **优点**：分辨率越高能获得越高细粒度的特征模板，网络学习的信息量就越多；
    - **缺点**：加大计算成本，同时分辨率太高，其能获得的网络准确率的增益也会减少。
  - **Depth**：
    - **优点**：增减网络深度能获得更加丰富、复杂的feature maps；
    - **缺点**：网络深度过深，可能会面临梯度消失、训练困难等问题。
  - **Width/Channels**：
    - **优点：**获得更高细粒度的Feature maps
    - **缺点**：width 很大但depth 不足的网络，很难学到图像深层次的特征

- **EfficienNet-B0**网络架构：<a name="B0"></a>

  <div align=center><img src="images\image-20211024185935453.png" alt="image-20211024185935453" style="zoom:50%;" /></div>

  - 表中MBConv6 表示该block 重复了6次，如果其stride=2，表示仅在该Stage 的第一个blok stride=2，其他block stride=1；

  - EfficienNet 使用的**MBConv** 和之前学习的MobileNet-V3 block 基本一致，如下图所示：

  - <div align=center><img src="images\image-20211024190207559.png" alt="image-20211024190207559" style="zoom:50%;" /></div>

    - 其中SE模块如下图所示：

      <div align=center><img src="images\image-20211024190819749.png" alt="image-20211024190819749" style="zoom:50%;" /></div>

    - FC1、FC2是两个全连接层，第一个全连接层的input features 是输入block 的feature map's channel 的四分之一，后跟siwsh 激活函数(**Pytorch中swish 的函数**为[`torch.nn.SiLU`](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU))

- EfficienNet的扩展：
- <div align=center><img src="images\image-20211024191444194.png" alt="image-20211024191444194" style="zoom: 50%;" /></div>
  
  - **width_coefficient** 即width 的倍率因子，例如在[B0结构](#B0)的Stage 1 中，其3$\times$3卷积核的Channels 为32，那么在B6 中就为$32\times 1.8=57.6$，取离它最近的**8的整数倍**，即为$56$，其他Stage 同理；
    - 取8的整数倍是源码中的设定，其目的是为了加速卷积计算的速度；
  - **depth_coefficient**的取整和**width_coefficient**不同，其为**向上取整**。

# 程序篇

## [Module Hooks](https://pytorch.org/docs/stable/notes/modules.html?highlight=hook#id9)

- 作用：
  - 在`forward` 下，检查每层Layer 的输入和输出
  - 在`backward` 下，检查每层Layer 输入和输出的 grad

### Hooks for tensor

- **用法**：常用来观察反向传播的中间变量的grad（因为反向传播中只有`leaf Tensor`的`grad`才会被保存，如下图的前向传播中，`x,y,w`为`leaf Tensor`，`z,o`为中间变量）

  - <div align=center><img src="images\image-20211022171515827.png" alt="image-20211022171515827" style="zoom:50%;" /></div>

- 示例代码：

  ```python
  import torch
  
  x = torch.Tensor([0, 1, 2, 3]).requires_grad_()
  y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
  w = torch.Tensor([1, 2, 3, 4]).requires_grad_()
  z = x+y
  
  # ===================
  z.register_hook(lambda x: 2*x)	# register_hook(hook_fn)
  z.register_hook(lambda x: print(x))	# 简单的hook_fn可使用lambda 进行定义
  # ===================
  
  o = w.matmul(z)
  
  print('=====Start backprop=====')
  o.backward()
  print('=====End backprop=====')
  
  print('x.grad:', x.grad)
  print('y.grad:', y.grad)
  print('w.grad:', w.grad)
  print('z.grad:', z.grad)
  ```

### Hooks for Module

- [`register_forward_hook(hook_fn)`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)：在module中注册一个`forward hook`, 可以获取每层Layer 的输出

  - 原文解释：`It can modify the input inplace but it will not have effect on forward since this is called after :func:forward is called.`指该函数在程序调用`forward`前不起作用，在调用完之后才起作用

  - 示例代码：

    ```python
    # 定义 forward hook function
    def hook_fn_forward(module, input, output):
        print(module)  # 打印模块名称
        print('input', input)  # 打印本层输入->Tensor
        print('output', output)	# 打印本层输出->Tensor
        total_feat_out.append(output)  # 然后分别存入全局 list 中
        total_feat_in.append(input)
    
    model = Model() # 假设定义了一个简单网络架构Model()
    
    modules = model.named_children()  # 返回iter,里面包含modules的 name和module
    for name, module in modules:
        module.register_forward_hook(hook_fn_forward)
        # module.register_backward_hook(hook_fn_backward)
    ```

- [`module.register_full_backward_hook(hook_fn)`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook)

  - 使用方法和`foward`模式相似，其函数签名为`hook_fn(module, grad_input, grad_output) -> Tensor or None`，需要注意的是
    - `grad_input` 和`grad_output`是在foward 视角下的input、output，如 $o=W\times x+b$，输入端则是$W,x,b$，输出端为$o$
  - 和`forward hook`的不同：
    - `forward hook` 的input只有$x$，没有$W,b$
    - `backward hook` 不能直接改变输入量，但可以返回新的grad_input，反向传播到前一层

## 图像二值化

- 输入OctaveUNet的是RGB图片，其Tensor格式为[4,3,565,584] (batch size=4)，网络输出的图片格式为[4,1,512,512] (Tensor格式)

- `def get_binary_maps(self, prob_maps):` prob_maps为网络输出图片经`sigmoid`

  - `'constant'`模式：

    ```python
    """
    	Args:
    		threshold = {ndarry:{4,}} [0.5, 0.5, 0.5, 0.5] ：以0.5为中间值，进行二值划分
    		num_samples = {int} 4  : prob_maps的 batchsize
    """
    binary_maps = (prob_maps > np.reshape(threshold_values, (num_samples, 1, 1))).astype(float)
    ```

  - **技巧**：`threshold`设置为4维对应 prob_maps 的batch大小

- 检查是否为二值图像：

  ```python
  def check_binary_map(binary_map):
      """Check integrity of binary_map data."""
      if isinstance(binary_map, torch.Tensor):
          check_unique_fn = torch.unique
      elif isinstance(binary_map, np.ndarray):
          check_unique_fn = np.unique
  	# 检测数值矩阵中是否有0 and 1
      cond_a = bool(binary_map.min() == 0)
      cond_b = bool(binary_map.max() == 1)
      cond_c = bool(len(check_unique_fn(binary_map)) == 2)
  	# 检测数值矩阵中是否有0 or 1
      cond_d = bool(len(check_unique_fn(binary_map)) == 1)
      cond_e = bool(binary_map.min() == 1) # 因为仅有0，1中的一个数值，只能最大为1或最小为0
      cond_f = bool(binary_map.max() == 0)
  
      case_a = bool(cond_a and cond_b and cond_c)
      case_b = bool(cond_d and (cond_e or cond_f))
  
      if case_a is True or case_b is True:
          return True
  
      return False
  ```

