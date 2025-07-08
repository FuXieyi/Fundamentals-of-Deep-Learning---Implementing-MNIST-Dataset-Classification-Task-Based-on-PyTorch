## 零、导包

不得不提， [PyTorch] 是目前深度学习领域非常火的框架之一（其实在我心中，没有之一）。今天我以 **[MNIST数据集]分类** 为任务，为大家演示 **PyTorch** 的基本使用；并且，文末为大家提 **供粘贴可运行** 的代码。 **如有错误，还请指正。** 在开始之前，首先需要介绍 PyTorch 在计算机视觉领域中最重要的两个包： **torch** 与 **torchvision：**

**torch** 是 PyTorch 的核心库，提供了张量计算、自动微分等功能。

**torchvision** 是 PyTorch 的一个独立子库，主要用于计算机视觉任务，包括图像处理、数据加载、数据增强、预训练模型等。

```python
import torch  # pytorch 1.12.1
from torch import nn  # neural network 模块
from torch import optim  # 参数优化模块
from torch.utils.data import DataLoader  # 数据加载器模块

import torchvision
from torchvision import datasets  # 计算机视觉常用的数据集模块
from torchvision import transforms  # 数据增强模块
```

目前来看，采用 AI 解决任何一个任务往往都需要这几个步骤：定义模型、训练模型、测试模型。

## 一、模型定义

针对 MNIST 数据集的分类任务，可采用经典的 AlexNet 模型，并使用 PyTorch 带有的预训练模型参数，初始化模型。

```python
# 自定义神经网络模型类，需要继承自torch.nn.Module
class MyAlexNet(nn.Module):

    def __init__(self, class_num):

        """
        :param class_num: 分类任务的类别数目
        """

        super().__init__()
        # 采用torchvision库中已有的alexnet模型
        self.net = torchvision.models.alexnet(
            # 使用PyTorch附带的预训练参数
            pretrained=True,
            # dropout防止过拟合
            dropout=0.3
        )

        # 修改AlexNet模型的分类器，主要修改分类任务的类目数量
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

    # 定义模型的前向传播函数
    def forward(self, inputs):

        """
        :param inputs: 模型的输入
        :return: 模型的输出
        """

        outputs = self.net(inputs)
        return outputs

# 得到10分类的AlexNet模型对象
alexnet = MyAlexNet(10)
```

## 二、模型训练

### 2.1 数据增强&数据集&数据加载器

数据增强，即 transform，一方面用于增加训练的数据量，提高模型的泛化能力，提升模型的鲁棒性；另一方面用于对样本进行处理，使其符合模型的输入。

```python
transform = transforms.Compose([
    # 修改图片尺寸
    transforms.Resize((96, 96)),
    # 将单通道的MNIST图片升为三通道
    transforms.Grayscale(num_output_channels=3),
    # 转换为张量
    transforms.ToTensor(),
    # 对数据进行归一化，有利于提升模型拟合能力
    transforms.Normalize((0.2, 0.2, 0.2,), (0.3, 0.3, 0.3,))
])
```

数据集，即 dataset，主要分为训练集与测试集。PyTorch 自带有 MNIST 数据集；该数据集是一个手写体数字的图片数据集，该图片为 $28\times2828\times28$ 单通道黑白图片，训练集一共包含了 60,000 张样本，测试集一共包含了 10,000 张样本。

```python
train_set = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
```

数据加载器，即 data loader，是 PyTorch 为了方便模型训练与数据集加载提供的工具类；data loader 会将数据集中样本根据 batch\_size 分割成一个一个的 mini-batch，方便后序的训练与测试过程（目前的训练与测试采用的大都是 mini-batch 的方式）。

```python
train_loader = DataLoader(dataset=train_set, batch_size=512, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=1024, shuffle=True, drop_last=False)
```

### 2.2 loss 函数

分类任务，毫无疑问， [交叉熵损失函数](https://zhida.zhihu.com/search?content_id=213087946&content_type=Article&match_order=1&q=%E4%BA%A4%E5%8F%89%E7%86%B5%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0&zhida_source=entity) 。根据模型的输出与正确结果，计算损失，同时可以根据自动微分功能，实现由 loss 函数而始的梯度反向传播。不得不说，通常说的模型，往往指的是参数模型，并不包含梯度；反向传播，传播的是模型的梯度。

```python
criterion = nn.CrossEntropyLoss()
```

### 2.3 优化器及学习率

优化器，即 optimizer，实现模型参数的更新。主要利用反向传播而来的梯度，以及采用梯度下降的方式，实现参数的更新： $\theta_n=\theta_{n-1}-g*lr\theta_n=\theta_{n-1}-g*lr$ 。其中 $\theta_{n-1}$ 代指模型当前参数； $g$ 代指参数的梯度； $lr$ 代指学习率，往往与学模型学习过程快慢有关； $\theta_n$ 代指优化器进行参数更新后的模型参数。

```python
# 优化器，传入模型的参数与学习率
optimizer = optim.AdamW(params=alexnet.parameters(), lr=0.0001)
```

### 2.4 训练

训练 (有监督训练) 是指将训练集输入到算法模型中，根据模型输出与正确标签计算损失，并通过反向传播与梯度下降的方式，对模型参数不断优化，使模型能够识别、分析和预测各种情况。注：最终得到的是模型的 **参数** 。

```python
# 模型训练
def train_model(data_loader: DataLoader, model: nn.Module, param_optimizer: optim.Optimizer, loss_criterion: nn.Module, device_: torch.device, epoch, print_freq):

    """
    :param print_freq: 打印频率
    :param epoch: 训练轮次
    :param data_loader: 训练数据集
    :param model: 待训练模型
    :param param_optimizer: 模型优化器
    :param loss_criterion: 计算loss的评判器
    :param device_: 使用的计算设备
    :return: 平均loss，平均acc
    """

    # 模型进入训练模式，会有计算图生成与梯度计算
    model.train()
    # 将模型迁移到设备上(CPU or GPU)
    model.to(device_)
    total_loss = 0.
    total_acc = 0.
    batch_num = len(data_loader)
    # 遍历数据加载器，数据加载器可以看作是Mini-Bath的集合
    for idx, (img, target) in enumerate(data_loader, start=1):
        # 将数据也迁移到与模型相同的设备上(CPU or GPU)
        img = img.to(device_)
        target = target.to(device_)
        # 计算训练集loss
        outputs = model(img)
        loss = loss_criterion(outputs, target)
        # 优化器梯度清零
        param_optimizer.zero_grad()
        # 反向传播，得到参数的梯度
        loss.backward()
        # 梯度下降
        param_optimizer.step()
        # 计算训练集准确率
        predict = outputs.max(dim=-1)[-1]
        acc = predict.eq(target).float().mean()

        current_loss = loss.cpu().item()
        current_acc = acc.cpu().item()
        total_loss += current_loss
        total_acc += current_acc
        if idx % print_freq == 0:
            print(f"Epoch:{epoch:03d}  Batch num:[{idx:03d}/{batch_num:03d}]    Loss:{current_loss:.4f}    Acc:{current_acc:.4f}")
    return total_loss / batch_num, total_acc / batch_num
```

## 三、模型测试

在模型训练过程中或者训练结束后，需要使用测试集对模型进行测试；并根据模型在测试集上的推理准确率，评判模型的优劣。

```python
# 模型测试
def test_model(data_loader: DataLoader, model: nn.Module, loss_criterion: nn.Module, device_: torch.device):

    """
    :param data_loader: 测试数据集
    :param model: 待测试模型
    :param loss_criterion: 计算loss的评判器
    :param device_: 使用的计算设备
    :return: 平均loss，平均acc
    """

    # 模型进入评估模式，不会有计算图生成，也不会计算梯度
    model.eval()
    # 将模型迁移到设备上(CPU or GPU)
    model.to(device_)
    total_loss = 0.
    total_acc = 0.
    batch_num = len(data_loader)
    # 遍历数据加载器
    for idx, (img, target) in enumerate(data_loader, start=1):
        # 将数据也迁移到与模型相同的设备上(CPU or GPU)
        img = img.to(device_)
        target = target.to(device_)
        # 计算测试集loss
        with torch.no_grad():
            outputs = model(img)
            loss = loss_criterion(outputs, target)
        # 计算测试集准确率
        predict = outputs.max(dim=-1)[-1]
        acc = predict.eq(target).float().mean()

        current_loss = loss.cpu().item()
        current_acc = acc.cpu().item()
        total_loss += current_loss
        total_acc += current_acc
    return total_loss / batch_num, total_acc / batch_num

test_loss, test_acc = test_model(test_loader, alexnet, criterion, torch.device("cpu"))
print(f"Test-Loss:{test_loss:.4f}    Test-Acc:{test_acc:.4f}")
```

## 四、完整代码

主要对前面代码的总结（ **粘贴可运行** ，前提是环境没问题）。

```python
import torch
from torch import nn  # neural network
from torch import optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载器

import torchvision
from torchvision import datasets  # 数据集
from torchvision import transforms  # 数据正前期

# 自定义神经网络模型类，需要继承自torch.nn.Module
class MyAlexNet(nn.Module):

    def __init__(self, class_num):
        """
        :param class_num: 分类任务的类别数目
        """
        super().__init__()
        # 采用torchvision库中已有的alexnet模型
        self.net = torchvision.models.alexnet(
            # 使用PyTorch附带的预训练参数
            pretrained=True,
            # dropout防止过拟合
            dropout=0.3
        )
        # 修改AlexNet模型的分类器，主要修改分类数量
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num),
        )

    # 定义模型的前向传播函数
    def forward(self, inputs):
        """
        :param inputs: 模型的输入
        :return: 模型的输出
        """
        outputs = self.net(inputs)
        return outputs

# 得到模型实例
alexnet = MyAlexNet(10)

# 数据增强 & 数据集 & 数据加载器
transform = transforms.Compose([
    # 修改图片尺寸
    transforms.Resize((96, 96)),
    # 将单通道的MNIST图片升为三通道
    transforms.Grayscale(num_output_channels=3),
    # 转换为张量
    transforms.ToTensor(),
    # 对数据进行归一化，有利于提升模型拟合能力
    transforms.Normalize((0.2, 0.2, 0.2,), (0.3, 0.3, 0.3,))
])
train_set = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_set, batch_size=512, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=512, shuffle=True, drop_last=False)

# loss函数
criterion = nn.CrossEntropyLoss()

# 优化器，传入模型的参数与学习率
optimizer = optim.AdamW(params=alexnet.parameters(), lr=0.0001)

# 模型训练
def train_model(data_loader: DataLoader, model: nn.Module, param_optimizer: optim.Optimizer, loss_criterion: nn.Module, device_: torch.device, epoch, print_freq):
    """
    :param print_freq: 打印频率
    :param epoch: 训练轮次
    :param data_loader: 训练数据集
    :param model: 待训练模型
    :param param_optimizer: 模型优化器
    :param loss_criterion: 计算loss的评判器
    :param device_: 使用的计算设备
    :return: 平均loss，平均acc
    """
    # 模型进入训练模式，会有计算图生成与梯度计算
    model.train()
    # 将模型迁移到设备上(CPU or GPU)
    model.to(device_)
    total_loss = 0.
    total_acc = 0.
    batch_num = len(data_loader)
    # 遍历数据加载器，数据加载器可以看作是Mini-Bath的集合
    for idx, (img, target) in enumerate(data_loader, start=1):
        # 将数据也迁移到与模型相同的设备上(CPU or GPU)
        img = img.to(device_)
        target = target.to(device_)
        # 计算训练集loss
        outputs = model(img)
        loss = loss_criterion(outputs, target)
        # 优化器梯度清零
        param_optimizer.zero_grad()
        # 反向传播，得到参数的梯度
        loss.backward()
        # 梯度下降
        param_optimizer.step()
        # 计算训练集准确率
        predict = outputs.max(dim=-1)[-1]
        acc = predict.eq(target).float().mean()

        current_loss = loss.cpu().item()
        current_acc = acc.cpu().item()
        total_loss += current_loss
        total_acc += current_acc
        if idx % print_freq == 0:
            print(f"Epoch:{epoch:03d}  Batch num:[{idx:03d}/{batch_num:03d}]    Loss:{current_loss:.4f}    Acc:{current_acc:.4f}")
    return total_loss / batch_num, total_acc / batch_num

# 模型测试
def test_model(data_loader: DataLoader, model: nn.Module, loss_criterion: nn.Module, device_: torch.device):
    """
    :param data_loader: 测试数据集
    :param model: 待测试模型
    :param loss_criterion: 计算loss的评判器
    :param device_: 使用的计算设备
    :return: 平均loss，平均acc
    """
    # 模型进入评估模式，不会有计算图生成，也不会计算梯度
    model.eval()
    # 将模型迁移到设备上(CPU or GPU)
    model.to(device_)
    total_loss = 0.
    total_acc = 0.
    batch_num = len(data_loader)
    # 遍历数据加载器
    for idx, (img, target) in enumerate(data_loader, start=1):
        # 将数据也迁移到与模型相同的设备上(CPU or GPU)
        img = img.to(device_)
        target = target.to(device_)
        # 计算测试集loss
        with torch.no_grad():
            outputs = model(img)
            loss = loss_criterion(outputs, target)
        # 计算测试集准确率
        predict = outputs.max(dim=-1)[-1]
        acc = predict.eq(target).float().mean()

        current_loss = loss.cpu().item()
        current_acc = acc.cpu().item()
        total_loss += current_loss
        total_acc += current_acc
    return total_loss / batch_num, total_acc / batch_num

# 训练5轮
for i in range(5):
    train_model(train_loader, alexnet, optimizer, criterion, torch.device("cpu"), i + 1, 16)
    test_loss, test_acc = test_model(test_loader, alexnet, criterion, torch.device("cpu"))
    print(f"Epoch:{i + 1:03d}  Test-Loss:{test_loss:.4f}    Test-Acc:{test_acc:.4f}")

# 保存模型参数
torch.save(alexnet.state_dict(), "alexnet_weight.pth")
```
