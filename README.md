# Generative Model Demos

一个包含多种生成模型实现的演示项目，包括扩散模型（DDPM、DDIM）和变分自编码器（VAE）。

## 📋 项目概述

本项目实现了三种主要的生成模型：

- **DDPM (Denoising Diffusion Probabilistic Models)** - 经典的扩散概率模型
- **DDIM (Denoising Diffusion Implicit Models)** - 基于DDPM的加速采样方法
- **VAE (Variational Autoencoder)** - 变分自编码器

## 🏗️ 项目结构

```
generative-model-demos/
├── ddim/                    # DDIM实现
│   ├── configs.py          # 配置文件（MNIST、CelebA等）
│   ├── dataset.py          # 数据加载模块
│   ├── ddim.py             # DDIM采样算法
│   ├── ddpm.py             # DDPM基类
│   ├── network.py          # 神经网络架构
│   ├── network_my.py       # 备用网络实现
│   ├── main.py             # 训练和推理主文件
│   ├── dist_train.py       # 分布式训练脚本
│   ├── dist_sample.py      # 分布式采样脚本
│   ├── mnist.pth           # 预训练MNIST模型
│   └── work_dirs/          # 工作目录（保存检查点和结果）
│
├── ddpm/                    # DDPM实现
│   ├── dataset.py          # MNIST数据加载
│   ├── ddpm_simple.py      # 简化版DDPM
│   ├── ddpm.py             # DDPM主实现
│   ├── network.py          # 网络架构定义
│   ├── main.py             # 训练脚本
│   ├── model_unet_res.pth  # 预训练模型
│   ├── data/               # 数据目录
│   └── work_dirs/          # 工作目录
│
├── VAE/                     # VAE实现
│   ├── load_celebA.py      # CelebA数据加载
│   ├── model.py            # VAE模型定义
│   ├── main.py             # 训练脚本
│   ├── result/             # 结果保存目录
│   └── work_dirs/          # 工作目录
│
└── LICENSE, README.md
```

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install torch torchvision einops opencv-python numpy tqdm
```

### DDPM - MNIST图像生成

```bash
cd ddpm
python main.py
```

**参数说明：**
- `batch_size`: 512
- `n_epochs`: 100
- `learning_rate`: 1e-3
- 优化器：Adam
- 损失函数：MSE

**模型说明：**
- 输入：28×28 MNIST图像
- 训练过程：前向加噪→预测噪声
- 输出：生成的数字图像

### DDIM - 加速采样

```bash
cd ddim
python main.py
```

**特性：**
- 支持MNIST和CelebAHQ数据集
- 可配置的加速因子（ddim_step参数）
- 可调节的采样方差（eta参数）
- 支持不同的网络架构

**配置示例（configs.py）：**
```python
# MNIST配置
mnist_cfg = {
    'dataset_type': 'MNIST',
    'img_shape': [1, 28, 28],
    'batch_size': 512,
    'n_epochs': 50,
    'channels': [10, 20, 40, 80],
    'pe_dim': 128
}

# CelebAHQ配置
celebahq_cfg2 = {
    'dataset_type': 'CelebAHQ',
    'img_shape': [3, 64, 64],
    'batch_size': 128,
    'n_epochs': 2500,
    'channels': [64, 128, 256, 512],
}
```

### VAE - CelebA人脸生成

```bash
cd VAE
python main.py
```

**参数说明：**
- 输入分辨率：64×64 CelebA图像
- 隐层维度：[16, 32, 64, 128, 256]
- 隐向量维度：128
- 学习率：0.005
- KL权重：0.00025
- 损失函数：MSE + KL散度

**模型架构：**
- Encoder：5层卷积，每层图像尺寸减半
- Decoder：5层转置卷积，恢复原始尺寸

## 🔑 核心算法说明

### DDPM（扩散概率模型）

1. **前向过程**：逐步向图像添加高斯噪声
   - 给定图像x₀，在时刻t添加噪声：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

2. **反向过程**：训练网络预测添加的噪声
   - 网络目标：$\epsilon_\theta(x_t, t) \approx \epsilon$

3. **采样**：从纯噪声开始，逐步去噪生成图像

### DDIM（隐式扩散模型）

- 基于DDPM但使用**隐式推理**
- 支持**非马尔可夫链**，可跳过中间步骤
- **加速采样**：通过减少时间步（ddim_step）加快生成速度
- **可调节方差**：通过eta参数权衡生成速度和质量

### VAE（变分自编码器）

- Encoder：将输入图像压缩到128维隐向量
- Latent space：遵循标准正态分布
- Decoder：从隐向量重建图像
- 损失函数：重建损失 + KL散度正则化

## 📊 关键特性

### DDIM实现
- ✅ 支持多个数据集（MNIST、CelebAHQ）
- ✅ 可配置的采样加速（ddim_step参数）
- ✅ 灵活的方差选择（eta参数）
- ✅ 分布式训练支持（dist_train.py、dist_sample.py）
- ✅ 多种网络架构选择

### DDPM实现
- ✅ 简化和完整两种版本
- ✅ 使用UNet作为去噪网络
- ✅ 完整的训练管道
- ✅ 预训练模型支持

### VAE实现
- ✅ 编码器-解码器架构
- ✅ 批正则化和激活函数
- ✅ 配置式隐层维度
- ✅ 预留CelebA数据加载接口

## 📈 性能指标

各模型在对应数据集上的训练配置：

| 模型 | 数据集 | 分辨率 | Batch Size | Epochs | 优化器 | 学习率 |
|------|--------|--------|-----------|--------|--------|---------|
| DDPM | MNIST | 28×28 | 512 | 100 | Adam | 1e-3 |
| DDIM | MNIST | 28×28 | 512 | 50 | Adam | 2e-4 |
| DDIM | CelebAHQ | 64×64 | 128 | 2500 | Adam | - |
| VAE | CelebA | 64×64 | - | 10 | Adam | 5e-3 |

## 🛠️ 使用示例

### 基本训练循环

```python
# DDPM训练示例
from ddpm.ddpm_simple import DDPM
from ddpm.network import build_network

ddpm = DDPM(device='cuda', n_steps=1000)
net = build_network('unet_res')
train(ddpm, net, device='cuda', ckpt_path='model.pth')
```

### 推理/采样

```python
# DDIM采样（加速生成）
from ddim.ddim import DDIM

ddim = DDIM(device='cuda', n_steps=1000, eta=1.0)
samples = ddim.sample_backward(
    img_or_shape=(16, 3, 64, 64),  # 生成16张64×64彩色图
    net=pretrained_net,
    device='cuda',
    ddim_step=20,  # 仅用20步加速采样
    eta=0.5  # 可调节方差
)
```

## 📚 依赖

- **PyTorch**: 深度学习框架
- **torchvision**: 图像处理和数据集
- **einops**: 张量操作简化
- **opencv-python**: 图像处理
- **numpy**: 数值计算
- **tqdm**: 进度条显示

## 🎓 学习资源

### 论文参考
- DDPM: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- DDIM: "Denoising Diffusion Implicit Models" (Song et al., 2021)
- VAE: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014)

## 📝 注意事项

1. **数据集**：MNIST自动下载；CelebAHQ需手动准备
2. **GPU内存**：建议使用GPU加速训练，某些模型需要较大显存
3. **预训练模型**：已提供MNIST和部分CelebA的预训练权重
4. **分布式训练**：DDIM模块提供分布式训练脚本支持多GPU

## 📄 许可证

详见[LICENSE](LICENSE)文件

---

**最后更新**: 2026年6月