import torch
import torch.nn as nn


class VAE(nn.Module):
    """VAE for 64x64 face generation.

    The hidden dimensions can be tuned.
    """

    def __init__(self, hiddens=[16, 32, 64, 128, 256], latent_dim=128) -> None:
        super().__init__()

        # encoder
        prev_channels = 3
        modules = []
        img_length = 64  # 输入为64x64的图片
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))  # 每次图像尺寸减半，通道数翻倍
            prev_channels = cur_channels
            img_length //= 2  # 特征图像尺寸
        self.encoder = nn.Sequential(*modules)
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)  # 全连接层，拟合均值
        self.var_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)  # 拟合对数方差
        self.latent_dim = latent_dim  # 128维向量

        # decoder
        modules = []
        self.decoder_projection = nn.Linear(latent_dim, prev_channels * img_length * img_length)  # 反向，类似UNet结构
        self.decoder_input_chw = (prev_channels, img_length, img_length)
        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]), nn.ReLU()))  # 转置卷积
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]), nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()))  # 从16通道恢复到彩色3通道
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoded = self.encoder(x)  # 将原始图像编码为特征图
        encoded = torch.flatten(encoded, 1)  # 从维度1开始展平，（batch_size, 通道数*图像尺寸*图像尺寸）
        mean = self.mean_linear(encoded)  # (batch_size, latent_dim)
        logvar = self.var_linear(encoded)  # 均值+方差，拟合高维高斯分布
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean  # 重采样技巧，生成潜在变量z
        x = self.decoder_projection(z)  # 隐变量恢复为特征图维度
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)

        return decoded, mean, logvar

    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)  # 随机采样
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded
