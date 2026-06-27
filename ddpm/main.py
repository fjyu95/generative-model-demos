import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from dldemos.ddpm.dataset import get_dataloader, get_img_shape
from dldemos.ddpm.ddpm_simple import DDPM
from dldemos.ddpm.network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)

batch_size = 512
n_epochs = 100


def train(ddpm: DDPM, net, device='cuda', ckpt_path='dldemos/ddpm/model.pth'):
    print('batch size:', batch_size)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)  # 下载数据并构造读取器
    net = net.to(device)  # 默认UNet架构
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x, lbl in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)  # 随机生成要训练的时刻t [low, high)
            eps = torch.randn_like(x).to(device)  # 随机噪声
            x_t = ddpm.sample_forward(x, t, eps)  # 前向过程，加噪声
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))  # 给定噪声图像（服从高斯分布）和时刻t，预测所加噪声
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample ** 0.5))  # 数据重组

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)

    n_steps = 1000
    config_id = 4
    device = 'cuda'
    # model_path = '../../dldemos/ddpm/model_unet_res.pth'
    model_path = 'model_unet_res.pth'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    finetune = True
    if not model_path or finetune:
        train(ddpm, net, device=device, ckpt_path=model_path)

    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, 'work_dirs/diffusion.jpg', device=device)
