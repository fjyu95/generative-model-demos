from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from dldemos.VAE.load_celebA import get_dataloader
from dldemos.VAE.model import VAE

# Hyperparameters
n_epochs = 10
kl_weight = 0.00025
lr = 0.005


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)  # L2 loss
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)  # 和标准正态分布的KL散度
    loss = recons_loss + kl_loss * kl_weight
    return loss


def train(device, dataloader, model, ckpt_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)

    begin_time = time()
    print(f'total batches: {len(dataloader)}')
    # train
    for i in range(n_epochs):
        loss_sum = 0
        for i, x in enumerate(dataloader):
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
            if i % 10 == 0:
                print(f'batch {i}: loss {loss}')

        loss_sum /= dataset_len
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {i}: loss {loss_sum} {minute}:{second}')
        torch.save(model.state_dict(), ckpt_path)


# 检验模型对输入数据的编码-解码能力，看还原得像不像
def reconstruct(device, dataloader, model):
    model.eval()
    batch = next(iter(dataloader))  # 取dataloader中的第一个batch，每次会shuffle
    x = batch[0:1, ...].to(device)  # batch中的第一张图片
    output = model(x)[0]
    output = output[0].detach().cpu()

    input = batch[0].detach().cpu()
    combined = torch.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save('work_dirs/recon.jpg')


def generate(device, model):
    model.eval()
    output = model.sample(device)
    output = output[0].detach().cpu()
    img = ToPILImage()(output)
    img.save('work_dirs/generate.jpg')


def main():
    device = 'cuda:0'
    data_dir = Path("../ddim/data/celebA/celeba_hq_256").resolve()  # 任意图片目录，最后都会缩放到同样大小
    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist.")

    dataloader = get_dataloader(data_dir.as_posix())

    model = VAE().to(device)

    # If you obtain the ckpt, load it
    ckpt_path = 'work_dirs/model.pth'
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, device))

    # Choose the function
    # train(device, dataloader, model, ckpt_path)
    reconstruct(device, dataloader, model)
    generate(device, model)


if __name__ == '__main__':
    main()
