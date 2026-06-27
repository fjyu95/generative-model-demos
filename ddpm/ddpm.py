import torch


class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product

        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        alpha_prev = torch.empty_like(alpha_bars)  # 高效初始化(内存随机值)
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self,
                        img_shape,
                        net,
                        device,
                        simple_var=True,
                        clip_x0=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True):

        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                        1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        if clip_x0:
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) *
                   eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0
        else:
            mean = (x_t -
                    (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t


def visualize_forward(n_imgs=5, n_steps=100, n_sample_steps=10):
    import cv2
    import einops
    import numpy as np

    from dldemos.ddpm.dataset import get_dataloader

    device = 'cuda'
    dataloader = get_dataloader(n_imgs)  # 每个批次取5张图
    x, lbl = next(iter(dataloader))  # 只取一次数据
    x = x.to(device)

    ddpm = DDPM(device, n_steps)
    xts = []
    percents = torch.linspace(0, 0.99, n_sample_steps)  # 从所有的时间步中抽取了10步？
    for percent in percents:
        t = torch.tensor([int(n_steps * percent)])
        t = t.unsqueeze(1)
        x_t = ddpm.sample_forward(x, t)
        xts.append(x_t)
    res = torch.stack(xts, 0)  # 合并到第0维
    res = einops.rearrange(res, 'n1 n2 c h w -> (n2 h) (n1 w) c')  # 维度重组，维度名称任意
    res = (res.clip(-1, 1) + 1) / 2 * 255  # 变换回0～255
    res = res.cpu().numpy().astype(np.uint8)

    cv2.imwrite(f'work_dirs/diffusion_forward_{n_steps}.jpg', res)


def main():
    visualize_forward(10, 1000, 50)  # 前向加噪过程可视化


if __name__ == '__main__':
    main()
