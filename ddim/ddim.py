import torch
from tqdm import tqdm

from dldemos.ddim.ddpm import DDPM


class DDIM(DDPM):

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

    def sample_backward(self,
                        img_or_shape,
                        net,
                        device,
                        simple_var=True,
                        ddim_step=20,
                        eta=1):
        if simple_var:
            eta = 1  # DDPM中理论方差
        ts = torch.linspace(self.n_steps, 0, (ddim_step + 1)).to(device).to(torch.long)  # 均匀取用于加速迭代的时刻
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)  # 随机噪声
        batch_size = x.shape[0]
        net = net.to(device)
        for i in tqdm(range(1, ddim_step + 1), f'DDIM sampling with eta {eta} simple_var {simple_var}'):
            cur_t = ts[i - 1] - 1  # 可加速采样
            prev_t = ts[i] - 1

            ab_cur = self.alpha_bars[cur_t]
            ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1

            t_tensor = torch.tensor([cur_t] * batch_size, dtype=torch.long).to(device).unsqueeze(1)  # 为每个batch准备相同的时间步
            eps = net(x, t_tensor)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)  # 可调节的方差
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur) ** 0.5 * x
            second_term = ((1 - ab_prev - var) ** 0.5 - (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev) ** 0.5 * noise  # 原始DDPM beta_t方差（做过变量替换）sigma_hat
            else:
                third_term = var ** 0.5 * noise  # 广义DDPM可调节方差
            x = first_term + second_term + third_term

        return x
