from typing import Tuple, List

import numpy
import torch
import torch.nn.functional as F
import torchvision.transforms
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from modules import UNet


def cosine_beta_schedule(time_steps: int, s: float = 0.008) -> torch.Tensor:
    """Computes the beta schedule for cosine annealing."""
    steps = time_steps + 1
    x = torch.linspace(0, time_steps, steps)
    alpha = torch.cos(((x / time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha = alpha / alpha[0]
    beta = 1 - (alpha[1:] / alpha[:-1])
    return torch.clip(beta, 0.0001, 0.9999)


timesteps = 1000

betas = cosine_beta_schedule(time_steps=timesteps)

alphas = 1. - betas
# define alphas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(x: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extracts the value of x at time t, reshaping it to match x_shape."""
    batch_size = t.shape[0]
    out = x.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def add_noise_step(x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
    """Next step of diffusion process. Adds noise to x_start at time t."""
    if noise is None:
        noise = torch.randn_like(x_start, device=x_start.device)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)

    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


@torch.no_grad()
def reduce_noise_step(model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x, device=x.device)

        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def reduce_noise_loop(model: nn.Module, shape: Tuple[int, ...]) -> List[numpy.ndarray]:
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm.tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = reduce_noise_step(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().detach().numpy())
    return imgs


def train(model: nn.Module, dataloader: DataLoader, epochs: int, optimizer: torch.optim.Optimizer,
          device: torch.device) -> None:
    model.train()
    for epoch in range(epochs):
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch number {epoch}"):
            optimizer.zero_grad()
            x, _ = batch
            x = x.to(device)

            time = torch.randint(0, timesteps, (x.shape[0],), device=device).long()

            # p_loss function
            noise = torch.randn_like(x, device=device)
            noisy_x = add_noise_step(x, time, noise)
            predicted_noise = model(noisy_x, time)

            loss = F.mse_loss(predicted_noise, noise)

            loss.backward()
            optimizer.step()


def sample(model: nn.Module, image_size: int, batch_size: int = 16, channels: int = 3) -> List[numpy.ndarray]:
    return reduce_noise_loop(model, shape=(batch_size, channels, image_size, image_size))


if __name__ == "__main__":
    dataset = torchvision.datasets.MNIST("mnist/", download=True, transform=torchvision.transforms.ToTensor())
    num_epochs = 30
    channels = 1
    image_size = 28
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    u_net_model = UNet(image_size, channels=channels, dim_mults=(2, 4, 8), resnet_block_groups=14)
    u_net_model = u_net_model.to(device)

    optimizer = torch.optim.Adam(u_net_model.parameters(), lr=1e-3)

    train(u_net_model, data_loader, num_epochs, optimizer, device)

    images = sample(u_net_model, image_size, batch_size, channels)