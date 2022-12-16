from typing import Tuple

from matplotlib import pyplot as plt

from modules import UNet
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
import tqdm
from torch import nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alpha = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha = alpha / alpha[0]
    beta = 1 - (alpha[1:] / alpha[:-1])
    return torch.clip(beta, 0.0001, 0.9999)


timesteps = 300

# define beta schedule
betas = cosine_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extracts the value of a at time t, reshaping it to match x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise=None):
    """Sample from q(x_t | x_0) at time t."""
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)

    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def train(model: nn.Module, dataloader: DataLoader, epochs: int, optimizer: torch.optim.Optimizer,
          device: torch.device):
    model.train()
    for epoch in range(epochs):
        for batch in tqdm.tqdm(dataloader):
            optimizer.zero_grad()
            x, _ = batch
            x.to(device)

            time = torch.randint(0, timesteps, (x.shape[0],), device=device).long()

            # p_loss function
            noise = torch.randn_like(x)
            noisy_x = q_sample(x, time, noise)
            predicted_noise = model(noisy_x, time)

            loss = F.mse_loss(predicted_noise, noise)

            loss.backward()
            optimizer.step()

# if __name__ == "__main__":
#     dataset = datasets.MNIST(root="mnist/", train=True, download=True, transform=transforms.ToTensor())
#
#     batch_size = 128
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     num_epochs = 10
#     image_size = 28
#     channels = 1
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     model = UNet(image_size, dim_mults=(1, 2, 4), channels=channels)
#
#     loss_fn = nn.MSELoss()
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     # train(model, dataloader, loss_fn, optimizer, device, num_epochs)
#
#     sample(model, device, 40)
