from matplotlib import pyplot as plt

from modules import UNet
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm
from torch import nn


def corrupt(x: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """Corrupts the input tensor by adding noise to it.
        @param x: The input tensor.
        @param amount: The amount of noise to add.
        @return: The corrupted tensor.
    """
    noise = torch.randn_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount


dataset = datasets.MNIST(root="mnist/", train=True, download=True, transform=transforms.ToTensor())

batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10
image_size = 28
channels = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(image_size, dim_mults=(1, 2, 4), channels=channels)
model.to(device)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for epoch in range(num_epochs):
    for batch in tqdm.tqdm(dataloader):
        optimizer.zero_grad()

        images, _ = batch
        images = images.to(device)
        noise_amount = torch.rand(images.shape[0]).to(device)
        corrupted_images = corrupt(images, noise_amount)

        out = model(corrupted_images, torch.zeros(images.shape[0], 1, 1, 1).to(device))

        loss = loss_fn(out, images)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    avg_loss = sum(losses[-len(dataloader):]) / len(dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

# View the loss curve
plt.plot(losses)
plt.ylim(0, 0.1)