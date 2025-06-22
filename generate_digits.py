import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import os

# Generator architecture must match training
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = torch.nn.Embedding(10, 10)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(100 + 10, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 784),
            torch.nn.Tanh()
        )
    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.model(x).view(-1, 1, 28, 28)

# Load generator
device = torch.device("cpu")
G = Generator().to(device)
G.load_state_dict(torch.load("models/generator.pth", map_location=device))
G.eval()

def generate_digit_images(digit, n=5):
    z = torch.randn(n, 100).to(device)
    labels = torch.full((n,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        imgs = G(z, labels)
    grid = make_grid((imgs + 1) / 2, nrow=n, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy() * 255
    return Image.fromarray(np_img.astype(np.uint8))
