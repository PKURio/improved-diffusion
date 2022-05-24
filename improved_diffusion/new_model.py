import numpy as np
import torch
from torch import nn
from . import gaussian_diffusion as gd
from . import dist_util

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.dense(x)[..., None]


class DiffusionNet(nn.Module):
    """
    the only difference between ScoreNet and DiffusionNet is class embedding
    """

    def __init__(self, channels=[8, 16, 32, 64, 128], embed_dim=256, noise_schedule="linear", diffusion_steps=1000):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.linear = nn.Linear(2074, 512, bias=False)
        self.tlinear = nn.Linear(512, 2074, bias=False)
        self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        self.class_embed = nn.Embedding(100, embed_dim)
        self.conv1 = nn.Conv1d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv1d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv1d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv1d(channels[3], channels[4], 3, stride=2, padding=1, bias=False)
        self.tconv4 = nn.ConvTranspose1d(channels[4], channels[3], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.tconv3 = nn.ConvTranspose1d(2 * channels[3], channels[2], 3, stride=2, padding=1, output_padding=1,
                                         bias=False)
        self.tconv2 = nn.ConvTranspose1d(2 * channels[2], channels[1], 3, stride=2, padding=1, output_padding=1,
                                         bias=False)
        self.tconv1 = nn.ConvTranspose1d(2 * channels[1], channels[0], 3, stride=2, padding=1, output_padding=1,
                                         bias=False)

        self.dense1 = Dense(embed_dim, channels[1])
        self.dense2 = Dense(embed_dim, channels[2])
        self.dense3 = Dense(embed_dim, channels[3])
        self.dense4 = Dense(embed_dim, channels[4])
        self.dense5 = Dense(embed_dim, channels[3])
        self.dense6 = Dense(embed_dim, channels[2])
        self.dense7 = Dense(embed_dim, channels[1])

        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[1])
        self.gnorm2 = nn.GroupNorm(16, num_channels=channels[2])
        self.gnorm3 = nn.GroupNorm(16, num_channels=channels[3])
        self.gnorm4 = nn.GroupNorm(16, num_channels=channels[4])
        self.tgnorm4 = nn.GroupNorm(16, num_channels=channels[3])
        self.tgnorm3 = nn.GroupNorm(16, num_channels=channels[2])
        self.tgnorm2 = nn.GroupNorm(16, num_channels=channels[1])

        self.act = lambda x: x * torch.sigmoid(x)
        betas = torch.Tensor(gd.get_named_beta_schedule(noise_schedule, diffusion_steps))
        self.betas = betas

    def forward(self, x, t, c):
        x = self.linear(x)
        x = x.view(-1, 8, 64)
        embed = self.act(self.time_embed(t)) + self.class_embed(c)
        ## encoding path
        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))
        ## decoding path
        h = self.tconv4(h4) + self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(torch.cat([h, h1], dim=1))
        h = h.view(-1, 512)
        h = self.tlinear(h)
        ## normalize
        h = h / torch.sqrt(self.betas[t.long(),None].to(dist_util.dev()))
        return h
