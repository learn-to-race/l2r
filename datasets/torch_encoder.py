"""
https://github.com/sksq96/pytorch-vae
"""
import copy
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=20736):
        return input.view(input.size(0), size, 1, 1)

class ReshapeImg(nn.Module):
	def forward(self, input, im_w=144, im_h=144):
		return input.view(input.size(0), im_w, im_h, 3)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=20736, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=3),
            nn.Sigmoid(),
            ReshapeImg()
        )

        self.optimizer = torch.optim.Adam(self.parameters())
        self.dataloader = None
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def encode(self, x):
    	h = self.encoder(x)
    	z, mu, logvar = self.bottleneck(h)
    	return z, mu, logvar

    def decode(self, z):
    	z = self.fc3(z)
    	return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss(self, recon, actual, mu, logvar):
    	bce = F.binary_cross_entropy(recon, actual, size_average=False)
    	kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    	return bce + kld

    def train(self, num_epochs):
        if not self.dataloader:
            raise Exception('VAE has no data!')

        _shape = (self.batch_size, 3, self.im_w, self.im_h)

        for epoch in range(num_epochs):
            for idx, _images in enumerate(self.dataloader):
                images = _images[-1].to(device)
                image_hats, mu, logvar = self.forward(images.view(_shape))
                loss = self.loss(image_hats, images, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            print(f'[{epoch+1} of {num_epochs}] Loss: {loss.item()/self.batch_size}')

    def test(self):
    	raise NotImplementedError

    def load_images(self, path, im_w, im_h, shuffle=True, batch_size=32):
        """Utility to load images
        """
        self.im_w, self.im_h = im_w, im_h
        self.batch_size = batch_size
        
        images = []
        for index, file in enumerate(glob.glob(path + "*.png")):
            if len(images) == 4992: break
            image = cv2.imread(file)
            images.append(cv2.resize(image, (im_w, im_h)))

        images = np.array(images).astype("float32") / 255
        data = TensorDataset(torch.from_numpy(images))
        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        self.test_img = images[0]
