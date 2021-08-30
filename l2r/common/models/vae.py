"""
This is a slight modification of Shubham Chandel's implementation of a
variational autoencoder in PyTorch.

Source: https://github.com/sksq96/pytorch-vae
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


class VAE(nn.Module):
    """Expects input of (batch_size, C=3, H=144, W=144)
    """
    def __init__(self, im_w=144, im_h=144, image_channels=3, z_dim=32):
        super(VAE, self).__init__()

        self.im_w = im_w
        self.im_h = im_h

        encoder_list = [
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        ]

        self.encoder = nn.Sequential(*encoder_list)
        sample_input = torch.zeros([1, image_channels, im_h, im_w])
        em_shape = nn.Sequential(*encoder_list[:-1])(sample_input).shape[1:]
        h_dim = np.prod(em_shape)

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, em_shape),
            nn.ConvTranspose2d(em_shape[0], 128, kernel_size=4, stride=2, padding=1, output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=mu.device)
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

    def encode_raw(self, x, device):
        p = np.zeros([x.shape[0], 144, 144, 3])
        for i in range(x.shape[0]):
            p[i] = cv2.resize(x[i], (144, 144)) / 255.
        x = p.transpose(0, 3, 1, 2)
        x = torch.as_tensor(x, device=device, dtype=torch.float)
        v = self.representation(x)
        return v.detach().cpu().numpy()

    def decode(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        # expects (N, C, H, W)
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss(self, actual, recon, mu, logvar, kld_weight=1.):
        bce = F.binary_cross_entropy(recon, actual, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return bce + kld * kld_weight


if __name__ == '__main__':
    files_paths = ['/home/jimmy/track_imgs/anglesey', '/home/jimmy/track_imgs/thruxton']

    file_list = []
    for path in files_paths:
        for root, _, files in os.walk(path):
            for file in files:
                file_list.append(os.path.join(root, file))

    print('Loading images')
    imgs = [np.load(file)['image'] for file in file_list]
    imgs = np.array(imgs)
    print('Complete')

    # parameters
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    bsz = 64
    lr = 1e-3
    num_epochs = 1000
    best_loss = 1e10

    # training setup
    n = imgs.shape[0]
    indices = np.random.permutation(n)
    thres = int(n * 0.9)
    train_indices, test_indices = indices[:thres], indices[thres:]
    vae = VAE().to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_indices = np.random.permutation(train_indices)
        test_indices = np.random.permutation(test_indices)

        # train
        train_loss = []
        vae.train()
        for i in tqdm.trange(len(train_indices) // bsz, desc=f"Epoch #{epoch + 1} train"):

            # preprocess
            index = train_indices[bsz * i: bsz * (i + 1)]
            img = imgs[index] / 255.
            p = np.zeros([img.shape[0], 144, 144, 3])
            for i in range(img.shape[0]):
                p[i] = cv2.resize(img[i], (144, 144))
            img = p.transpose(0, 3, 1, 2)
            img = torch.as_tensor(img, device=device, dtype=torch.float)

            # compute loss
            loss = vae.loss(img, *vae(img))
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        # eval
        test_loss = []
        vae.eval()
        for i in tqdm.trange(len(test_indices) // bsz, desc=f"Epoch #{epoch + 1} test"):

            # preprocess
            index = test_indices[bsz * i: bsz * (i + 1)]
            img = imgs[index] / 255.
            p = np.zeros([img.shape[0], 144, 144, 3])
            for i in range(img.shape[0]):
                p[i] = cv2.resize(img[i], (144, 144))
            img = p.transpose(0, 3, 1, 2)
            img = torch.as_tensor(img, device=device, dtype=torch.float)

            # loss
            loss = vae.loss(img, *vae(img), kld_weight=0.)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # end of epoch handling
        print(f'#{epoch + 1} train_loss: {train_loss:.6f}, test_loss: {test_loss:.6f}')
        if test_loss < best_loss and epoch > num_epochs / 10:
            best_loss = test_loss
            print(f"save model at epoch #{epoch + 1}")
            torch.save(vae.state_dict(), '/home/jimmy/track_imgs/vae.pth')

        # print imgs for visualization
        orig_img = cv2.resize(imgs[test_indices[0]] / 255., (144, 144))
        orig_img = torch.as_tensor(orig_img, device=device, dtype=torch.float)
        orig_img = orig_img.permute(2, 0, 1)

        vae_img = vae(orig_img[None])[0][0]
        # (C, H, W)/RGB -> (H, W, C)/BGR
        cv2.imwrite("/home/jimmy/track_imgs/orig.png", orig_img.detach().cpu().numpy()[::-1].transpose(1, 2, 0) * 255)
        cv2.imwrite("/home/jimmy/track_imgs/vae.png", vae_img.detach().cpu().numpy()[::-1].transpose(1, 2, 0) * 255)
