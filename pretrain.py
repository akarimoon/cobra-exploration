import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.monet import MONet


num_epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MONet().to(device)
optimzer = optim.RMSprop(model.parameters(), lr=1e-4)

for epoch in tqdm(range(num_epochs)):
    model.train()
    for idx, batch in tqdm(enumerate(train_loader)):
        images, labels = batch
        images = images.to(device)

        optimzer.zero_grad()
        output, losses, stats, att_stats, comp_stats = model(images, labels['mask'])

        # Reconstruction error
        err = losses.err.mean(0)
        # KL divergences
        kl_m, kl_l = torch.tensor(0), torch.tensor(0)
        kl_l_texture, kl_l_shape = torch.tensor(0), torch.tensor(0)

            # -- KL stage 1
        if 'kl_m' in losses:
            kl_m = losses.kl_m.mean(0)
        elif 'kl_m_k' in losses:
            kl_m = torch.stack(losses.kl_m_k, dim=1).mean(dim=0).sum()
        # -- KL stage 2
        if 'kl_l' in losses:
            kl_l = losses.kl_l.mean(0)
        elif 'kl_l_k' in losses:
            kl_l = torch.stack(losses.kl_l_k, dim=1).mean(dim=0).sum()
        # Compute ELBO
        loss = err + beta * kl_l + gamma * kl_m

        loss.backward()
        optimzer.step()