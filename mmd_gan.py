#!/usr/bin/env python
# encoding: utf-8

import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as F
import os
import timeit
import numpy as np

import util
import base_module
from mmd import mix_rbf_mmd2


# -------------------------
# Generator
# -------------------------
class NetG(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        return self.decoder(z)


# -------------------------
# Discriminator
# -------------------------
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        f_enc = self.encoder(x)
        f_dec = self.decoder(f_enc)

        f_enc = f_enc.view(x.size(0), -1)
        f_dec = f_dec.view(x.size(0), -1)

        return f_enc, f_dec


# -------------------------
# One sided loss
# -------------------------
class ONE_SIDED(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return -self.relu(-x).mean()


# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser()
parser = util.get_args(parser)
args = parser.parse_args()

print(args)

if args.experiment is None:
    args.experiment = "samples"

os.makedirs(args.experiment, exist_ok=True)

device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# Seed
# -------------------------
args.manual_seed = 1126

random.seed(args.manual_seed)
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

cudnn.benchmark = True


# -------------------------
# Dataset
# -------------------------
trn_dataset = util.get_data(args, train_flag=True)

trn_loader = torch.utils.data.DataLoader(
    trn_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
)

if args.dataset == 'mnist':
    args.nc = 1
    args.nz = 10
else:
    args.nc = 3
    args.nz = 128
# -------------------------
# Models
# -------------------------
hidden_dim = args.nz

G_decoder = base_module.Decoder(args.image_size, args.nc, k=args.nz, ngf=64)
D_encoder = base_module.Encoder(args.image_size, args.nc, k=hidden_dim, ndf=64)
D_decoder = base_module.Decoder(args.image_size, args.nc, k=hidden_dim, ngf=64)

netG = NetG(G_decoder).to(device)
netD = NetD(D_encoder, D_decoder).to(device)

one_sided = ONE_SIDED().to(device)

print("netG:", netG)
print("netD:", netD)

netG.apply(base_module.weights_init)
netD.apply(base_module.weights_init)


# -------------------------
# MMD kernel
# -------------------------
# Single kernel: [1.0], [0.5], [0.2], [0.1]
# Multi-kernel:  [1.0, 0.5, 0.2, 0.1]
mmd_kernel_multipliers = [1.0]
print("MMD kernel denominator multipliers:", mmd_kernel_multipliers)

# -------------------------
# Fixed noise
# -------------------------
fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

one = torch.tensor(1.0, device=device)
mone = -one


# -------------------------
# Optimizers
# -------------------------
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr)


lambda_MMD = 1.0
lambda_AE_X = 8.0
lambda_AE_Y = 8.0
lambda_rg = 16.0


# -------------------------
# Training
# -------------------------
start_time = timeit.default_timer()

gen_iterations = 0

for t in range(args.max_iter):

    data_iter = iter(trn_loader)
    i = 0

    while i < len(trn_loader):

        # =========================
        # Train Discriminator
        # =========================

        for p in netD.parameters():
            p.requires_grad = True

        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = 5

        for _ in range(Diters):

            if i >= len(trn_loader):
                break

            # for p in netD.encoder.parameters():
            #     p.data.clamp_(-0.01, 0.01)

            data = next(data_iter)
            i += 1

            x_cpu, _ = data
            x = x_cpu.to(device)

            batch_size = x.size(0)

            netD.zero_grad()

            f_enc_X_D, f_dec_X_D = netD(x)
            f_enc_X_D = f_enc_X_D / (f_enc_X_D.norm(dim=1, keepdim=True) + 1e-8)

            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)

            with torch.no_grad():
                y = netG(noise)

            f_enc_Y_D, f_dec_Y_D = netD(y)
            f_enc_Y_D = f_enc_Y_D / (f_enc_Y_D.norm(dim=1, keepdim=True) + 1e-8)

            mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, mmd_kernel_multipliers)
            mmd2_D = F.relu(mmd2_D)

            one_side_errD = one_sided(
                f_enc_X_D.mean(0) - f_enc_Y_D.mean(0)
            )

            L2_AE_X_D = util.match(x.view(batch_size, -1), f_dec_X_D, "L2")
            L2_AE_Y_D = util.match(y.view(batch_size, -1), f_dec_Y_D, "L2")

            errD = (
                torch.sqrt(mmd2_D)
                + lambda_rg * one_side_errD
                - lambda_AE_X * L2_AE_X_D
                - lambda_AE_Y * L2_AE_Y_D
            )

            errD.backward(mone)

            optimizerD.step()

        # =========================
        # Train Generator
        # =========================

        for p in netD.parameters():
            p.requires_grad = False

        if i >= len(trn_loader):
            break

        data = next(data_iter)
        i += 1

        x_cpu, _ = data
        x = x_cpu.to(device)

        batch_size = x.size(0)

        netG.zero_grad()

        f_enc_X, _ = netD(x)
        f_enc_X = f_enc_X / (f_enc_X.norm(dim=1, keepdim=True) + 1e-8)

        noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
        y = netG(noise)

        f_enc_Y, _ = netD(y)
        f_enc_Y = f_enc_Y / (f_enc_Y.norm(dim=1, keepdim=True) + 1e-8)

        mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, mmd_kernel_multipliers)
        mmd2_G = F.relu(mmd2_G)

        one_side_errG = one_sided(
            f_enc_X.mean(0) - f_enc_Y.mean(0)
        )

        errG = torch.sqrt(mmd2_G) + lambda_rg * one_side_errG

        errG.backward()

        optimizerG.step()

        gen_iterations += 1

        # =========================
        # Logging
        # =========================

        run_time = (timeit.default_timer() - start_time) / 60

        print(
            f"[{t}/{args.max_iter}][{i}/{len(trn_loader)}]"
            f"[{gen_iterations}] ({run_time:.2f}m)"
            f" MMD2_D {mmd2_D.item():.6f}"
            f" hinge {one_side_errD.item():.6f}"
            f" L2_AE_X {L2_AE_X_D.item():.6f}"
            f" L2_AE_Y {L2_AE_Y_D.item():.6f}"
            f" loss_D {errD.item():.6f}"
            f" loss_G {errG.item():.6f}"
        )

        # =========================
        # Save samples
        # =========================

        if gen_iterations % 500 == 0:

            with torch.no_grad():
                y_fixed = netG(fixed_noise)

            y_fixed = y_fixed * 0.5 + 0.5

            f_dec_X_D = f_dec_X_D.view(
                f_dec_X_D.size(0),
                args.nc,
                args.image_size,
                args.image_size,
            )

            f_dec_X_D = f_dec_X_D * 0.5 + 0.5

            vutils.save_image(
                y_fixed,
                f"{args.experiment}/fake_samples_{gen_iterations}.png",
            )

            vutils.save_image(
                f_dec_X_D,
                f"{args.experiment}/decode_samples_{gen_iterations}.png",
            )

    if t % 50 == 0 or t == args.max_iter - 1:

        torch.save(
            netG.state_dict(),
            f"{args.experiment}/netG_iter_{t}.pth",
        )

        torch.save(
            netD.state_dict(),
            f"{args.experiment}/netD_iter_{t}.pth",
        )
