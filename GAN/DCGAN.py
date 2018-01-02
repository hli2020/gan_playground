from __future__ import print_function
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as T
import torchvision.datasets as dset
import torchvision.utils as vutils

# Created by Hongyang, Aug 16 2017
# Code refactored from the official pytorch example


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | fake')
parser.add_argument('--dataroot', default='./', help='path to dataset')
parser.add_argument('--out_folder', default='./dcgan/', help='folder to output images and model checkpoints')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--imageSize', default=64, type=int)
parser.add_argument('--resume_D_path', default='')
parser.add_argument('--no_cuda', action='store_true', help='no use of gpu')
opt = parser.parse_args()

# put additional parameters here
opt.imageSize = 64
opt.batchSize = 64
opt.resume_D_path = ''
opt.resume_G_path = ''

opt.lr = 0.0002
opt.beta1 = 0.5
opt.n_eps = 25
opt.gpu_ids = [1,2]
opt.cuda = not opt.no_cuda

print(opt)

try:
    os.makedirs(opt.out_folder)
except OSError:
    print('out_folder mkdir failed')
    pass

seed = random.randint(1, 10000)
print("Random seed is: ", seed)
random.seed(seed)
torch.manual_seed(seed)

if opt.cuda:
    torch.cuda.manual_seed_all(seed)

if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataset, download=True,
                           transform=T.Compose([
                               T.Scale(opt.imageSize),
                               T.ToTensor(),
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fashion-mnist':
    pass
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize), transform=T.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=2)

nc, nz, ngf, ndf = 3, 100, 64, 64


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(_netG, self).__init__()
        self.gpu_ids = gpu_ids
        self.ngpu = len(gpu_ids)
        self.main = nn.Sequential(
            # first layer
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # next state
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # next state
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # next state
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # final layer
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, self.gpu_ids)
        else:
            output = self.main(input)
        return output

netG = _netG(opt.gpu_ids)
netG.apply(weights_init)
if opt.resume_G_path != '':
    netG.load_state_dict(torch.load(opt.resume_G_path))
print(netG)


class _netD(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(_netD, self).__init__()
        self.gpu_ids = gpu_ids
        self.ngpu = len(gpu_ids)
        self.main = nn.Sequential(
            # first layer
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # out size = W
            nn.LeakyReLU(0.2, inplace=True),
            # next state
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # next state
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # next state
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # final layer
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # out size = W - 3
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, self.gpu_ids)
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

netD = _netD(opt.gpu_ids)
netD.apply(weights_init)
if opt.resume_D_path != '':
    netD.load_state_dict(torch.load(opt.resume_D_path))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = Variable(torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1))
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# set up optimizer
optm_D = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optm_G = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.n_eps):
    for i, data in enumerate(dataloader, 0):

        #########################
        # (1) update D network: maximize log D(x) + log (1 - D(G(z)))
        #########################
        # real data
        netD.zero_grad()
        real_cpu, _ = data
        # the actual batch size may be smaller in the last iter of an epoch
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        input_v = Variable(input)
        label_v = Variable(label)


        # errD_real.backward()
        # D_x = output.data.mean()

        # fake data
        noise_v = Variable(noise.resize_(batch_size, nz, 1, 1).normal_(0, 1))
        fake = netG(noise_v)

        label_v2 = Variable(label.fill_(fake_label))
        input_var = torch.cat((input_v, fake.detach()), dim=0)
        label_var = torch.cat((label_v, label_v2), dim=0)
        output = netD(input_var)    # D(G(z)), MUST have detach here
        # output = netD(fake)

        errD = criterion(output, label_var)
        #errD_fake.backward()
        #D_G_z1 = output.data.mean()
        # errD = errD_fake + errD_real
        errD.backward()         # loss D and G will explode
        optm_D.step()
        D_x = 0

        #########################
        # (2) update G network: maximize log D(G(z))
        #########################
        netG.zero_grad()
        label_v = Variable(label.resize_(batch_size).fill_(real_label))     # the label is "1" for G
        # must resize the label to actual batch_size since the last iter in the epoch may be smaller
        output = netD(fake)     # D(G(z)), where G(z) = fake
        errG = criterion(output, label_v)
        errG.backward()             # should network D also be computed gradients?
        D_G_z2 = output.data.mean()
        optm_G.step()

        print('hyli-mac: [%d / %d][%d / %d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, real %.4f, fake %.4f' %
              (epoch, opt.n_eps, i, len(dataloader), errD.data[0], errG.data[0], D_x, 0, D_G_z2, 0, 0))

        if i % 100 == 0:
            vutils.save_image(real_cpu, '%s/real_sample_epoch_%d_i_%d.png' % (opt.out_folder, epoch, i), normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data, '%s/fake_sample_epoch_%d_i_%d.png' % (opt.out_folder, epoch, i), normalize=True)
            print('=== samples saved! ===')

    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.out_folder, epoch))
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.out_folder, epoch))



