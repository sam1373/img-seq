import os
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import *
from tqdm import *

from loss import *
from optimizers import *

import pandas as pd

import matplotlib.pyplot as plt

LEARNING_RATE = 0.001

from torch.optim.lr_scheduler import StepLR


def KLD(mean, logvar):

    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

class Trainer(object):

    def __init__(self, model, sample_path="sample", checkpoints='model.cp'):

        self.model = model
        self.sample_path = sample_path
        self.checkpoints = checkpoints

        self.start_epoch = 0

        self.optimizer = Novograd(self.model.parameters())
        #self.optimizer = torch.optim.Adam(self.model.parameters())
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.99)

        self.load_checkpoint()

        self.avgpool = nn.AvgPool2d(2)


    def sample(self, bs, same_z=False, set_z=False, mult=1.):

        side_len = self.model.module.side_len 

        data = torch.zeros(bs, self.model.module.in_channels, side_len * side_len).cuda()

        shape2d = (bs, self.model.module.in_channels, side_len, side_len)

        if set_z:
            z = torch.Tensor([set_z]).view(1, -1).cuda().repeat(bs, 1)
        elif same_z:
            z = torch.randn([1, self.model.module.z_dim]).cuda().repeat(bs, 1) * mult
        else:
            z = torch.randn([bs, self.model.module.z_dim]).cuda() * mult
        

        for i in range(data.shape[2]):

                out = self.model.forward(data.view(shape2d), set_z=z)

                sample = sample_from_logistic_mix(out)
                cur_out = sample.view(data.shape)[:, :, i]
                        
                data[:, :, i] = cur_out

                print(i, "/", data.shape[2])

        final = self.model.module.final_pass(data.view(-1, self.model.module.in_channels, side_len, side_len))

        return data, final


    def sample_frames(self, epoch, mult=1.):
        with torch.no_grad():

            side_len = self.model.module.side_len

            gen, gen_final = self.sample(40, same_z=False, mult=mult)

            gen = gen.view(-1, self.model.module.in_channels, side_len, side_len)

            gen = gen / 2. + 0.5

            gen_final = gen_final.view(-1, self.model.module.in_channels, side_len * 2, side_len * 2)

            gen_final = gen_final / 2. + 0.5

            """
            gen_same = self.sample(40, same_z=True, mult=mult)

            gen_same = gen_same.view(-1, self.model.module.in_channels, side_len, side_len)

            gen_same = gen_same / 2. + 0.5
            """

            print(gen.min(), gen.max())

            torchvision.utils.save_image(gen,'%s/epoch%d.png' % (self.sample_path,epoch), nrow=10)
            torchvision.utils.save_image(gen_final,'%s/epoch%d_final.png' % (self.sample_path,epoch), nrow=10)
            #torchvision.utils.save_image(gen_same,'%s/epoch%d_same.png' % (self.sample_path,epoch), nrow=10)


    def recon_frames(self, epoch, x):

        #self.model.train()

        with torch.no_grad():
            side_len = self.model.module.side_len 

            img_use = 10

            x0 = x[:img_use]

            x0_down = self.avgpool(x0)

            recon = self.model.forward(x0_down)

            recon = sample_from_logistic_mix(recon)

            recon = recon.view(-1, self.model.module.in_channels, side_len, side_len)

            recon_final = self.model.module.final_pass(recon)

            imgs_with_recon = torch.cat((x0_down, recon), dim=0)
            imgs_with_recon = imgs_with_recon / 2. + 0.5

            imgs_with_recon_final = torch.cat((x0, recon_final), dim=0)
            imgs_with_recon_final = imgs_with_recon_final / 2. + 0.5

            torchvision.utils.save_image(imgs_with_recon, '%s/epoch%d_recon.png' % (self.sample_path,epoch), nrow=10)
            torchvision.utils.save_image(imgs_with_recon_final, '%s/epoch%d_recon_final.png' % (self.sample_path,epoch), nrow=10)
    
    def save_checkpoint(self,epoch):
        torch.save({
            'epoch' : epoch+1,
            'state_dict' : self.model.state_dict(),
            },
            self.checkpoints)
        
    def load_checkpoint(self):
        self.start_epoch = 0
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except Exception as e:
            print(e)
            #print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            #self.start_epoch = 0

        for i in range(self.start_epoch):
            self.scheduler.step()

        

    def train_model(self, trainloader, epochs=100, test_every_x=4, epochs_per_level=50):

        

        #avgDiff = 0
        for epoch in range(self.start_epoch, epochs):

           self.model.train()

           #current_level = self.model.module.levels - epoch // epochs_per_level

           #urrent_level = max(1, current_level)

           #print("Current level:", current_level)
           #trainloader.shuffle()
           losses = []
           kld_fs = []
           kld_zs = []
           print("Running Epoch : {}".format(epoch+1))
           print(len(trainloader))


           lastDiff = 0
           #lastDiff = avgDiff
           avgDiff = 0

           loss_type = "levels"

           mse_loss = nn.MSELoss()

           for i, dataitem in tqdm(enumerate(trainloader, 1)):
               if i >= len(trainloader):
                break
               data, _ = dataitem
               data = data.cuda()

               if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
                bs = data.shape[0]
                data *= torch.rand([bs, 3, 1, 1]).cuda()

                #data[:, :, 10:20, 10:20] = 1.

                #plt.imshow(data[0].permute(1, 2, 0).cpu())
                #plt.show()
                #input()

               data = (data - 0.5) * 2

               data = Variable(data)

               self.optimizer.zero_grad()

               data_down = self.avgpool(data)

               data_noise = data_down + torch.randn(data_down.shape).cuda() * 0.1

               #plt.imshow(data[0].permute(1, 2, 0).cpu() / 2. + 0.5)
               #plt.show()
               #plt.imshow(data_noise[0].permute(1, 2, 0).cpu() / 2. + 0.5)
               #plt.show()
               #print(data_noise.shape)

               out = self.model.forward(data_noise)
               
               loss_mix = discretized_mix_logistic_loss(data_down, out)

               out_sample = sample_from_logistic_mix(out)

               out_final = self.model.module.final_pass(out_sample)

               loss_mse = mse_loss(out_final, data)

               #norm = torch.randn(z.shape).cuda()

               #loss_z = mmd(z, norm) * 200000

               loss = loss_mix + loss_mse# + loss_z

               loss = loss.mean()

               if i % 10 == 0:
                 print(loss_mix, loss_mse)

               loss.backward()

               torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

               self.optimizer.step()

               losses.append(loss.item())

           self.scheduler.step()

           meanloss = np.mean(losses)

           #avgDiff /= len(trainloader)
           #meanf = np.mean(kld_fs)
           #meanz = np.mean(kld_zs)
           #self.epoch_losses.append(meanloss)
           print("Epoch {} : Average Loss: {}".format(epoch+1, meanloss))

           #print("Disc. quality: {}".format(avgDiff))
           self.save_checkpoint(epoch)


           self.model.eval()
           _, (sample, _)  = next(enumerate(trainloader))
           #sample = torch.unsqueeze(sample,0)


           sample = sample.cuda()

           if sample.shape[1] == 1:
               sample = sample.repeat(1, 3, 1, 1)
               bs = sample.shape[0]
               sample *= torch.rand([bs, 3, 1, 1]).cuda()

           sample = (sample - 0.5) * 2

           if (epoch + 1) % test_every_x == 0:
            self.recon_frames(epoch+1, sample)
            self.sample_frames(epoch+1)
            #self.umap_codes(epoch+1, trainloader)
           self.model.train()
        print("Training is complete")
