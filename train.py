import os
import torch
import torchvision
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
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.98)

        self.load_checkpoint()


    def sample(self, bs, level, var_mult):
        #self.model.train(False)

        print(level)
        #TODO:actually implement this for multiple level generation

        #side_len = self.model.module.side_len // (2 ** (level - 1))

        side_len = self.model.module.side_len // (2 ** (self.model.module.levels - 1))

        data = []

        for i in range(self.model.module.levels - level + 1):
            data.append(torch.zeros(bs, self.model.module.in_channels, side_len * side_len).cuda())
            side_len *= 2

        #data = (data_3, data_2)

        """
        if same_z:
            z = torch.randn([1, self.model.z_dim]).cuda().repeat(bs, 1)
        else:
            z = torch.randn([bs, self.model.z_dim]).cuda()
        """

        for i in range(data[-1].shape[2]):
                #print(data[0].shape)
                out = self.model.forward(data, level=level, var_mult=var_mult)

                for j in range(len(out)):
                    if i < data[j].shape[2]:
                        #cur_out = out[j].view(data[j].shape[0], out[j].shape[1], data[j].shape[2])[:, :, i]
                        sample = sample_from_logistic_mix(out[j])
                        cur_out = sample.view(data[j].shape)[:, :, i]
                        
                        data[j][:, :, i] = cur_out

                """
                #s = torch.sum(out[:, :, i, j], dim=1)
                probs = F.softmax(out[:, :, i, j])
                sel = torch.multinomial(probs, 1).view(bs).long()

                #print(sel)                
                #print(sel)
                #print(sel.shape)
                #input()
                b = sel % sep
                g = sel / sep % sep
                r = sel / sep / sep
                
                #print(torch.multinomial(probs1, 1).shape)
                data[:, 0, i, j] = r.float() * (1. / (sep - 1))
                data[:, 1, i, j] = g.float() * (1. / (sep - 1))
                data[:, 2, i, j] = b.float() * (1. / (sep - 1))
                
                """

                #probs1 = F.softmax(out[:, :sep, i, j], dim=1)
                #probs2 = F.softmax(out[:, sep:sep*2, i, j], dim=1)
                #probs3 = F.softmax(out[:, sep*2:, i, j], dim=1)
                #data[:, 0, i, j] = torch.multinomial(probs1, 1).view(bs).float() / (sep - 1)
                #data[:, 1, i, j] = torch.multinomial(probs2, 1).view(bs).float() / (sep - 1)
                #data[:, 2, i, j] = torch.multinomial(probs3, 1).view(bs).float() / (sep - 1)
                

                #out_sample = sample_from_logistic_mix(out)
                #data[:, :, i, j] = out_sample.data[:, :, i, j]
                print(i, "/", data[-1].shape[2])
        return data


    def sample_frames(self, epoch, level, var_mult=1.):
        with torch.no_grad():
            gen = self.sample(40, level, var_mult)
            #x_gen_3 = x_gen_3.view(-1, self.model.module.in_channels, self.model.module.side_len // 4, self.model.module.side_len // 4)
            side_len = self.model.module.side_len // (2 ** (level - 1))
            gen = gen[-1].view(-1, self.model.module.in_channels, side_len, side_len)

            gen = gen / 2. + 0.5

            print(gen.min(), gen.max())

            torchvision.utils.save_image(gen,'%s/epoch%d.png' % (self.sample_path,epoch), nrow=10)
            #torchvision.utils.save_image(x_gen_2,'%s/epoch%d_14.png' % (self.sample_path,epoch), nrow=10)

            #x_gen = self.sample(40, same_z=True)
            #print(x_gen.min(), x_gen.max())
            #torchvision.utils.save_image(x_gen,'%s/epoch%d_same.png' % (self.sample_path,epoch), nrow=10)

    def recon_frames(self, epoch, x, level):

        #self.model.train()

        with torch.no_grad():
            side_len = self.model.module.side_len // (2 ** (level - 1))

            img_use = 10

            x0 = x[:img_use]
            data = []
            for j in range(1, self.model.module.levels + 1):
                if j >= level:
                    data.append(x0)
                x0 = self.model.module.downsample(x0)

            data = data[::-1]



            recon, _ = self.model.forward(data, level=level, training=True)
            recon = recon[-1]
            recon = sample_from_logistic_mix(recon)

            """
            for i in range(len(data)):
               plt.imshow(data[i][0].permute(1, 2, 0).cpu())
               plt.show()

               plt.imshow(recon[i][0].permute(1, 2, 0).detach().cpu())
               plt.show()
            """

            recon = recon.view(-1, self.model.module.in_channels, side_len, side_len)
            imgs_with_recon = torch.cat((data[-1], recon), dim=0)
            imgs_with_recon = imgs_with_recon / 2. + 0.5
            torchvision.utils.save_image(imgs_with_recon, '%s/recon_epoch%d.png' % (self.sample_path,epoch), nrow=10)
    
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

           current_level = self.model.module.levels - epoch // epochs_per_level

           current_level = max(1, current_level)

           print("Current level:", current_level)
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

               x0 = data#.view(-1, self.model.module.in_channels, self.side_len, self.side_len)
               data = []
               for j in range(1, self.model.module.levels + 1):
                if j >= current_level:
                    data.append(x0)
                x0 = self.model.module.downsample(x0)

               data = data[::-1]

               out, logvar = self.model.forward(data, training=True, level=current_level)
               


               #plt.imshow(out[0].permute(1, 2, 0).detach().cpu())
               #plt.show()

               #data_2 = F.avg_pool2d(data, 2)
               #data_3 = F.avg_pool2d(data_2, 2)

               """
               for i in range(len(data)):
                   plt.imshow(data[i][0].permute(1, 2, 0).cpu())
                   plt.show()

                   plt.imshow(out[i][0].permute(1, 2, 0).detach().cpu())
                   plt.show()
               """        
               
               

               if loss_type == "levels":



                    #print(logvar)
                    #print(z_logvar)

                    

                    #print(data_3.shape)
                    #plt.imshow(data_3[0].permute(1, 2, 0).cpu())
                    #plt.show()

                    loss = 0

                    #loss += -logvar * 0.0001#0.0002

                    if (i + 1) % 10 == 0:
                        print()
                        print("Step", i + 1)
                        print("logvar:", logvar.sum())

                    for j in range(len(out)):
                        loss_j = discretized_mix_logistic_loss(data[j], out[j])
                        #loss_j = mse_loss(out[j], data[j])

                        if (i + 1) % 10 == 0:
                            print("loss res", j, ":", loss_j)

                        #if j == len(out) - 1:
                        #    loss_j *= (len(out))

                        loss = loss + loss_j

                    #loss = mse_3 + mse_2 * 2 - logvar * 0.0001# + KLD(z_mean, z_logvar)

                    if (i + 1) % 10 == 0:
                       print()

               #ce
               elif loss_type == "ce":
                   sep = 8
               
                   data = data + 0.01

                   target1 = Variable((data[:, 0] * (sep - 1)).long())
                   target2 = Variable((data[:, 1] * (sep - 1)).long())
                   target3 = Variable((data[:, 2] * (sep - 1)).long())
                   target = target1 * sep * sep + target2 * sep + target3
                   #print(out.shape, target.shape)
                   #print(target[0])
                   #input()
                   
                   loss = F.cross_entropy(out, target)#F.cross_entropy(out[:, :sep], target1) + F.cross_entropy(out[:, sep:sep*2], target2) + F.cross_entropy(out[:, sep*2:], target3)
               
               else:

                loss = discretized_mix_logistic_loss(data_3, out)

               loss = loss.sum()

               loss.backward()


               torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.001)

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
            self.recon_frames(epoch+1, sample, level=current_level)
            self.sample_frames(epoch+1, level=current_level)
            #self.umap_codes(epoch+1, trainloader)
           self.model.train()
        print("Training is complete")
