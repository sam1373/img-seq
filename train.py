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

import pandas as pd

import matplotlib.pyplot as plt

LEARNING_RATE = 0.001

from torch.optim.lr_scheduler import StepLR


def KLD(mean, logvar):

    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

class Trainer(object):

    def __init__(self, model, sample_path="sample", recon_path="recon", codes_path="codes", checkpoints='model.cp'):

        self.model = model
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.codes_path = codes_path
        self.checkpoints = checkpoints

        self.start_epoch = 0

        self.load_checkpoint()

        self.optimizer = torch.optim.Adam(self.model.parameters())
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)


    def sample(self, bs, same_z=False):
        self.model.train(False)
        data = torch.zeros(bs, self.model.in_channels, self.model.side_len, self.model.side_len)
        data = data.cuda()

        """
        if same_z:
            z = torch.randn([1, self.model.z_dim]).cuda().repeat(bs, 1)
        else:
            z = torch.randn([bs, self.model.z_dim]).cuda()
        """

        for i in range(self.model.side_len):
            for j in range(self.model.side_len):
                out   = self.model(data)

                data[:, :, i, j] = out[:, :, i, j]
                
                sep = 8

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
                print(i, j)
        return data


    def sample_frames(self, epoch):
        with torch.no_grad():
            x_gen = self.sample(40)
            print(x_gen.min(), x_gen.max())
            torchvision.utils.save_image(x_gen,'%s/epoch%d.png' % (self.sample_path,epoch), nrow=10)

            #x_gen = self.sample(40, same_z=True)
            #print(x_gen.min(), x_gen.max())
            #torchvision.utils.save_image(x_gen,'%s/epoch%d_same.png' % (self.sample_path,epoch), nrow=10)
    
    def save_checkpoint(self,epoch):
        torch.save({
            'epoch' : epoch+1,
            'state_dict' : self.model.state_dict(),
            },
            self.checkpoints)
        
    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def train_model(self, trainloader, epochs=100, test_every_x=4):

        self.model.train()

        #avgDiff = 0
        for epoch in range(self.start_epoch, epochs):
           #trainloader.shuffle()
           losses = []
           kld_fs = []
           kld_zs = []
           print("Running Epoch : {}".format(epoch+1))
           print(len(trainloader))


           lastDiff = 0
           #lastDiff = avgDiff
           avgDiff = 0

           loss_type = "val"

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

                #plt.imshow(data[0].permute(1, 2, 0).cpu())
                #plt.show()
                #input()

               data = Variable(data)

               self.optimizer.zero_grad()

               out = self.model.forward(data)

               #plt.imshow(out[0].permute(1, 2, 0).detach().cpu())
               #plt.show()

               if loss_type == "val":

                    #print(logvar)
                    #print(z_logvar)

                    loss = mse_loss(out, data)# - logvar * 0.0001# + KLD(z_mean, z_logvar)

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

                loss = discretized_mix_logistic_loss(data, out)

               if (i + 1) % 10 == 0:
                print()
                print("Step", i + 1, "loss:", loss.item())
                print()

               loss.backward()


               torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)

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
           if (epoch + 1) % test_every_x == 0:
            self.sample_frames(epoch+1)
            #self.recon_frame(epoch+1,sample)
            #self.umap_codes(epoch+1, trainloader)
           self.model.train()
        print("Training is complete")