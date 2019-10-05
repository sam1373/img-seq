import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from train import Trainer

from model import *

import matplotlib.pyplot as plt

from data import CelebaDataset

import numpy as np




batch_size = 128

sep = 8

#model = ImgAttendModel(side_len=16, kernels=3, channels=128, blocks_rep=4, conv_rep=4)
model = PixelCNN(side_len=14, kernels=9, in_channels=3, channels=128, out_channels=100, total_convs=12, use_z=False)
#model = PixelAttendPers(side_len=16, kernels=9, in_channels=3, channels=128, out_channels=100, total_layers=12, z_dim=32)
#model = PixelCNNProg(side_len=28, kernels=9, in_channels=3, channels=128, out_channels=100, total_attn=9, levels=2)
model = nn.DataParallel(model)

total = 0
for p in model.parameters():
  total += np.prod(p.shape)
  print(p.shape)
  
print(total)
#input()

rescaling = lambda x : (x - .5) * 2.

custom_transform = transforms.Compose([transforms.ToTensor()])

dataset = "MNIST"

local = True

if dataset == "CelebA":

  txt_path = "/home/samuel/Data/CelebAligned/list_attr_celeba.txt"
  img_dir = "/home/samuel/Data/CelebAligned/"

  if local == False:
    txt_path = "/data/skriman2/CelebAligned/list_attr_celeba.txt"
    img_dir = "/data/skriman2/CelebAligned/"

  train_dataset = CelebaDataset(txt_path=txt_path,
                                img_dir=img_dir,
                                transform=custom_transform, in_size=model.module.side_len)

  trainloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
elif dataset == "MNIST":


  data_path = "MNIST/"

  if not local:
    data_path = "/data/skriman2/MNIST/"

  trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_path, train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                 #transforms.Resize((16, 16)),
                                 transforms.ToTensor(),
                                 #rescaling
                                 #transforms
                                 #transforms.ColorJitter(0.1, 0.1, 0.1),
                                 #transforms.Normalize(
                                 #  (0.5,), (0.5,))
                               ])),
    batch_size=batch_size, shuffle=True, num_workers=0)


examples = enumerate(trainloader)



_, (sample, _) = next(examples)
print(sample.shape)


#plt.imshow(sample[0].permute(1, 2, 0))
#plt.show()

#input()

#sample = torch.unsqueeze(sample,0)

sample = sample.cuda()

if sample.shape[1] == 1:
  sample = sample.repeat(1, 3, 1, 1)
  bs = sample.shape[0]
  sample *= torch.rand([bs, 3, 1, 1]).cuda()

sample = (sample - 0.5) * 2

sample_path = "sample"

if not local:
  sample_path = "/data/skriman2/sample"

trainer = Trainer(model, sample_path=sample_path, checkpoints='model.cp')

trainer.model.eval()

trainer.recon_frames(0, sample)
trainer.sample_frames(0, mult=0.8)

trainer.train_model(trainloader, test_every_x=5, epochs=1000, epochs_per_level=500)

