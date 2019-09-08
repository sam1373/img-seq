import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

#from loss import *

import functools

@functools.lru_cache(maxsize=32)
def get_causal_mask(canvas_size, see_self=True):
    mask = torch.zeros([canvas_size, canvas_size])
    for i in range(canvas_size):
        mask[i, :i + 1] = 1.
    return mask.cuda()

def get_pos_embeddings(side_len):

    x = torch.arange(0., 1., 1. / side_len).view(1, 1, -1, 1).repeat(1, 1, 1, side_len)

    y = torch.arange(0., 1., 1. / side_len).view(1, 1, 1, -1).repeat(1, 1, side_len, 1)

    return torch.cat((x, y), dim=1).cuda()

#class MyDataParallel(nn.DataParallel):
#    def __getattr__(self, name):
#        return getattr(self.module, name)


#causal conv
class CausalConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernels, see_center=True):
        super(CausalConv, self).__init__(in_channels, out_channels, kernels, padding=kernels//2)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kh, kw = self.weight.size()
        yc, xc = kh // 2, kw // 2
        #self.mask.fill_(1)

        pre_mask = np.ones(self.weight.shape)

        """
        # same pixel masking - pixel won't access next color (conv filter dim)
        def bmask(i_out, i_in):
            cout_idx = np.expand_dims(np.arange(out_channels) // 3 == i_out, 1)
            cin_idx = np.expand_dims(np.arange(in_channels) // 3 == i_in, 0)
            a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            return a1 * a2

        for j in range(3):
            pre_mask[bmask(j, j), yc, xc] = 1.0 if see_center else 0.0

        pre_mask[bmask(0, 1), yc, xc] = 0.0
        pre_mask[bmask(0, 2), yc, xc] = 0.0
        pre_mask[bmask(1, 2), yc, xc] = 0.0
        """

        pre_mask[:, :, yc, xc + see_center:] = 0
        pre_mask[:, :, yc + 1:] = 0

        self.mask = torch.Tensor(pre_mask)


        #self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(CausalConv, self).forward(x)
        #x = self.bn(x)
        return x

class ConvBlock(nn.Module):

    def __init__(self, channels, kernels=5):

        super(ConvBlock, self).__init__()

        self.conv_1 = nn.Sequential(nn.ELU(), CausalConv(channels, channels, kernels), nn.ELU())

        self.conv_2 = CausalConv(channels, channels, kernels)

        self.conv_3 = nn.Sequential(CausalConv(channels, channels, kernels), nn.Sigmoid())

        #self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):

        x_original = x

        x = self.conv_1(x)

        a = self.conv_2(x)

        x = self.conv_3(x) * a + x_original

        #x = self.bn(x)

        return x

#attention block
class AttentionBlock(nn.Module):

    def __init__(self, in_channels, K, V, side_len, bg=None):

        super(AttentionBlock, self).__init__()


        self.side_len = side_len
        #self.bg = bg

        #if not self.bg:
        #    self.bg = torch.zeros([in_channels, canvas_size])

        self.mask = get_causal_mask(side_len ** 2)

        self.q_conv = nn.Sequential(nn.Conv1d(in_channels + 2, K, 1), nn.ReLU())

        self.k_conv = nn.Sequential(nn.Conv1d(in_channels + 2, K, 1), nn.ReLU())

        self.v_conv = nn.Sequential(nn.Conv1d(in_channels + 2, V, 1), nn.ReLU())

        self.fc = nn.Sequential(nn.Conv1d(V, V, 1), nn.ReLU())

        self.out_channels = V

    def forward(self, x):

        orig_shape = x.shape

        pos_emb = get_pos_embeddings(self.side_len).repeat(orig_shape[0], 1, 1, 1)

        x0 = torch.cat((x, pos_emb), dim=1)

        x0 = x0.view(orig_shape[0], orig_shape[1] + 2, -1)

        q = self.q_conv(x0)

        k = self.k_conv(x0)

        v = self.v_conv(x0)

        #print(q.shape, k.shape)

        attn = torch.bmm(q.transpose(-2, -1), k)

        #print(attn.shape)  
        #print(attn[0], self.mask[0])
        #print(attn[0])
        mask = self.mask.to(attn.device)

        #print(mask[0])

        attn_masked = attn * mask + (1 - mask) * (-10000.)

        #print(attn_masked[0])

        attn_masked = nn.functional.softmax(attn_masked, dim=-1) * mask

        #print(attn_masked[0])
        #input()

        #attn_bg = attn - attn_masked

        #out = torch.bmm(v, attn_masked) + torch.bmm(bh, attn_bg)

        #print(attn.shape, causal_val.shape)
        #print(attn_masked.shape, v.shape)

        #use attn_masked
        out = torch.bmm(attn_masked, v.transpose(-2, -1)).transpose(-2, -1)

        out = self.fc(out)

        out = out.view(orig_shape[0], self.out_channels, orig_shape[2], orig_shape[3])

        out = x + out

        return out


class SnailBlock(nn.Module):

    def __init__(self, channels, out_channels, side_len, conv_block_rep=3, in_channels=3, kernels=5):

        super(SnailBlock, self).__init__()

        self.in_channels = in_channels

        self.conv_blocks = []

        for i in range(conv_block_rep):
            self.conv_blocks.append(ConvBlock(channels, kernels))
            self.conv_blocks.append(nn.BatchNorm2d(channels))

        self.conv_blocks = nn.Sequential(*self.conv_blocks)

        self.attn_block = AttentionBlock(channels, 16, channels, side_len)

        self.extra_conv1 = nn.Sequential(CausalConv(channels, channels, kernels), nn.ELU())

        self.extra_conv2 = nn.Sequential(CausalConv(channels, channels, kernels), nn.ELU())

        self.extra_conv3 = nn.Sequential(CausalConv(channels, out_channels, kernels), nn.ELU())

    def forward(self, x, inp=None):


        for l in self.conv_blocks:
            x = l(x)


        x1 = self.extra_conv1(x)

        bs, ch, side_len, _ = x.shape

        #x2 = torch.cat((x, inp), dim=1)
        x2 = x

        #pos_emb = get_pos_embeddings(side_len).repeat(bs, 1, 1, 1)

        #print(x2.shape, pos_emb.shape)

        #x2 = torch.cat((x2, pos_emb), dim=1)

        #x2 = x2.view(bs, ch + 2, -1)

        x2 = self.attn_block(x2)

        #x2 = x2.view(bs, ch, side_len, side_len)

        x2 = self.extra_conv2(x2)

        x = x1 + x2

        x = self.extra_conv3(x)

        return x

class ImgAttendModel(nn.DataParallel):

    def __init__(self, side_len=64, in_channels=3, channels=64, out_channels=100, blocks_rep=6, conv_rep=4, kernels=5):

        super(ImgAttendModel, self).__init__()

        self.canvas_size = side_len * side_len
        self.in_channels = in_channels
        self.side_len = side_len
        self.out_channels = out_channels

        self.in_conv = CausalConv(in_channels, channels, kernels, see_center=False)

        self.blocks = []

        for i in range(blocks_rep):
            #out_ch = channels
            #if i == blocks_rep - 1:
            #    out_ch = 100
            self.blocks.append(SnailBlock(channels, out_channels=channels, side_len=side_len, conv_block_rep=conv_rep, in_channels=in_channels, kernels=kernels))
            self.blocks.append(nn.BatchNorm2d(channels))

        self.blocks = nn.Sequential(*self.blocks)

        #self.out_conv = nn.Conv2d(channels, out_channels, 1)#, nn.Tanh())

        self.out_l = nn.Sequential(nn.Conv2d(channels, out_channels, 1), nn.Sigmoid())

        self.final = nn.Sequential(nn.Conv2d(out_channels // 2, channels, 1), nn.ELU(), nn.Conv2d(channels, in_channels, 1), nn.Sigmoid())

        self.cuda()

    def forward(self, x):

        #print(x.mean())

        inp = x

        #print(x.mean())

        x = self.in_conv(x)

        x0 = x

        #print(self.in_conv.weight.shape)
        #input()
        #print(self.in_conv.weight[0])

        #print(x.mean())

        for l in self.blocks:
            x = l(x)
            #print(x.mean())

        #print(x.mean())

        #x = self.out_conv(x)# * 3

        bs = x.shape[0]

        x = self.out_l(x)

        mean = x[:, :self.out_channels // 2]
        logvar = x[:, self.out_channels // 2:] * 4 - 2

        x = torch.randn(bs, self.out_channels // 2, self.side_len, self.side_len).cuda()

        x = x * torch.exp(logvar * 0.5) + mean

        #z = z * torch.exp(log_var) + mean

        x = self.final(x)

        #print(x.mean())

        return x

    """

    def gen_img(self, batch_size=32):

        x = torch.zeros([batch_size, self.in_channels, self.canvas_size]).cuda()

        for i in range(self.canvas_size):
            l = self.forward(x.view(-1, self.in_channels, self.side_len, self.side_len))
            out = sample_from_logistic_mix(l)

            out = out.view(-1, self.in_channels, self.canvas_size)

            x[:, :, i] = out[:, :, i]
            print(i + 1, "done")

        return x
    """

class PixelCNN(nn.Module):

    def __init__(self, side_len=64, in_channels=3, channels=64, out_channels=100, total_convs=8, kernels=5, z_dim=32):

        super(PixelCNN, self).__init__()

        self.canvas_size = side_len * side_len
        self.in_channels = in_channels
        self.side_len = side_len
        self.total_convs = total_convs
        self.out_channels = out_channels


        self.z_dim = z_dim

        self.z_model_conv = nn.Sequential(nn.Conv2d(in_channels, 64, kernels, padding=kernels//2), nn.ELU(),
                                          nn.Conv2d(64, 32, kernels, padding=kernels//2), nn.ELU(),
                                          nn.Conv2d(32, 16, kernels, padding=kernels//2), nn.ELU(),
                                          nn.Conv2d(16, 4, 1),nn.ELU())
        self.z_model_flat = nn.Sequential(nn.Linear(4 * self.canvas_size, z_dim * 2))

        self.in_l = nn.Sequential(CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nn.ELU())

        self.layers = []

        for i in range(total_convs):
            l = CausalConv(channels, channels, kernels, see_center=True)
            #self.layers.append(nn.Sequential(l, nn.BatchNorm2d(channels), nn.ELU()))
            self.layers.append(nn.Sequential(l, AttentionBlock(channels, K=16, V=channels, side_len=self.side_len), nn.BatchNorm2d(channels), nn.ELU()))
        self.out_l = nn.Sequential(nn.Conv2d(channels, out_channels, 1), nn.Sigmoid())

        self.final = nn.Sequential(nn.Conv2d(out_channels // 2, channels, 1), nn.ELU(), nn.Conv2d(channels, in_channels, 1), nn.Sigmoid())

        self.layers = nn.Sequential(*self.layers)

        #self.in_l = nn.DataParallel(self.in_l)
        #self.layers = nn.DataParallel(self.layers)
        #self.out_l = nn.DataParallel(self.out_l)
        #self.final = nn.DataParallel(self.final)

        self.cuda()


    def forward(self, x, z=None, training=False):


        bs = x.shape[0]

        """
        if training:

            z_conv = self.z_model_conv(x)
            z_flat = z_conv.view(bs, -1)
            z_flat = self.z_model_flat(z_flat)
            z_mean = z_flat[:, :self.z_dim]
            z_logvar = z_flat[:, self.z_dim:]

            z = torch.randn(self.z_dim).cuda()
            z = z * torch.exp(z_logvar * 0.5) + z_mean

            #print(z.shape)
        """

        x = self.in_l(x)

        i = 0

        past_out = []

        #orig_shape = x.shape

        #x = x.view(bs, x.shape[1], -1)

        for l in self.layers:
            x = l(x)
            if 1:#isinstance(l, CausalConv):
                if i < self.total_convs // 2:
                    i += 1
                    past_out.append(x)
                    #print(1)
                else:
                    #print(2)
                    j = self.total_convs - i - 1
                    x = x + past_out[j]

        #x = x.view(*orig_shape)


        #z0 = z.view(bs, self.z_dim, 1, 1).repeat(1, 1, self.side_len, self.side_len)

        #x = torch.cat((x, z0), dim=1)

        x = self.out_l(x)



        mean = x[:, :self.out_channels // 2]
        logvar = x[:, self.out_channels // 2:] * 4 - 2

        x = torch.randn(bs, self.out_channels // 2, self.side_len, self.side_len).cuda()

        x = x * torch.exp(logvar * 0.5) + mean

        #z = z * torch.exp(log_var) + mean

        x = self.final(x)

        #if training:
        #    return x, torch.mean(logvar)#, z_mean, z_logvar

        return x

    """
    def gen_img(self, batch_size=32):

        x = torch.zeros([batch_size, self.in_channels, self.canvas_size]).cuda()

        for i in range(self.canvas_size):
            l = self.forward(x.view(-1, self.in_channels, self.side_len, self.side_len))
            out = sample_from_logistic_mix(l)

            out = out.view(-1, self.in_channels, self.canvas_size)

            x[:, :, i] = out[:, :, i]
            print(i + 1, "done")

        return x
    """


if __name__ == '__main__':



    inp = torch.randn((32, 3, 16, 16)).cuda()

    x = torch.randn((32, 64, 16, 16)).cuda()

    #out = conv(x)

    #print(out.shape)

    #attn = AttentionBlock(64, 16, 64, 32 * 32).cuda()

    #x = x.view(32, 64, 32 * 32)

    #out = attn(x)

    #print(out.shape)

    model = ImgAttendModel(side_len=16)

    total = 0
    for p in model.parameters():
      total += np.prod(p.shape)
      print(p.shape, np.prod(p.shape))
      
    print(total)

    for i in range(16 * 16):
        out = model(inp)
        print(i + 1, "done")

    #print(out.shape)

    pos = get_pos_embeddings(16)

    print(pos)
    print(pos.shape)
