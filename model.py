import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

from loss import *

import functools

@functools.lru_cache(maxsize=32)
def get_causal_mask(canvas_size, see_self=True):
    mask = torch.zeros([canvas_size, canvas_size])
    for i in range(canvas_size):
        mask[i, :i + 1] = 1.
    return mask.cuda()

def get_pos_enc_table(side_len, emb_dim=16):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(side_len)])
    

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_pos_embeddings(side_len, emb_dim=16):

    x = get_pos_enc_table(side_len, emb_dim).view(1, emb_dim, side_len, 1).repeat(1, 1, 1, side_len)

    y = get_pos_enc_table(side_len, emb_dim).view(1, emb_dim, 1, side_len).repeat(1, 1, side_len, 1)

    #print(x.shape, y.shape)

    return torch.cat((x, y), dim=1).cuda()


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(x)
#causal conv
class CausalConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernels, see_center=True, dilation=1, stride=1, groups=1, sep=False):

        gr1 = in_channels if sep else groups

        super(CausalConv, self).__init__(in_channels, out_channels if not sep else in_channels, kernels, padding=(kernels+(kernels-1)*(dilation-1))//2, dilation=dilation, stride=stride, groups=gr1)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kh, kw = self.weight.size()
        yc, xc = kh // 2, kw // 2

        pre_mask = np.ones(self.weight.shape)

        pre_mask[:, :, yc, xc + see_center:] = 0
        pre_mask[:, :, yc + 1:] = 0

        self.mask = torch.Tensor(pre_mask)

        #print(self.mask[0][0])
        #input()

        if sep:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, groups=groups)

        self.sep = sep

    def forward(self, x):
        #torch.set_printoptions(threshold=5000, precision=2)
        #print(self.weight.data.shape)
        #print(self.weight.data[0])
        #input()
        self.weight.data *= self.mask
        #print(self.weight.data.shape)
        #print(self.weight.data[0])
        #input()
        x = super(CausalConv, self).forward(x)
        if self.sep:
            x = self.conv1x1(x)
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

class MultiDilConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernels=5, see_center=True):

        super(MultiDilConvBlock, self).__init__()

        self.conv_dil_1 = CausalConv(in_channels, out_channels // 2, kernels, dilation=1, see_center=see_center)

        self.conv_dil_2 = CausalConv(in_channels, out_channels // 4, kernels, dilation=2, see_center=see_center)

        self.conv_dil_3 = CausalConv(in_channels, out_channels // 4, kernels, dilation=3, see_center=see_center)

    def forward(self, x):

        x1 = self.conv_dil_1(x)
        x2 = self.conv_dil_2(x)
        x3 = self.conv_dil_3(x)

        x = torch.cat((x1, x2, x3), dim=1)

        return x

#attention block
class AttentionBlock(nn.Module):

    def __init__(self, in_channels, K, V, side_len, nonlin, bg=None, emb_dim=16, heads=1):

        super(AttentionBlock, self).__init__()


        self.side_len = side_len

        self.K = K
        self.V = V

        self.emb_dim = emb_dim
        #self.bg = bg

        #if not self.bg:
        #    self.bg = torch.zeros([in_channels, canvas_size])

        self.mask = get_causal_mask(side_len ** 2)

        self.q_conv = nn.Sequential(nn.Conv1d(in_channels + emb_dim * 2, K, 1), nonlin)

        self.k_conv = nn.Sequential(nn.Conv1d(in_channels + emb_dim * 2, K, 1), nonlin)

        self.v_conv = nn.Sequential(nn.Conv1d(in_channels + emb_dim * 2, V, 1), nonlin)

        self.fc = nn.Sequential(nn.Conv1d(V, V, 1), nonlin)

        self.out_channels = V

    def forward(self, x):

        orig_shape = x.shape

        pos_emb = get_pos_embeddings(self.side_len, self.emb_dim).repeat(orig_shape[0], 1, 1, 1)

        x0 = torch.cat((x, pos_emb), dim=1)

        x0 = x0.view(orig_shape[0], orig_shape[1] + self.emb_dim * 2, -1)

        q = self.q_conv(x0)

        k = self.k_conv(x0)

        v = self.v_conv(x0)

        attn = torch.bmm(q.transpose(-2, -1), k) / (self.K ** 0.5)

        mask = self.mask.to(attn.device)

        attn_masked = attn * mask + (1 - mask) * (-10000.)

        attn_masked = nn.functional.softmax(attn_masked, dim=-1) * mask

        #use attn_masked
        out1 = torch.bmm(attn_masked, v.transpose(-2, -1)).transpose(-2, -1)

        out2 = out1 + self.fc(out1)

        out2 = out1.view(orig_shape[0], self.out_channels, orig_shape[2], orig_shape[3])

        return out2

class AttentionBlockConv(nn.Module):

    def __init__(self, in_channels, K, V, side_len, nonlin, bg=None, emb_dim=16, conv_side=2):

        super(AttentionBlockConv, self).__init__()


        self.side_len = side_len
        self.conv_side = conv_side

        self.K = K
        self.V = V

        self.emb_dim = emb_dim

        self.mask = get_causal_mask((side_len // conv_side) ** 2)

        self.q_conv = nn.Sequential(nn.Conv2d(in_channels + emb_dim * 2, K, conv_side, stride=conv_side, padding=conv_side - 1), nonlin)

        self.k_conv = nn.Sequential(nn.Conv2d(in_channels + emb_dim * 2, K, conv_side, stride=conv_side, padding=conv_side - 1), nonlin)

        self.v_conv = nn.Sequential(nn.Conv2d(in_channels + emb_dim * 2, V, conv_side, stride=conv_side, padding=conv_side - 1), nonlin)

        self.fc = nn.Sequential(nn.Conv1d(V, V, 1), nonlin)

        self.upsample = nn.ConvTranspose2d(V, V, 2, stride=2)

        self.out_channels = V

    def forward(self, x):

        orig_shape = x.shape


        pos_emb = get_pos_embeddings(self.side_len, self.emb_dim).repeat(orig_shape[0], 1, 1, 1)

        x0 = torch.cat((x, pos_emb), dim=1)

        #print(x0.shape)
        #x0 = x0.view(orig_shape[0], orig_shape[1] + self.emb_dim * 2, -1)
        #print(self.q_conv(x0).shape)

        q = self.q_conv(x0)[:, :, :-(self.conv_side - 1), :-(self.conv_side - 1)].contiguous().view(orig_shape[0], self.K, -1)

        #print(q.shape)

        k = self.k_conv(x0)[:, :, :-(self.conv_side - 1), :-(self.conv_side - 1)].contiguous().view(orig_shape[0], self.K, -1)

        v = self.v_conv(x0)[:, :, :-(self.conv_side - 1), :-(self.conv_side - 1)].contiguous().view(orig_shape[0], self.V, -1)

        attn = torch.bmm(q.transpose(-2, -1), k) / (self.K ** 0.5)

        mask = self.mask.to(attn.device)

        attn_masked = attn * mask + (1 - mask) * (-10000.)

        attn_masked = nn.functional.softmax(attn_masked, dim=-1) * mask

        #use attn_masked
        out1 = torch.bmm(attn_masked, v.transpose(-2, -1)).transpose(-2, -1)

        out2 = out1.view(orig_shape[0], self.out_channels, orig_shape[2] // self.conv_side, orig_shape[3] // self.conv_side)

        out2 = x[:, :self.V] + self.upsample(out2)

        return out2

class AttentionBlockPersistent(nn.Module):

    def __init__(self, in_channels, K, V, side_len, nonlin, bg=None, emb_dim=16):

        super(AttentionBlockPersistent, self).__init__()


        self.side_len = side_len

        self.K = K
        self.V = V

        self.emb_dim = emb_dim
        #self.bg = bg

        #if not self.bg:
        #    self.bg = torch.zeros([in_channels, canvas_size])

        self.mask = get_causal_mask(side_len ** 2)

        self.q = nn.Sequential(nn.Conv1d(in_channels + emb_dim * 2, K, 1), nonlin)

        self.f_k = nn.Sequential(nn.Conv1d(in_channels + K + emb_dim * 2, K, 1), nn.Sigmoid())

        #self.i_k = nn.Sequential(nn.Conv1d(in_channels + K + emb_dim * 2, K, 1), nn.Sigmoid())

        #self.o_k = nn.Sequential(nn.Conv1d(in_channels + K + emb_dim * 2, K, 1), nn.Sigmoid())

        self.c_k = nn.Sequential(nn.Conv1d(in_channels + K + emb_dim * 2, K, 1), nonlin)

        self.f_v = nn.Sequential(nn.Conv1d(in_channels + V + emb_dim * 2, V, 1), nn.Sigmoid())

        #self.i_v = nn.Sequential(nn.Conv1d(in_channels + V + emb_dim * 2, V, 1), nn.Sigmoid())

        #self.o_v = nn.Sequential(nn.Conv1d(in_channels + V + emb_dim * 2, V, 1), nn.Sigmoid())

        self.c_v = nn.Sequential(nn.Conv1d(in_channels + V + emb_dim * 2, V, 1), nonlin)

        self.fc = nn.Sequential(nn.Conv1d(V, V, 1), nonlin)

        self.out_channels = V

    def forward(self, x, k, v):

        #x - current propogated inputs
        #k, v - persistent dictionary of keys and values
        #f_k = W_f(x, k) sigm
        #i_k = W_i(x, k) sigm
        #o_k = W_o(x, k) sigm
        #c_k = W_c(x, k)
        #k_new = o * (f * k + i * c)
        #same for v
        #then
        #q = W_q(x)
        #x_new = Attn(q, k_new, v_new)
        #return x_new, k_new, v_new

        orig_shape = x.shape

        pos_emb = get_pos_embeddings(self.side_len, self.emb_dim).repeat(orig_shape[0], 1, 1, 1)

        x0 = torch.cat((x, pos_emb), dim=1)

        x0 = x0.view(orig_shape[0], orig_shape[1] + self.emb_dim * 2, -1)

        x0_with_k = torch.cat((x0, k), dim=1)#.view(orig_shape[0], x0.shape[1] + k.shape[1], -1)

        x0_with_v = torch.cat((x0, v), dim=1)#.view(orig_shape[0], x0.shape[1] + v.shape[1], -1)

        f_k = self.f_k(x0_with_k)

        #i_k = self.i_k(x0_with_k)

        #o_k = self.o_k(x0_with_k)

        c_k = self.c_k(x0_with_k)

        #k_new = o_k * (f_k * k + i_k * c_k)
        k_new = f_k * k + c_k

        f_v = self.f_v(x0_with_v)

        #i_v = self.i_v(x0_with_v)

        #o_v = self.o_v(x0_with_v)

        c_v = self.c_v(x0_with_v)

        v_new = f_v * v + c_v

        #v_new = o_v * (f_v * v + i_v * c_v)

        q = self.q(x0)

        attn = torch.bmm(q.transpose(-2, -1), k) / (self.K ** 0.5)

        mask = self.mask.to(attn.device)

        attn_masked = attn * mask + (1 - mask) * (-10000.)

        attn_masked = nn.functional.softmax(attn_masked, dim=-1) * mask

        #use attn_masked
        out1 = torch.bmm(attn_masked, v.transpose(-2, -1)).transpose(-2, -1)

        out2 = out1 + self.fc(out1)

        out2 = out1.view(orig_shape[0], self.out_channels, orig_shape[2], orig_shape[3])

        return out2, k_new, v_new


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

    def __init__(self, side_len=64, in_channels=3, channels=64, out_channels=100, total_convs=8, kernels=5, use_z=True, z_dim=16, nonlin=nn.LeakyReLU()):

        super(PixelCNN, self).__init__()

        self.canvas_size = side_len * side_len
        self.in_channels = in_channels
        self.side_len = side_len
        self.total_convs = total_convs
        self.out_channels = out_channels
        self.z_dim = z_dim

        #print(channels * side_len * side_len // (2**6))

        self.use_z = use_z

        if use_z == False:
            self.z_dim = 0

        if use_z:

            self.z_net = nn.Sequential(nn.Conv2d(in_channels, channels, 5, stride=2, padding=2), nn.BatchNorm2d(channels), nonlin,
                                       nn.Conv2d(channels, channels, 5, stride=2, padding=2), nn.BatchNorm2d(channels), nonlin,
                                       nn.Conv2d(channels, channels, 5, stride=2, padding=2), nn.BatchNorm2d(channels), nonlin)

            self.z_net_lin = nn.Sequential(nn.Linear(channels * side_len * side_len // (2**6), 128), nonlin, nn.Linear(128, z_dim), nn.Tanh())

            self.z_drop = nn.Dropout(p=0.3)

        self.in_l = nn.Sequential(CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nonlin)

        self.layers = []

        for i in range(total_convs):
            #l = MultiDilConvBlock(channels, channels, kernels)
            in_ch = channels
            if i > self.total_convs // 2:
                in_ch *= 2
            elif i == 0:
                in_ch += self.z_dim

            if i % 2 == 0:
                l = CausalConv(in_ch, channels, kernels, see_center=True, sep=True)
            else:
                l = AttentionBlock(in_ch, channels, channels, side_len, nonlin)
            self.layers.append(nn.Sequential(l, nn.BatchNorm2d(channels), nonlin))
            #self.layers.append(nn.Sequential(l, AttentionBlock(channels, K=16, V=channels, side_len=self.side_len), nn.BatchNorm2d(channels), nn.ELU()))
        self.out_l = nn.Sequential(nn.Conv2d(channels * 2, out_channels, 1))

        self.layers = nn.Sequential(*self.layers)


        self.final_pass = nn.Sequential(nn.ConvTranspose2d(in_channels, channels, kernels, stride=2, padding=kernels//2, output_padding=1), nn.BatchNorm2d(channels), nonlin,
                                        nn.ConvTranspose2d(channels, channels, kernels, stride=1, padding=kernels//2, output_padding=0), nn.BatchNorm2d(channels), nonlin,
                                        nn.ConvTranspose2d(channels, channels, kernels, stride=1, padding=kernels//2, output_padding=0), nn.BatchNorm2d(channels), nonlin,
                                        nn.Conv2d(channels, channels, kernels, stride=1, padding=kernels//2), nn.BatchNorm2d(channels), nonlin,
                                        nn.ConvTranspose2d(channels, channels, kernels, stride=2, padding=kernels//2, output_padding=1), nn.BatchNorm2d(channels), nonlin,
                                        nn.ConvTranspose2d(channels, channels, kernels, stride=1, padding=kernels//2, output_padding=0), nn.BatchNorm2d(channels), nonlin,
                                        nn.Conv2d(channels, channels, kernels, stride=1, padding=kernels//2), nn.BatchNorm2d(channels), nonlin,
                                        nn.Conv2d(channels, in_channels, 1), nn.Tanh())

        self.cuda()


    def forward(self, x, return_z=False, set_z=None):

        bs = x.shape[0]

        if self.use_z:

            if set_z is None:

                z = self.z_net(x)

                #print(z.shape)

                z = z.view(bs, -1)

                z = self.z_net_lin(z) * 3

            else:

                z = set_z

            z = self.z_drop(z)

            #print(z.shape)

            z0 = z.view(bs, self.z_dim, 1, 1).repeat([1, 1, self.side_len, self.side_len])

        x = self.in_l(x)

        if self.use_z:

            x = torch.cat((x, z0), dim=1)

        i = 0

        past_out = []


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
                    x = torch.cat((x, past_out[j]), dim=1)

        x = self.out_l(x)

        if self.use_z and return_z:
            return x, z

        return x

class PixelAttendPers(nn.Module):

    def __init__(self, side_len=64, in_channels=3, channels=64, out_channels=100, total_layers=8, kernels=5, z_dim=16, nonlin=nn.LeakyReLU(), K=64):

        super(PixelAttendPers, self).__init__()

        self.canvas_size = side_len * side_len
        self.in_channels = in_channels
        self.side_len = side_len
        self.total_layers = total_layers
        self.out_channels = out_channels
        self.z_dim = z_dim
        self.K = K
        self.V = channels

        #print(channels * side_len * side_len // (2**6))

        self.z_net = nn.Sequential(nn.Conv2d(in_channels, channels, 5, stride=2, padding=2), nn.BatchNorm2d(channels), nonlin,
                                   nn.Conv2d(channels, channels, 5, stride=2, padding=2), nn.BatchNorm2d(channels), nonlin,
                                   nn.Conv2d(channels, channels, 5, stride=2, padding=2), nn.BatchNorm2d(channels), nonlin)

        self.z_net_lin = nn.Sequential(nn.Linear(channels * side_len * side_len // (2**6), 128), nonlin, nn.Linear(128, z_dim), nn.Tanh())

        self.in_l = nn.Sequential(CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nonlin)

        self.layers = []

        for i in range(total_layers):
            in_ch = channels
            l = AttentionBlockPersistent(in_ch, self.K, self.V, side_len, nonlin)
            self.layers.append(l)
            self.layers.append(nn.BatchNorm2d(channels))
            self.layers.append(nonlin)
            #self.layers.append(nn.Sequential(l, nn.BatchNorm2d(channels), nonlin))
            #self.layers.append(nn.Sequential(l, AttentionBlock(channels, K=16, V=channels, side_len=self.side_len), nn.BatchNorm2d(channels), nn.ELU()))
        self.out_l = nn.Sequential(nn.Conv2d(channels, out_channels, 1))

        self.layers = nn.Sequential(*self.layers)

        self.cuda()


    def forward(self, x, return_z=False, set_z=None):

        bs = x.shape[0]

        """

        if set_z is None:

            z = self.z_net(x)

            #print(z.shape)

            z = z.view(bs, -1)

            z = self.z_net_lin(z) * 3

        else:

            z = set_z

        #print(z.shape)

        z0 = z.view(bs, self.z_dim, 1, 1).repeat([1, 1, self.side_len, self.side_len])

        """
        z = torch.randn([bs, self.z_dim]).cuda()

        x = self.in_l(x)

        #x = torch.cat((x, z0), dim=1)

        i = 0

        
        K = self.K
        V = self.V

        k = torch.zeros([bs, K, self.side_len * self.side_len]).cuda()
        v = torch.zeros([bs, V, self.side_len * self.side_len]).cuda()

        for i, l in enumerate(self.layers):
            if i % 3 == 0:
                x, k, v = l(x, k, v)
            else:
                x = l(x)

        x = self.out_l(x)

        if return_z:
            return x, z

        return x


class PixelCNNProg(nn.Module):

    def __init__(self, side_len=64, in_channels=3, channels=64, out_channels=100, total_attn=7, kernels=5, levels=3):

        super(PixelCNNProg, self).__init__()

        self.canvas_size = side_len * side_len
        self.in_channels = in_channels
        self.side_len = side_len
        self.total_attn = total_attn
        self.out_channels = out_channels
        self.levels = levels

        self.downsample = nn.AvgPool2d(2)#CausalConv(in_channels, in_channels, 3, see_center=False, dilation=2, stride=2)

        #self.in_l = nn.Sequential(CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nn.ELU())
        self.level_in = nn.ModuleList()

        self.level_out = nn.ModuleList()

        self.level_latent = nn.ModuleList()

        self.level_final = nn.ModuleList()

        self.level_up = nn.ModuleList()

        nonlin = nn.LeakyReLU()

        conv_rep = 3 + (3 - levels)

        for i in range(self.levels):

            conv_rep += 1

            gr = 1
            sep = True

            level_in_cur = [CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nonlin]

            for j in range(conv_rep):
                if j % 2 == 1:
                    level_in_cur.append(AttentionBlockConv(channels * (2 if (j == 0 and i > 0) else 1), channels, channels, self.side_len // (2 ** (self.levels - 1 - i)), nonlin))
                else:
                    level_in_cur.append(CausalConv(channels * (2 if (j == 0 and i > 0) else 1), channels, kernels, dilation=1, groups=gr, sep=sep))
                level_in_cur.append(nn.BatchNorm2d(channels))
                level_in_cur.append(nonlin)

            self.level_in.append(nn.Sequential(*level_in_cur))

            level_out_cur = [CausalConv(channels, channels, kernels, groups=gr, sep=sep), nn.BatchNorm2d(channels), nonlin]

            for j in range(conv_rep):
                level_out_cur.append(CausalConv(2 * channels, channels, kernels, dilation=1 + j % 2, groups=gr, sep=sep))
                level_out_cur.append(nn.BatchNorm2d(channels))
                level_out_cur.append(nonlin)

            self.level_out.append(nn.Sequential(*level_out_cur))


            #if i > 0:
            #self.level_in.append(nn.Sequential(CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nn.ELU(),
            #                            ConvBlock(channels, kernels), nn.BatchNorm2d(channels), ConvBlock(channels, kernels), nn.BatchNorm2d(channels)))

            #self.level_out.append(nn.Sequential(CausalConv(channels * (2 if i > 0 else 1), channels, kernels), nn.BatchNorm2d(channels), nn.ELU(),
            #                             ConvBlock(channels, kernels), nn.BatchNorm2d(channels), ConvBlock(channels, kernels), nn.BatchNorm2d(channels)))
            #else:
            #    self.level_in.append(nn.Sequential(CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nn.ELU(),
            #                            CausalConv(channels, channels, kernels), nn.BatchNorm2d(channels), nn.ELU()))
            #    self.level_out.append(nn.Sequential(CausalConv(channels, channels, kernels), nn.BatchNorm2d(channels), nn.ELU(),
            #                             CausalConv(channels, channels, kernels), nn.BatchNorm2d(channels), nn.ELU()))

            self.level_latent.append(nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.ELU(), nn.Conv2d(channels, out_channels, 1))) #Tanh
            #self.level_latent.append(nn.Sequential(nn.Conv2d(channels * 2, out_channels, 1), nn.Tanh())) #Tanh

            self.level_final.append(nn.Sequential(nn.Conv2d(out_channels // 2, channels, 1), nonlin, nn.Conv2d(channels, in_channels, 1), nn.Sigmoid()))

            if i < self.levels - 1:
                self.level_up.append(nn.ConvTranspose2d(channels * 2, channels, 2, stride=2))


        """
        self.level_3_in = nn.Sequential(CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nn.ELU(),
                                        CausalConv(channels, channels, kernels), nn.BatchNorm2d(channels), nn.ELU())

        self.level_3_out = nn.Sequential(CausalConv(channels, channels, kernels), nn.BatchNorm2d(channels), nn.ELU(),
                                         CausalConv(channels, channels, kernels), nn.BatchNorm2d(channels), nn.ELU())

        self.level_3_up = nn.ConvTranspose2d(channels, channels, 2, stride=2)

        self.level_2_in = nn.Sequential(CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nn.ELU(),
                                        ConvBlock(channels, kernels), nn.BatchNorm2d(channels), ConvBlock(channels, kernels), nn.BatchNorm2d(channels))

        self.level_2_out = nn.Sequential(CausalConv(channels * 2, channels, kernels), nn.BatchNorm2d(channels), nn.ELU(),
                                         ConvBlock(channels, kernels), nn.BatchNorm2d(channels), ConvBlock(channels, kernels), nn.BatchNorm2d(channels))

        self.level_2_up = nn.ConvTranspose2d(channels, channels, 2, stride=2)

        self.level_1_in = nn.Sequential(CausalConv(in_channels, channels, kernels, see_center=False), nn.BatchNorm2d(channels), nn.ELU(),
                                        CausalConv(channels, channels, kernels), nn.BatchNorm2d(channels), nn.ELU())

        self.level_1_out = nn.Sequential(CausalConv(channels * 2, channels, kernels), nn.BatchNorm2d(channels), nn.ELU(),
                                         CausalConv(channels, channels, kernels), nn.BatchNorm2d(channels), nn.ELU())
        """

        self.layers_attn = []

        for i in range(total_attn):
            in0 = channels
            if i > self.total_attn // 2:
                in0 *= 2
            self.layers_attn.append(nn.Sequential(#CausalConv(channels, channels, kernels),
                                                  AttentionBlock(in0, K=channels, V=channels, side_len=self.side_len // (2 ** (self.levels - 1)), nonlin=nonlin),
                                                  nn.BatchNorm2d(channels), nonlin))
            #self.layers_attn.append(nn.Sequential(AttentionBlock(channels, K=channels, V=channels, side_len=self.side_len // 4), nn.BatchNorm2d(channels), nn.ELU()))


        #self.out_l = nn.Sequential(nn.Conv2d(channels, out_channels, 1), nn.Tanh())

        #self.out_2 = nn.Sequential(nn.Conv2d(channels, out_channels, 1), nn.Tanh())

        #self.out_1 = nn.Sequential(nn.Conv2d(channels, out_channels, 1), nn.Tanh())

        #self.final = nn.Sequential(nn.Conv2d(out_channels // 2, channels, 1), nn.ELU(), nn.Conv2d(channels, in_channels, 1), nn.Sigmoid())

        #self.final_2 = nn.Sequential(nn.Conv2d(out_channels // 2, channels, 1), nn.ELU(), nn.Conv2d(channels, in_channels, 1), nn.Sigmoid())

        self.layers_attn = nn.Sequential(*self.layers_attn)

        self.cuda()


    def forward(self, x, level=1, training=False, var_mult=1.):


        #x is list/tuple with resolutions starting from lowest

        """
        if training:
            x0 = x.view(-1, self.in_channels, self.side_len, self.side_len)
            x = []
            for i in range(self.levels):
                x.append(x0)
                x0 = self.downsample(x0)

            x = x[::-1]
            #x1 = x
            #x2 = self.downsample(x1)
            #x3 = self.downsample(x2)

            #x1 = x1.view(-1, self.in_channels, self.side_len)
        """

        if (not training) and len(x[0].shape) == 3:

            x0 = x
            x = []
            for i in x0:
                cur_side = int(i.shape[2] ** 0.5)
                x.append(i.view(-1, self.in_channels, cur_side, cur_side))
                cur_side //= 2

        #x2 = x2.view(-1, self.in_channels, self.side_len // 2, self.side_len // 2)
        #x3 = x3.view(-1, self.in_channels, self.side_len // 4, self.side_len // 4)

            #print(x1.shape, x2.shape, x3.shape)
        bs = x[0].shape[0]

        outs = []

        pix = []


        for i, x0 in enumerate(x):

            outer_past = []

            for j, l in enumerate(self.level_in[i]):

                #print(j, l)
                #print(x0.shape)

                x0 = l(x0)

                if j % 3 == 2:
                    outer_past.append(x0)

                if i > 0 and j == 2:
                    prev = self.level_up[i - 1](outs[-1])
                    x0 = torch.cat((x0, prev), dim=1)


            #x0 = self.level_in[i](x0)

            if i == 0:
                past_out = []

                j = 0

                for j, l in enumerate(self.layers_attn):
                    
                    if j < self.total_attn // 2:
                        x0 = l(x0)
                        past_out.append(x0)
                        #print(1)
                    elif j > self.total_attn // 2:
                        #print(2)
                        k = self.total_attn - j - 1
                        x0 = torch.cat((x0, past_out[k]), dim=1)
                        x0 = l(x0)
                    else:
                        x0 = l(x0)
            #else:
            #    prev = self.level_up[i - 1](outs[-1])
            #    x0 = torch.cat((x0, prev), dim=1)


            for j, l in enumerate(self.level_out[i]):

                #print(j, l)
                #print(x0.shape)
                
                x0 = l(x0)

                if j % 3 == 2:
                    x0 = torch.cat((x0, outer_past[-1 - (j // 3)]), dim=1)

            #x0 = self.level_out[i](x0)

            outs.append(x0)


            

            x0 = self.level_latent[i](x0)

          
            pix.append(x0)

            """
            mean = x0[:, :self.out_channels // 2]
            logvar = x0[:, self.out_channels // 2:] * 3

            if i == 0:
                lv_mean = logvar.mean()

            side_len = x0.shape[2]

            out = torch.randn(bs, self.out_channels // 2, side_len, side_len).cuda()

            out = out * torch.exp(logvar * 0.5) * var_mult + mean

            out = self.level_final[i](out)

            pix.append(out)
            """
            

            

        """
        if level <= 3:

            x3 = self.level_3_in(x3)

            past_out = []

            for i, l in enumerate(self.layers_attn):
                
                if i < self.total_attn // 2:
                    x3 = l(x3)
                    i += 1
                    past_out.append(x3)
                    #print(1)
                elif i > self.total_attn // 2:
                    #print(2)
                    j = self.total_attn - i - 1
                    x3 = torch.cat((x3, past_out[j]), dim=1)
                    x3 = l(x3)
                else:
                    x3 = l(x3)

            x3 = self.level_3_out(x3)

            #x3_out = self.out_l(x3)

            outs.append(x3)

        if level <= 2:

            x2 = self.level_2_in(x2)

            x3 = self.level_3_up(x3)

            x2 = torch.cat((x2, x3), dim=1)

            x2 = self.level_2_out(x2)

            #x2_out = self.out_2(x2)

            outs.append(x2)

        if level <= 1:

            x1 = self.level_1_in(x1)

            x2 = self.level_2_up(x2)

            x1 = torch.cat((x1, x2), dim=1)

            x1 = self.level_2_out(x1)

            #x1_out = self.out_1(x1)

            outs.append(x1)

        

        pix = []

        lv_mean = None

        for i, out in enumerate(outs):

            out = self.out_l(out)

            mean = out[:, :self.out_channels // 2]
            logvar = out[:, self.out_channels // 2:] * 3

            if i == 0:
                lv_mean = logvar.mean()

            side_len = out.shape[2]

            out = torch.randn(bs, self.out_channels // 2, side_len, side_len).cuda()

            out = out * torch.exp(logvar * 0.5) + mean

            if i == 0:
                pix.append(self.final(out))
            else:
                pix.append(self.final_2(out))

        """

        return pix


if __name__ == '__main__':

    side_len = 16
    K = 16
    V = 64

    test = 4

    if test == 1:

        inp = torch.randn((32, 3, side_len, side_len)).cuda()

        a = AttentionBlock(3, K, V, side_len, nn.ReLU()).cuda()

        out = a(inp)

        print(out.shape)

    if test == 2:

        inp = torch.randn((32, 3, side_len, side_len)).cuda()

        a = AttentionBlockConv(3, K, V, side_len, nn.ReLU()).cuda()

        out = a(inp)

        print(out.shape)

    elif test == 3:

        inp = torch.randn((32, 3, side_len, side_len)).cuda()

        k = torch.randn((32, K, side_len * side_len)).cuda()
        v = torch.randn((32, V, side_len * side_len)).cuda()


        a = AttentionBlockPersistent(3, K, V, side_len, nn.ReLU()).cuda()

        out, k_new, v_new = a(inp, k, v)

        print(out.shape, k_new.shape, v_new.shape)

    else:

        model = PixelCNN(side_len=side_len, kernels=9, in_channels=3, channels=128, out_channels=100, total_convs=12)

        inp = torch.randn((32, 3, side_len, side_len)).cuda()

        out = model(inp)

        print(out.shape)

        out_sample = sample_from_logistic_mix(out)

        print(out_sample.shape)

        out_final = model.final_pass(out_sample)

        print(out_final.shape)