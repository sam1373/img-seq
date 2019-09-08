import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim 


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
   
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10) 
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    
    return -torch.sum(log_sum_exp(log_probs))

"""

def log_sum_exp(x):
    axis = len(x.shape) - 1
    m, _ = torch.max(x, axis)
    m2, _ = torch.max(x, axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), axis))


def log_prob_from_logits(x):
    axis = len(x.shape) - 1
    m, _ = torch.max(x, axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), axis, keepdim=True))

def discretized_mix_logistic_loss(x, l):
    #x is rescaled to [-1, 1]
    #print(x.min(), x.max(), x.mean())
    #print(l.min(), l.max(), l.mean())
    #input()

    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)

    xs = x.shape
    ls = l.shape

    n_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :n_mix]

    l = l[:, :, :, n_mix:].view(xs + (n_mix * 3,))

    means = l[:, :, :, :, :n_mix]
    log_scales = l[:, :, :, :, n_mix:2*n_mix]
    log_scales = log_scales.clamp(min=-7)
    coeffs = torch.tanh(l[:, :, :, :, 2*n_mix:3*n_mix])

    x = x.view(xs + (1,))
    #print(x[:, 0].shape, means[:, 1].shape, coeffs[:, 0].shape)

    m_shape = list(xs) + [n_mix]
    m_shape[-2] = 1

    m1 = means[:, :, :, 0]
    m2 = means[:, :, :, 1] + coeffs[:, :, :, 0] * x[:, :, :, 0]
    m3 = means[:, :, :, 2] + coeffs[:, :, :, 1] * x[:, :, :, 0] + coeffs[:, :, :, 2] * x[:, :, :, 1]

    m1 = m1.view(m_shape)
    m2 = m2.view(m_shape)
    m3 = m3.view(m_shape)

    means = torch.cat((m1, m2, m3), dim=-2)

    #print(x.shape, means.shape)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)

    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)

    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)

    cdf_delta = cdf_plus - cdf_min


    cdf_delta = cdf_delta.clamp(min=1e-12)

    #print(cdf_delta.min())

    log_probs = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta)))

    #print(log_probs.shape)
    #print(torch.sum(log_probs, dim=-2).shape)
    #print(log_prob_from_logits(logit_probs).shape)

    log_probs = torch.sum(log_probs, dim=-2) + log_prob_from_logits(logit_probs)



    return -torch.sum(torch.logsumexp(log_probs, dim=-1))

"""

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1).cuda()
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def sample_from_logistic_mix(l):
    #gets [bs, mix_num * 10, side, side]
    
    l = l.permute(0, 2, 3, 1)

    ls = l.shape
    xs = ls[:-1] + (3,)


    n_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :n_mix]

    l = l[:, :, :, n_mix:].view(xs + (n_mix * 3,))

    sel = to_one_hot(torch.argmax(logit_probs - torch.log(-torch.log(torch.rand(logit_probs.shape).cuda())), 3), n_dims=n_mix)
    
    sel = sel.view(xs[:-1] + (1, n_mix))

    means = torch.sum(l[:, :, :, :, :n_mix] * sel, dim=4)
    log_scales, _ = torch.max(l[:, :, :, :, n_mix:2*n_mix] * sel, dim=4)
    log_scales = log_scales.clamp(min=-7)
    coeffs = torch.sum(torch.tanh(l[:, :, :, :, 2*n_mix:3*n_mix]), dim=4)

    u = torch.rand(means.shape).cuda()
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1 - u))

    x0 = torch.clamp(x[:, :, :, 0], -1., 1.) 
    x1 = torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1., 1.)
    x2 = torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1., 1.)

    xss = xs[:-1] + (1,)

    x0 = x0.view(xss)
    x1 = x1.view(xss)
    x2 = x2.view(xss)

    out = torch.cat((x0, x1, x2), dim=-1)

    return out.permute(0, 3, 1, 2)



def quant(img):
  #[bs, 3, side_len, side_len]

  img = img / 0.125

  


if __name__ == '__main__':


    x = torch.randn((32, 3, 32, 32)).cuda()

    l = torch.randn((32, 100, 32, 32)).cuda()

    print(discretized_mix_logistic_loss(x, l))
