import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
from itertools import islice, combinations

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    #import pdb; pdb.set_trace()
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

def ranking_loss(risks, times, c, phi, reduction):
    batch_size = len(times)
    if batch_size == 1:
        raise NotImplementedError("Batch size must be at least 2")
    events = 1 - c
    ##############################
    # determine comparable pairs #
    ##############################
    more_risky = []
    less_risky = []
    #import pdb; pdb.set_trace()
    for (idx_a, idx_b) in combinations(range(batch_size), 2):
        time_a, event_a = times[idx_a], events[idx_a]
        time_b, event_b = times[idx_b], events[idx_b]

        if time_a < time_b and event_a:
            # a and b are comparable, a is more risky
            more_risky.append(risks[idx_a])
            less_risky.append(risks[idx_b])

        elif time_b < time_a and event_b:
            # a and b are comparable, b is more risky
            more_risky.append(risks[idx_b])
            less_risky.append(risks[idx_a])

    # if there are no comparable pairs then just return zero
    if len(less_risky) == 0:
        return torch.zeros(1, requires_grad=True).to(device)

    more_risky = torch.stack(more_risky).to(device)
    less_risky = torch.stack(less_risky).to(device)

    # compute approximate c indices
    r = more_risky - less_risky
    if phi == 'sigmoid':
        approx_c_indices = torch.sigmoid(r)
    elif phi == 'relu':
        approx_c_indices = torch.relu(r)

    # negative mean/sum of c-indices
    if reduction == 'mean':
        return -approx_c_indices.mean()
    elif reduction == 'sum':
        return -approx_c_indices.sum()


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)

class CoxSurvLoss(object):
    def __call__(self, risks, times, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(times)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)

        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = times[j] >= times[i] 

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = risks.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        return loss_cox


class RankingSurvLoss(object):
    def __init__(self, phi = 'sigmoid', reduction = 'mean'):
        super().__init__()
        self.phi = phi
        self.reduction = reduction
    def __call__(self, risks, times, c):
        ranking_ls = ranking_loss(risks, times, c, self.phi, self.reduction)
        return ranking_ls

class RankingNLLSurvLoss(object):
    def __init__(self, phi = 'sigmoid', reduction = 'mean', alpha=0.15, nll_ratio = 0.5):
        self.alpha = alpha
        self.phi = phi
        self.reduction = reduction
        self.nll_ratio = nll_ratio

    def __call__(self,hazards, risks, S ,Y, c, alpha=None):
        ranking_ls = ranking_loss(risks, Y, c, self.phi, self.reduction)
        if alpha is None:
            nll_ls = nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            nll_ls = nll_loss(hazards, S, Y, c, alpha=alpha)
        return ranking_ls + nll_ls * self.nll_ratio
