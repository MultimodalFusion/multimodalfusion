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
from itertools import islice, combinations
import collections

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_features(batch):
    img = [item[0][None,...] for item in batch if item is not None]
    if img:
        img = torch.cat(img, dim = 0)
        slice_id = np.hstack([item[1] for item in batch if item is not None])
        x = np.hstack([item[2] for item in batch if item is not None])
        y = np.hstack([item[3] for item in batch if item is not None])
        return [img, slice_id , x, y]
    else:
        return[None] * 4


def collate_MIL_survival(batch):
    radio_features = torch.cat([item[0] for item in batch], dim = 0)
    path_features = torch.cat([item[1] for item in batch], dim = 0)
    genomic_features = torch.cat([item[2] for item in batch], dim = 0)
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    masks = {'radio':[],'path':[],'genomic':[]}
    for item in batch:
        masks['radio'].append(item[6]['radio'])
        masks['path'].append(item[6]['path'])
        masks['genomic'].append(item[6]['genomic'])
    return [radio_features, path_features, genomic_features, label, event_time,c,masks]


def get_simple_loader(dataset, batch_size=1):
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
    return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, task_type='classification', batch_size=1):
    """
        return either the validation loader or training loader 
    """
    if task_type == 'classification':
        collate = collate_MIL
    elif task_type == 'survival':
        collate = collate_MIL_survival
    else:
        raise NotImplementedError

    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate,shuffle=False, **kwargs)    
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate,shuffle=False, **kwargs)
            else:
                #loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate,shuffle=False, **kwargs)
                #loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate,shuffle=False, **kwargs)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)

        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate,shuffle=False, **kwargs)
            #loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate,shuffle=False, **kwargs)
            
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate,shuffle=False, **kwargs )

    return loader

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

def initialize_weights(module):
    
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.bias.data.zero_()
            nn.init.kaiming_normal_(m.weight,mode = 'fan_in',nonlinearity = 'linear')
    """


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = 0 if T_cont \in (-inf, 0), Y = 1 if T_cont \in [0, a_1),  Y = 2 if T_cont in [a_1, a_2), ..., Y = k if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = 0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=0) = 0
# h(0) = 0 ---> do not need to model
# S(0) = P(Y > 0 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 1,2,...,k
corresponding Y = 1, ..., k. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''
# def neg_likelihood_loss(hazards, Y, c):
#   batch_size = len(Y)
#   Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
#   c = c.view(batch_size, 1).float() #censorship status, 0 or 1
#   S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
#   # without padding, S(1) = S[0], h(1) = h[0]
#   S_padded = torch.cat([torch.ones_like(c), S], 1) #S(0) = 1, all patients are alive from (-inf, 0) by definition
#   # after padding, S(0) = S[0], S(1) = S[1], etc, h(1) = h[0]
#   #h[y] = h(1)
#   #S[1] = S(1)
#   neg_l = - c * torch.log(torch.gather(S_padded, 1, Y)) - (1 - c) * (torch.log(torch.gather(S_padded, 1, Y-1)) + torch.log(hazards[:, Y-1]))
#   neg_l = neg_l.mean()
#   return neg_l


# divide continuous time scale into k discrete bins in total,  T_cont \in {[0, a_1), [a_1, a_2), ...., [a_(k-1), inf)}
# Y = T_discrete is the discrete event time:
# Y = -1 if T_cont \in (-inf, 0), Y = 0 if T_cont \in [0, a_1),  Y = 1 if T_cont in [a_1, a_2), ..., Y = k-1 if T_cont in [a_(k-1), inf)
# discrete hazards: discrete probability of h(t) = P(Y=t | Y>=t, X),  t = -1,0,1,2,...,k
# S: survival function: P(Y > t | X)
# all patients are alive from (-inf, 0) by definition, so P(Y=-1) = 0
# h(-1) = 0 ---> do not need to model
# S(-1) = P(Y > -1 | X) = 1 ----> do not need to model
'''
Summary: neural network is hazard probability function, h(t) for t = 0,1,2,...,k-1
corresponding Y = 0,1, ..., k-1. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
'''
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
    #import pdb;pdb.set_trace()

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
    def __call__(self,hazards, Y, c, **kwargs):
        
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(Y)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        #X_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        #import pdb;pdb.set_trace()

        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = Y[j] >= Y[i] 


        #import pdb; pdb.set_trace()
        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))#/ (torch.sum(1-c)+0.0001)
        #loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))))/ torch.sum(c == 0)

        #loss_cox = loss_cox- 0.05 * torch.mean((theta - torch.log(torch.sum(exp_theta*X_mat, dim=1)+1e-7)))
        #import pdb;pdb.set_trace()
        #R_mat = torch.ones(Y.shape[0], Y.shape[0])
        #R_mat[(Y.unsqueeze(1) - Y) > 0] = 0
        #R_mat = torch.FloatTensor(R_mat).to(device)
        #log_loss = torch.exp(hazards) * R_mat
        #log_loss = torch.sum(log_loss, dim=0) / torch.sum(R_mat, dim=0)
        #log_loss = torch.log(log_loss).reshape(-1, 1)
        #neg_log_loss = -torch.sum((hazards-log_loss) * (1-c)) / (torch.sum(1-c)+0.1)
        return loss_cox


class RankingSurvLoss(object):
    def __call__(self,hazards, Y, c, **kwargs):
        batch_size = len(Y)

        if batch_size == 1:
            raise NotImplementedError("Batch size must be at least 2")
        z = hazards
        times = Y
        events = 1 - c
        ##############################
        # determine comparable pairs #
        ##############################
        Z_more_risky = []
        Z_less_risky = []
        for (idx_a, idx_b) in combinations(range(batch_size), 2):
            time_a, event_a = times[idx_a], events[idx_a]
            time_b, event_b = times[idx_b], events[idx_b]

            if time_a < time_b and event_a:
                # a and b are comparable, a is more risky
                Z_more_risky.append(z[idx_a])
                Z_less_risky.append(z[idx_b])
    
            elif time_b < time_a and event_b:
                # a and b are comparable, b is more risky
                Z_more_risky.append(z[idx_b])
                Z_less_risky.append(z[idx_a])

        # if there are no comparable pairs then just return zero
        if len(Z_less_risky) == 0:
            # TODO: perhaps return None?
            return torch.zeros(1, requires_grad=True)

        Z_more_risky = torch.stack(Z_more_risky)
        Z_less_risky = torch.stack(Z_less_risky)

        # compute approximate c indices
        r = Z_more_risky - Z_less_risky
        approx_c_indices = torch.sigmoid(r)

        #approx_c_indices = torch.relu(r)

        # negative mean/sum of c-indices

        return -approx_c_indices.mean()
        #return - approx_c_indices.sum()


class RankingNLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self,hazards, risks, S ,Y, c, **kwargs):
        batch_size = len(Y)

        if batch_size == 1:
            raise NotImplementedError("Batch size must be at least 2")
        z = risks
        times = Y
        events = 1 - c
        ##############################
        # determine comparable pairs #
        ##############################
        Z_more_risky = []
        Z_less_risky = []
        for (idx_a, idx_b) in combinations(range(batch_size), 2):
            time_a, event_a = times[idx_a], events[idx_a]
            time_b, event_b = times[idx_b], events[idx_b]

            if time_a < time_b and event_a:
                # a and b are comparable, a is more risky
                Z_more_risky.append(z[idx_a])
                Z_less_risky.append(z[idx_b])
    
            elif time_b < time_a and event_b:
                # a and b are comparable, b is more risky
                Z_more_risky.append(z[idx_b])
                Z_less_risky.append(z[idx_a])

        # if there are no comparable pairs then just return zero
        if len(Z_less_risky) == 0:
            # TODO: perhaps return None?
            return torch.zeros(1, requires_grad=True)

        Z_more_risky = torch.stack(Z_more_risky)
        Z_less_risky = torch.stack(Z_less_risky)

        # compute approximate c indices
        r = Z_more_risky - Z_less_risky
        approx_c_indices = torch.sigmoid(r)

        #approx_c_indices = torch.relu(r)

        # negative mean/sum of c-indices
        ranking_ls = -approx_c_indices.mean()

        nll_ls = nll_loss(hazards, S, Y, c, alpha=self.alpha)

        return ranking_ls + nll_ls * 0.1
        #return - approx_c_indices.sum()

def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0

    l1_reg += l1_reg_all(model.fc_omic)
    try:
        l1_reg += l1_reg_all(model.mm)
    except:
        pass

    return l1_reg
