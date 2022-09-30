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

class collate_MIL_survival(object):
    def __init__(self, radio_modality):
        self.radio_modality =radio_modality
    def __call__(self, batch):
        radio_features = {modality: torch.cat([item[0][modality] for item in batch], dim = 0) for modality in self.radio_modality}
        path_features = torch.cat([item[1] for item in batch], dim = 0)

        genomic_features = torch.cat([item[2].unsqueeze(0) for item in batch], dim = 0)
        label = torch.LongTensor([item[3] for item in batch])
        event_time = np.array([item[4] for item in batch])
        c = torch.FloatTensor([item[5] for item in batch])
        return [radio_features, path_features, genomic_features, label, event_time,c]

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def collate_radio_features(batch):
    img = [item[0][None,...] for item in batch if item is not None]

    if img:
        img = torch.cat(img, dim = 0)
        slice_id = np.hstack([item[1] for item in batch if item is not None])
        return [img, slice_id]
    else:
        return[None] * 2

def collate_radio_features_mask(batch):
    img = [item[0][None,...] for item in batch if item is not None]
    try:
        mask = [item[2][None,...] for item in batch if item is not None]
    except:
        mask = None
    if img:
        img = torch.cat(img, dim = 0)
        
        slice_id = np.hstack([item[1] for item in batch if item is not None])
        if mask is not None:
            mask = torch.cat(mask, dim = 0)
        return [img, slice_id,mask]

    else:
        return[None] * 3


def collate_MIL_survival_pretrained(batch):
    radio_features = torch.cat([item[0] for item in batch], dim = 0)
    path_features = torch.cat([item[1] for item in batch], dim = 0)
    genomic_features = torch.cat([item[2] for item in batch], dim = 0)
    label = torch.LongTensor([item[3] for item in batch])
    event_time = np.array([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])

    other_var  = torch.cat([item[6] for item in batch], dim = 0)
    #masks = {'radio':[],'path':[],'genomic':[]}
    #for item in batch:
    #    masks['radio'].append(item[6]['radio'])
    #    masks['path'].append(item[6]['path'])
    #    masks['genomic'].append(item[6]['genomic'])
    return [radio_features, path_features, genomic_features, label, event_time,c, other_var]


def get_simple_loader(dataset, batch_size=1):
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_features, **kwargs)
    return loader 

def get_split_loader(split_dataset, training = False, weighted = False,  batch_size=1, radio_modality = ['T1','T2','T1Gd','FLAIR']):
    """
        return either the validation loader or training loader 
    """
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    collate_function = collate_MIL_survival(radio_modality)
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_function,shuffle=False, **kwargs)    
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_function, **kwargs)
            #loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_function, **kwargs)

    else:
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_function,shuffle=False, **kwargs)
    return loader

def get_pretrained_split_loader(split_dataset, training = False,  weighted = False,  batch_size=1):
    """
        return either the validation loader or training loader 
    """
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL_survival_pretrained,shuffle=False, **kwargs)    
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL_survival_pretrained, **kwargs)

    else:
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL_survival_pretrained,shuffle=False, **kwargs)

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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_ckpt = None

    def __call__(self, epoch, val_loss, model, ckpt_name = None):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if ckpt_name is not None:
            torch.save(model.state_dict(), ckpt_name)
        self.best_ckpt = model.state_dict()
        self.val_loss_min = val_loss


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
            #nn.init.kaiming_normal_(m.weight, mode = 'fan_out',non_linearity = 'relu')
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

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
