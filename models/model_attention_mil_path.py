import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights
from models.model_modules import Attn_Net, Attn_Net_Gated

import numpy as np
"""
A Modified Implementation of Deep Attention MIL
"""
class MIL_Attention_fc_path(nn.Module):
    def __init__(self, gate_path=True, dropout=True, model_size_wsi: str = 'small', n_classes=4):

        super(MIL_Attention_fc_path, self).__init__()
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}

        ### Attention MIL Construction - for Pathology
        size_WSI = self.size_dict_WSI[model_size_wsi]
        fc_WSI = [nn.Linear(size_WSI[0], size_WSI[1]), nn.ReLU()]
        fc_WSI.append(nn.Dropout(0.25))

        if gate_path:
            attention_net_WSI = Attn_Net_Gated(L = size_WSI[1], D = size_WSI[2], dropout = dropout, n_classes = 1)
        else:
            attention_net_WSI = Attn_Net(L = size_WSI[1], D = size_WSI[2], dropout = dropout, n_classes = 1)

        fc_WSI.append(attention_net_WSI)
        self.attention_net_WSI = nn.Sequential(*fc_WSI)

        
        self.classifier = nn.Linear(size_WSI[1], n_classes)

        initialize_weights(self)
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net_WSI = self.attention_net_WSI.to(device)
        self.classifier = self.classifier.to(device)

        
    def forward(self, h, return_features=False, attention_only=False):
        pass

class MIL_Attention_fc_surv_path(MIL_Attention_fc_path):
    def __init__(self,  gate_path=True,  model_size_wsi: str = 'small', dropout=False, n_classes=4):
        super(MIL_Attention_fc_surv_path, self).__init__(gate_path=gate_path, model_size_wsi=model_size_wsi,
            dropout=dropout, n_classes=n_classes)

    def forward(self, **kwargs):
        h = kwargs['path_features']
        A, h = self.attention_net_WSI(h)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 

        logits  = self.classifier(M) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {}

        if 'return_features' in kwargs.keys():
            if kwargs['return_features']:
                return M

        if 'attention_only' in kwargs.keys():
            if kwargs['attention_only']:
                return A_raw

        return hazards, S, Y_hat, A_raw

