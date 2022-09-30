from os.path import join
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights, dfs_freeze
from models.model_modules import Attn_Net, Attn_Net_Gated, XlinearFusion

import numpy as np

class MIL_Attention_fc_radio(nn.Module):
    def __init__(self, radio_fusion= 'concat', gate_radio=True, dropout=True,
        model_size_radio: str='small',n_classes=4, modalities = ['T1','T2','T1Gd','FLAIR']):
        super(MIL_Attention_fc_radio, self).__init__()

        self.radio_fusion = radio_fusion
        self.n_classes = n_classes
        self.size_dict_radio = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.modalities = modalities
        #self.radio_mil_type = radio_mil_type
        #print(radio_mil_type)
        
        size_radio = self.size_dict_radio[model_size_radio]

        if len(self.modalities) > 1:
            if self.radio_fusion == 'tensor':
                self.radio_xfusion = XlinearFusion(dim=1024, scale_dim=64,mmhid1 = 1024,mmhid2 = 1024, skip  = 0)

            elif self.radio_fusion == 'concat':
                self.reduce_dim = nn.Linear(size_radio[0]*len(self.modalities), size_radio[0])

        fc_radio = [nn.Linear(size_radio[0], size_radio[1]), nn.ReLU()]
        fc_radio.append(nn.Dropout(0.25))


        if gate_radio:
            attention_net_radio = Attn_Net_Gated(L = size_radio[1], D = size_radio[2], dropout = dropout, n_classes = 1)
        else:
            attention_net_radio = Attn_Net(L = size_radio[1], D = size_radio[2], dropout = dropout, n_classes = 1)

        fc_radio.append(attention_net_radio)
        self.attention_net_radio = nn.Sequential(*fc_radio)



        self.classifier = nn.Linear(size_radio[1], n_classes)   


        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if len(self.modalities) > 1:
            if self.radio_fusion == 'tensor':
               self.xfusion = self.xfusion.to(device)
            elif self.radio_fusion == 'concat':
                self.reduce_dim = self.reduce_dim.to(device)
        self.attention_net_radio = self.attention_net_radio.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, h, return_features=False, attention_only=False):
        pass

class MIL_Attention_fc_surv_radio(MIL_Attention_fc_radio):
    def __init__(self, radio_fusion= 'concat', gate_radio=True, dropout=True,
        model_size_radio: str='small',n_classes=4, modalities = ['T1','T2','T1Gd','FLAIR']):
        super(MIL_Attention_fc_surv_radio, self).__init__(
            radio_fusion=radio_fusion,gate_radio=gate_radio, model_size_radio='small', 
            dropout=dropout, n_classes=n_classes, modalities = modalities)

    def forward(self, **kwargs):
        A_raw = None
        h = []
        for m in self.modalities:
            h.append(kwargs[m])

        if len(self.modalities) > 1:
            if self.radio_fusion=='concat':
                h = torch.cat(h, axis=1)
                h = self.reduce_dim(h)
            elif self.radio_fusion=='tensor':
                h = self.xfusion(v_list=[m_h[0].unsqueeze(dim=0) for m_h in h])
        else:
            h = h[0]


        A, h = self.attention_net_radio(h)  
        A = torch.transpose(A, 1, 0)
        if 'attention_only' in kwargs.keys():
            if kwargs['attention_only']:
                return A
        #import pdb;pdb.set_trace()
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)
        #top_A = torch.argmax(A)
        #M = h[top_A:top_A+1]

        logits  = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)


        if 'return_features' in kwargs.keys():
            if kwargs['return_features']:
                return M

        if 'return_attention' in kwargs.keys():
            if kwargs['return_attention']:
                return A_raw
        
        return hazards, S, Y_hat, A_raw
