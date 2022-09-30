from os.path import join
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights, dfs_freeze
from models.model_modules import SNN_Block,Attn_Net, Attn_Net_Gated, XlinearFusion
import numpy as np

"""
A Modified Implementation of Deep Attention MIL
"""



class MM_MIL_Attention_fc(nn.Module):
    def __init__(self, input_dim: int=80, radio_fusion='concat', fusion = 'tensor',
        gate=True, gate_path=True, gate_radio = True, dropout=True,
        model_size_radio: str='small', model_size_wsi: str = 'small', model_size_omic: str='small', n_classes=4,
        modalities = ['T1','T2','T1Gd','FLAIR'], mode = 'radio_path_omic'
        ):
        super(MM_MIL_Attention_fc, self).__init__()
        self.radio_fusion = radio_fusion
        self.fusion = fusion
        self.n_classes = n_classes
        self.size_dict_radio = {"small": [1024, 256, 256], "big": [1024, 256, 384]}
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 256, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 256]}
        self.modalities = modalities
        self.mode = mode

        ### Constructing Genomic SNN
        size_omic = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=input_dim, dim2=size_omic[0])]
        for i, _ in enumerate(size_omic[1:]):
            fc_omic.append(SNN_Block(dim1=size_omic[i], dim2=size_omic[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)


        ### Attention MIL Construction - for Radiology
        size_radio = self.size_dict_radio[model_size_radio]
        fc_radio = [nn.Linear(size_radio[0], size_radio[1]), nn.ReLU()]
        fc_radio.append(nn.Dropout(0.25))


        if gate_radio:
            attention_net_radio = Attn_Net_Gated(L = size_radio[1], D = size_radio[2], dropout = dropout, n_classes = 1)
        else:
            attention_net_radio = Attn_Net(L = size_radio[1], D = size_radio[2], dropout = dropout, n_classes = 1)

        fc_radio.append(attention_net_radio)
        self.attention_net_radio = nn.Sequential(*fc_radio)

        if self.radio_fusion == 'tensor':
            self.radio_xfusion = XlinearFusion(dim=1024, scale_dim=64,mmhid1 = 1024,mmhid2 = 1024, skip  = 0)

        elif self.radio_fusion == 'concat':
            self.reduce_dim = nn.Linear(size_radio[0]*len(self.modalities), size_radio[0])

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

        ### Multimodal Fusion
        classifier_size = 0
        n_modalities = 0
        if 'radio' in mode:
            classifier_size+= size_radio[1]
            n_modalities+=1
        if 'path' in mode:
            classifier_size+= size_path[1]
            n_modalities+=1
        if 'omic' in mode:
            classifier_size+= size_omic[1]
            n_modalities+=1

        if self.fusion == 'tensor':
            self.mm = XlinearFusion(dim = 256, scale_dim = 16 , mmhid1 = 512, mmhid2 =512, num_modalities = n_modalities, gate=gate, skip = 1)
            self.classifier = nn.Sequential(*[nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, n_classes)])
        elif self.fusion == 'concat':
            #self.classifier = nn.Sequential(*[nn.Linear(classifier_size, classifier_size//2), nn.ReLU(),nn.Dropout(0.25),
            #                                nn.Linear(classifier_size//2, n_classes)])
            self.classifier = nn.Linear(classifier_size, n_classes)


        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc_omic = self.fc_omic.to(device)
        self.attention_net_radio = self.attention_net_radio.to(device)
        self.attention_net_WSI = self.attention_net_WSI.to(device)
        self.classifier = self.classifier.to(device)
        if self.fusion == 'tensor':
            self.mm = self.mm.to(device)
        if self.radio_fusion == 'tensor':
            self.radio_xfusion = self.radio_xfusion.to(device)
        elif self.radio_fusion == 'concat':
            self.reduce_dim = self.reduce_dim.to(device)

        
    def forward(self, h, return_features=False, attention_only=False):
        pass

class MM_MIL_Attention_fc_surv(MM_MIL_Attention_fc):
    def __init__(self, input_dim: int=80, radio_fusion: str='concat',  fusion: str='tensor',
        gate=True, gate_path=True, gate_omic=True, gate_radio = True,
        model_size_radio="small",  model_size_wsi: str = 'small',model_size_omic='small', 
        dropout=False, n_classes=4,  mode = 'radio_path_omic'):
        super(MM_MIL_Attention_fc_surv, self).__init__(
            input_dim = input_dim, radio_fusion=radio_fusion, fusion = fusion,
            gate=gate, gate_path=gate_path, gate_omic=gate_omic,gate_radio = gate_radio, model_size_radio='small',
            model_size_wsi=model_size_wsi, model_size_omic=model_size_omic, dropout=dropout, n_classes=n_classes,
             mode = mode)

    def forward(self, **kwargs):
        A_raw = {}
        if 'radio' in self.mode:
            #Radiology
            h_radio = []
            for m in self.modalities:
                h_radio.append(kwargs[m])

            if len(self.modalities) > 1:
                if self.radio_fusion=='concat':
                    h_radio = torch.cat(h_radio, axis=1)
                    h_radio = self.reduce_dim(h_radio)
                elif self.radio_fusion=='tensor':
                    h_radio = self.xfusion(v_list=[m_h[0].unsqueeze(dim=0) for m_h in h_radio])
            else:
                h_radio = h_radio.squeeze()

            A_radio, h_radio = self.attention_net_radio(h_radio)  
            A_radio = torch.transpose(A_radio, 1, 0)

            A_raw['radiology'] = A_radio
            A_radio = F.softmax(A_radio, dim=1) 
            M_radio = torch.mm(A_radio, h_radio)

        if 'path' in self.mode:
            #pathology
            h_path = kwargs['path_features']
            A_path, h_path = self.attention_net_WSI(h_path)  
            A_path = torch.transpose(A_path, 1, 0)

            A_raw['pathology'] = A_path
            A_path = F.softmax(A_path, dim=1) 
            M_path = torch.mm(A_path, h_path)

        if 'omic' in self.mode:
            #Omics
            X = kwargs['genomic_features']
            O = self.fc_omic(X.unsqueeze(0))


        if 'radio' in self.mode and 'path' in self.mode and 'omic' not in self.mode:
            if self.fusion == 'tensor':
                MM = self.mm(v_list=[M_radio,M_path])
            elif self.fusion == 'concat':
                MM = torch.cat([M_radio,M_path],axis = 1)
        elif 'radio' in self.mode and 'omic' in self.mode and 'path' not in self.mode:
            if self.fusion == 'tensor':
                MM = self.mm(v_list=[M_radio,O])
            elif self.fusion == 'concat':
                MM = torch.cat([M_radio,O],axis = 1)
        elif 'omic' in self.mode and 'path' in self.mode and 'radio' not in self.mode:
            if self.fusion == 'tensor':
                MM = self.mm(v_list=[O,M_path])
            elif self.fusion == 'concat':
                MM = torch.cat([O,M_path],axis = 1)
        elif 'radio' in self.mode and 'path' in self.mode and 'omic' in self.mode:
            if self.fusion == 'tensor':
                MM = self.mm(v_list=[M_radio,M_path,O])
            elif self.fusion == 'concat':
                MM = torch.cat([M_radio,M_path,O],axis = 1)
                #!!! add a layer plz

        logits  = self.classifier(MM)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        # hazards = F.softmax(logits, dim=1)
        S = torch.cumprod(1 - hazards, dim=1)

        if 'return_features' in kwargs.keys():
            if kwargs['return_features']:
                results_dict.update({'features': M})
        
        return hazards, S, Y_hat, A_raw

    def captum_radio_omic(self, T1,T2,T1Gd,FLAIR, h_omic):
        A_raw = {}

        if 'radio' in self.mode:
            #Radiology
            h_radio = []
            for m in self.modalities:
                h_radio.append(locals()[m])

            if self.radio_fusion=='concat':
                #print('concat correctly implemented')
                h_radio = torch.cat(h_radio, axis=2)
                h_radio = self.reduce_dim(h_radio)
            elif self.radio_fusion=='tensor':
                #print('tensor correctly implemented')
                h_radio = self.radio_xfusion(v_list=[m_h[0].unsqueeze(dim=0) for m_h in h_radio])

            A_radio, h_radio = self.attention_net_radio(h_radio)  
            A_radio = torch.transpose(A_radio, 2, 1)
            A_raw['radiology'] = A_radio
            A_radio = F.softmax(A_radio, dim=1) 
            M_radio = torch.matmul(A_radio, h_radio).squeeze()
        else:
            raise NotImplementedError('use another captum function')

        if 'omic' in self.mode:
            #Omics
            O = self.fc_omic(h_omic)
        else:
            raise NotImplementedError('use another captum function')

        if self.fusion == 'tensor':
            MM = self.mm(v_list=[M_radio,O])
        elif self.fusion == 'concat':
            MM = torch.cat([M_radio,O],axis = 1)

        logits  = self.classifier(MM)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S.squeeze(), dim=1)

        return risk

    def captum(self, T1,T2,T1Gd,FLAIR, h_path, h_omic):
        A_raw = {}

        if 'radio' in self.mode:
            #Radiology
            h_radio = []
            for m in self.modalities:
                h_radio.append(locals()[m])

            #

            if self.radio_fusion=='concat':
                #print('concat correctly implemented')
                h_radio = torch.cat(h_radio, axis=2)
                h_radio = self.reduce_dim(h_radio)
            elif self.radio_fusion=='tensor':
                #print('tensor correctly implemented')
                h_radio = self.radio_xfusion(v_list=[m_h[0].unsqueeze(dim=0) for m_h in h_radio])

            A_radio, h_radio = self.attention_net_radio(h_radio)  
            A_radio = torch.transpose(A_radio, 2, 1)
            A_raw['radiology'] = A_radio
            A_radio = F.softmax(A_radio, dim=1) 
            M_radio = torch.matmul(A_radio, h_radio).squeeze()
        else:
            raise NotImplementedError('use another captum function')

        if 'path' in self.mode:
            #pathology
            A_path, h_path = self.attention_net_WSI(h_path)  
            A_path = torch.transpose(A_path, 2,1)

            A_raw['pathology'] = A_path
            A_path = F.softmax(A_path, dim=1) 
            M_path = torch.matmul(A_path, h_path).squeeze()
        else:
            raise NotImplementedError('use another captum function')

        if 'omic' in self.mode:
            #Omics
            O = self.fc_omic(h_omic)
        else:
            raise NotImplementedError('use another captum function')

        if self.fusion == 'tensor':
            MM = self.mm(v_list=[M_radio,M_path,O])
        elif self.fusion == 'concat':
            MM = torch.cat([M_radio,M_path,O],axis = 1)

        logits  = self.classifier(MM)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S.squeeze(), dim=1)

        return risk

    def captum_radio_path(self, T1,T2,T1Gd,FLAIR, h_path):
        A_raw = {}

        if 'radio' in self.mode:
            #Radiology
            h_radio = []
            for m in self.modalities:
                h_radio.append(locals()[m])

            #

            if self.radio_fusion=='concat':
                #print('concat correctly implemented')
                h_radio = torch.cat(h_radio, axis=2)
                h_radio = self.reduce_dim(h_radio)
            elif self.radio_fusion=='tensor':
                #print('tensor correctly implemented')
                h_radio = self.radio_xfusion(v_list=[m_h[0].unsqueeze(dim=0) for m_h in h_radio])

            A_radio, h_radio = self.attention_net_radio(h_radio)  
            A_radio = torch.transpose(A_radio, 2, 1)
            A_raw['radiology'] = A_radio
            A_radio = F.softmax(A_radio, dim=1) 
            M_radio = torch.matmul(A_radio, h_radio).squeeze()
        else:
            raise NotImplementedError('use another captum function')

        if 'path' in self.mode:
            #pathology
            A_path, h_path = self.attention_net_WSI(h_path)  
            A_path = torch.transpose(A_path, 2,1)

            A_raw['pathology'] = A_path
            A_path = F.softmax(A_path, dim=1) 
            M_path = torch.matmul(A_path, h_path).squeeze()
        else:
            raise NotImplementedError('use another captum function')

        if 'omic' in self.mode:
            raise NotImplementedError('use another captum function')

        if self.fusion == 'tensor':
            MM = self.mm(v_list=[M_radio,M_path])
        elif self.fusion == 'concat':
            MM = torch.cat([M_radio,M_path],axis = 1)

        logits  = self.classifier(MM)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S.squeeze(), dim=1)

        return risk

    def captum_path_omic(self, h_omic, h_path):
        A_raw = {}

        if 'path' in self.mode:
            #pathology
            A_path, h_path = self.attention_net_WSI(h_path)  
            A_path = torch.transpose(A_path, 2,1)

            A_raw['pathology'] = A_path
            A_path = F.softmax(A_path, dim=1) 
            M_path = torch.matmul(A_path, h_path).squeeze()
        else:
            raise NotImplementedError('use another captum function')

        if 'omic' in self.mode:
            #Omics
            O = self.fc_omic(h_omic)
        else:
            raise NotImplementedError('use another captum function')

        if 'omic' in self.mode and 'path' in self.mode and 'radio' not in self.mode:
            if self.fusion == 'tensor':
                MM = self.mm(v_list=[O,M_path])
            elif self.fusion == 'concat':
                if len(M_path.shape) == 1:
                    M_path = torch.reshape(M_path,(1,-1))

                MM = torch.cat([O,M_path],axis = 1)
        else:
            raise NotImplementedError('use another captum function')

        logits  = self.classifier(MM)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        if len(S.shape) == 3:
            risk = -torch.sum(S.squeeze(), dim=1)
        elif len(S.shape) == 2:
            risk = -torch.sum(S,dim=1)
        return risk
