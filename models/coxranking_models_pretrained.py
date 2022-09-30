
from os.path import join
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils_pretrained import initialize_weights, dfs_freeze
from models.model_modules import Highway, ResidualBlock, Residual, XlinearFusion
import numpy as np


class unimonal_pretrained(nn.Module):
    def __init__(self,dropout=True, n_classes=4, mode = 'radio', train_type = None, bag_loss = None, n_layers =1 ):
        super(unimonal_pretrained, self).__init__()
        self.n_classes = n_classes
        self.train_type = train_type
        self.bag_loss = bag_loss
        self.mode = mode
        self.n_layers = n_layers

        if self.train_type == 'fcnn':
            self.classifier = nn.Sequential(*[nn.Linear(256, 128), nn.BatchNorm1d(128),nn.ReLU(), nn.Dropout(0.7),nn.Linear(128, 1)])
        elif self.train_type == 'highway':
            self.highway = Highway(256,n_layers,f = F.relu)
            self.classifier = nn.Linear(256, 1)
        elif self.train_type == 'residual':
            self.residual =Residual(256,n_layers)
            self.classifier = nn.Linear(256, 1)
        
        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = self.classifier.to(device)
        if self.train_type == 'highway':
            self.highway = self.highway.to(device)
        elif self.train_type == 'residual':
            self.residual = self.residual.to(device)

    def forward(self, **kwargs):
        if self.mode == 'path':
            h = kwargs['h_path']
        elif self.mode == 'radio':
            h = kwargs['h_radio']
        if self.mode == 'omic':
            h = kwargs['h_omic']

        if self.train_type == 'fcnn':
            risk  = self.classifier(h).squeeze()
        elif self.train_type == 'highway':
            h = self.highway(h)
            risk  = self.classifier(h).squeeze()
        elif self.train_type == 'residual':
            h = self.residual(h)
            risk  = self.classifier(h).squeeze()
        return risk, None , None



class multimodal_pretrained(nn.Module):
    def __init__(self,  dropout=True, n_classes=4, mode = 'radio_path_omic', 
        train_type = None, bag_loss = None, n_layers = 1):
        super(multimodal_pretrained, self).__init__()
        self.n_classes = n_classes
        self.mode = mode
        self.train_type = train_type
        self.bag_loss = bag_loss
        self.n_layers = n_layers

        num_modalities = 0
        if 'radio' in mode:
            num_modalities+= 1
        if 'path' in mode:
            num_modalities+= 1
        if 'omic' in mode:
            num_modalities+= 1

        if train_type == 'late-fcnn':
            self.layer_WSI = nn.Sequential(*[nn.Linear(256, 128), nn.BatchNorm1d(128),nn.ReLU(), nn.Dropout(0.7),nn.Linear(128, 1)])
            self.layer_MRI = nn.Sequential(*[nn.Linear(256, 128), nn.BatchNorm1d(128),nn.ReLU(), nn.Dropout(0.7),nn.Linear(128, 1)])
            self.layer_omic = nn.Sequential(*[nn.Linear(256, 128), nn.BatchNorm1d(128),nn.ReLU(), nn.Dropout(0.7),nn.Linear(128, 1)])      
            self.classifier = nn.Sequential(*[nn.Linear(num_modalities, 1 )]) 
        elif train_type == 'early-fcnn':
            self.classifier = nn.Sequential(*[nn.Linear(num_modalities * 256 ,128), nn.BatchNorm1d(128),nn.ReLU(), nn.Dropout(0.7),nn.Linear(128, 1)]) 
        elif train_type == 'early-highway':
            self.highway = Highway(num_modalities * 256 ,n_layers,F.relu)
            self.classifier = nn.Linear(num_modalities * 256 , 1 )
        elif train_type == 'late-highway':
            self.highway_radio = Highway(256,n_layers,F.relu)
            self.highway_path = Highway(256,n_layers,F.relu)
            self.highway_omic = Highway(256,n_layers,F.relu)
            self.classifier = nn.Linear(num_modalities * 256, 1 )
        elif train_type == 'kronecker':
            self.xfusion = XlinearFusion(num_modalities=num_modalities, dropout_rate = 0.7)
            self.classifier = nn.Linear(256, 1)
        #elif train_type == 'early-residual':
        #    classifier_size = 0
        #    if 'radio' in mode:
        #        classifier_size+= 256
        #    if 'path' in mode:
        #        classifier_size+= 256
        #    if 'omic' in mode:
        #        classifier_size+= 256
        #    self.residual = Residual(classifier_size,n_layers)
       #     self.classifier = nn.Linear(classifier_size, 1 )

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.train_type == 'late-fcnn':
            self.layer_WSI = self.layer_WSI.to(device)
            self.layer_MRI = self.layer_MRI.to(device)
            self.layer_omic = self.layer_omic.to(device)
        elif self.train_type == 'early-fcnn':
            self.classifier = self.classifier.to(device)
        elif self.train_type == 'early-highway':
            self.highway = self.highway.to(device)
        elif self.train_type == 'late-highway':
            self.highway_radio = self.highway_radio.to(device)       
            self.highway_path = self.highway_path.to(device)        
            self.highway_omic = self.highway_omic.to(device)   
        elif self.train_type == 'kronecker':
            self.xfusion = self.xfusion.to(device)         
 
        #elif self.train_type == 'early-residual':
        #    self.residual = self.residual.to(device)
        
        self.classifier = self.classifier.to(device)


    def forward(self, h_radio, h_path, h_omic):
        if 'late' in self.train_type:
            if self.train_type == 'late-fcnn':
                radio_layer = self.layer_MRI(h_radio).unsqueeze(0)
                path_layer = self.layer_WSI(h_path).unsqueeze(0)
                genomics_layer = self.layer_omic(h_omic).unsqueeze(0)
            elif self.train_type == 'late-highway':
                radio_layer = self.highway_radio(h_radio).unsqueeze(0)
                path_layer = self.highway_path(h_path).unsqueeze(0)
                genomics_layer = self.highway_omic(h_omic).unsqueeze(0)    

            if 'radio' in self.mode and 'path' in self.mode and 'omic' not in self.mode:
                MM = torch.cat([radio_layer,path_layer],axis = 2)
            if 'radio' in self.mode and 'omic' in self.mode and 'path' not in self.mode:
                MM = torch.cat([radio_layer,genomics_layer],axis = 2)
            if 'omic' in self.mode and 'path' in self.mode and 'radio' not in self.mode:
                MM = torch.cat([genomics_layer,path_layer],axis = 2)
            if 'radio' in self.mode and 'path' in self.mode and 'omic' in self.mode:
                MM = torch.cat([radio_layer,path_layer,genomics_layer],axis = 2)
            risk = self.classifier(MM).squeeze()

        elif 'early' in self.train_type :
            if 'radio' in self.mode and 'path' in self.mode and 'omic' not in self.mode:
                MM = torch.cat([h_radio,h_path],axis = 1)
            if 'radio' in self.mode and 'omic' in self.mode and 'path' not in self.mode:
                MM = torch.cat([h_radio,h_omic],axis = 1)
            if 'omic' in self.mode and 'path' in self.mode and 'radio' not in self.mode:
                MM = torch.cat([h_omic,h_path],axis = 1)
            if 'radio' in self.mode and 'path' in self.mode and 'omic' in self.mode:
                MM = torch.cat([h_radio,h_path,h_omic],axis = 1)

            if self.train_type == 'early-fcnn':
                pass
            elif self.train_type == 'early-highway':
                MM = self.highway(MM)
            risk = self.classifier(MM)

        elif self.train_type == 'kronecker':
            if 'radio' in self.mode and 'path' in self.mode and 'omic' not in self.mode:
                MM = self.xfusion(v_list = [h_radio,h_path])
            if 'radio' in self.mode and 'omic' in self.mode and 'path' not in self.mode:
                MM = self.xfusion(v_list = [h_radio,h_omic])
            if 'omic' in self.mode and 'path' in self.mode and 'radio' not in self.mode:
                MM = self.xfusion(v_list = [h_omic,h_path])
            if 'radio' in self.mode and 'path' in self.mode and 'omic' in self.mode:
                MM = self.xfusion(v_list = [h_radio,h_path,h_omic])
            risk = self.classifier(MM)


        return risk, None, None

        """
        elif self.train_type == 'multimodal-mean':
            all_layers = []
            if 'radio' in self.mode:
                radio_layer = self.layer_MRI(h_radio).unsqueeze(0)
                all_layers.append(radio_layer)
            if 'path' in self.mode:
                path_layer = self.layer_WSI(h_path).unsqueeze(0)
                all_layers.append(path_layer)
            if 'omic' in self.mode:
                genomics_layer = self.layer_omic(h_omic).unsqueeze(0)
                all_layers.append(genomics_layer)

            MM = torch.cat(all_layers,axis = 0)
            risk = torch.mean(MM, axis = 0)
        """

    def captum_radio_path(self, h_radio,h_path):
        if 'late' in self.train_type:
            if self.train_type == 'late-fcnn':
                radio_layer = self.layer_MRI(h_radio).unsqueeze(0)
                path_layer = self.layer_WSI(h_path).unsqueeze(0)
            elif self.train_type == 'late-highway':
                radio_layer = self.highway_radio(h_radio).unsqueeze(0)
                path_layer = self.highway_path(h_path).unsqueeze(0)

            MM = torch.cat([radio_layer,path_layer],axis = 2)
            risk = self.classifier(MM).squeeze()

        elif 'early' in self.train_type :
            MM = torch.cat([h_radio,h_path],axis = 1)
            if self.train_type == 'early-fcnn':
                pass
            elif self.train_type == 'early-highway':
                MM = self.highway(MM)
            risk = self.classifier(MM)

        elif self.train_type == 'kronecker':
            MM = self.xfusion(v_list = [h_radio,h_path])
            risk = self.classifier(MM)
        return risk

    def captum_path_omic(self, h_omic,h_path):
        if 'late' in self.train_type:
            if self.train_type == 'late-fcnn':
                path_layer = self.layer_WSI(h_path).unsqueeze(0)
                genomics_layer = self.layer_omic(h_omic).unsqueeze(0)
            elif self.train_type == 'late-highway':
                path_layer = self.highway_path(h_path).unsqueeze(0)
                genomics_layer = self.highway_omic(h_omic).unsqueeze(0)    

            MM = torch.cat([genomics_layer,path_layer],axis = 2)
            risk = self.classifier(MM).squeeze()

        elif 'early' in self.train_type :
            MM = torch.cat([h_omic,h_path],axis = 1)
            if self.train_type == 'early-fcnn':
                pass
            elif self.train_type == 'early-highway':
                MM = self.highway(MM)
            risk = self.classifier(MM)

        elif self.train_type == 'kronecker':
            MM = self.xfusion(v_list = [h_omic,h_path])
            risk = self.classifier(MM)
        return risk

    def captum_radio_omic(self, h_radio,h_omic):
        if 'late' in self.train_type:
            if self.train_type == 'late-fcnn':
                radio_layer = self.layer_MRI(h_radio).unsqueeze(0)
                genomics_layer = self.layer_omic(h_omic).unsqueeze(0)
            elif self.train_type == 'late-highway':
                radio_layer = self.highway_radio(h_radio).unsqueeze(0)
                genomics_layer = self.highway_omic(h_omic).unsqueeze(0)    

            MM = torch.cat([radio_layer,genomics_layer],axis = 2)
            risk = self.classifier(MM).squeeze()

        elif 'early' in self.train_type :
            MM = torch.cat([h_radio,h_omic],axis = 1)
            if self.train_type == 'early-fcnn':
                pass
            elif self.train_type == 'early-highway':
                MM = self.highway(MM)
            risk = self.classifier(MM)

        elif self.train_type == 'kronecker':
            MM = self.xfusion(v_list = [h_radio,h_omic])
            risk = self.classifier(MM)
        return risk

    def captum(self, h_radio, h_path, h_omic):
        if 'late' in self.train_type:
            if self.train_type == 'late-fcnn':
                radio_layer = self.layer_MRI(h_radio).unsqueeze(0)
                path_layer = self.layer_WSI(h_path).unsqueeze(0)
                genomics_layer = self.layer_omic(h_omic).unsqueeze(0)
            elif self.train_type == 'late-highway':
                radio_layer = self.highway_radio(h_radio).unsqueeze(0)
                path_layer = self.highway_path(h_path).unsqueeze(0)
                genomics_layer = self.highway_omic(h_omic).unsqueeze(0)    

            MM = torch.cat([radio_layer,path_layer,genomics_layer],axis = 2)
            risk = self.classifier(MM).squeeze()

        elif 'early' in self.train_type :
            MM = torch.cat([h_radio,h_path,h_omic],axis = 1)

            if self.train_type == 'early-fcnn':
                pass
            elif self.train_type == 'early-highway':
                MM = self.highway(MM)
            risk = self.classifier(MM)

        elif self.train_type == 'kronecker':
            MM = self.xfusion(v_list = [h_radio,h_path,h_omic])
            risk = self.classifier(MM)

        return risk
    """

    def captum_multimodal_mean(self, h_radio, h_path, h_omic):
            all_layers = []
            if 'radio' in self.mode:
                radio_layer = self.layer_MRI(h_radio).unsqueeze(0)
                all_layers.append(radio_layer)
            if 'path' in self.mode:
                path_layer = self.layer_WSI(h_path).unsqueeze(0)
                all_layers.append(path_layer)
            if 'omic' in self.mode:
                genomics_layer = self.layer_omic(h_omic).unsqueeze(0)
                all_layers.append(genomics_layer)

            MM = torch.cat(all_layers,axis = 0)
            if self.bag_loss == 'cox_surv' or self.bag_loss == 'ranking_surv':
                risk = torch.mean(MM, axis = 0)

            elif 'nll_surv' in self.bag_loss: 
                logits  = torch.mean(MM, axis = 0)
                Y_hat = torch.topk(logits, 1, dim = 1)[1]
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(S, dim=1)
            return risk, None , None
    """
