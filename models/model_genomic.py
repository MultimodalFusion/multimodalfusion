import math

import torch
from torch import nn
from models.model_modules import SNN_Block
from utils.utils import init_max_weights

############
# Omic Model
############


class MaxNet_base(nn.Module):
    def __init__(self, input_dim: int, model_size_omic: str='small', bag_loss = None, n_classes: int=4):
        super(MaxNet_base, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 256]}
        self.bag_loss = bag_loss
        #self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0],)]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)

        #if transfer:
        #    classifiers = [nn.Linear(hidden[-1], hidden[-1]), nn.ReLU(),nn.Dropout(0.25), 
        #    nn.Linear(hidden[-1], hidden[-1]), nn.ReLU(),nn.Dropout(0.25), 
        #    nn.Linear(hidden[-1], n_classes)]
        #    self.classifier = nn.Sequential(*classifiers)
        #else:
        if 'nll' in self.bag_loss:
            self.classifier = nn.Linear(hidden[-1], n_classes)     
        else:
            self.classifier = nn.Linear(hidden[-1], 1)     


        init_max_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc_omic = self.fc_omic.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        pass

class MaxNet(MaxNet_base):
    def __init__(self, input_dim: int, model_size_omic: str='small', bag_loss = None, n_classes: int=4):
        super().__init__(input_dim,model_size_omic,bag_loss,n_classes)
    def forward(self, **kwargs):
        x = kwargs['genomic_features']

        features = self.fc_omic(x)
        
        if 'return_features' in kwargs:
            if kwargs['return_features']:
                return features

        if 'nll' in self.bag_loss:
            logits = self.classifier(features).unsqueeze(0)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)

                    
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None
        else:
            risk  = self.classifier(features).squeeze()
            return risk, None , None, None




class MaxNet_captum(MaxNet_base):
    def __init__(self, input_dim: int, model_size_omic: str='small', bag_loss = None, n_classes: int=4):
        super().__init__(input_dim,model_size_omic,bag_loss,n_classes)

    def forward(self, x):
        features = self.fc_omic(x)
        if 'nll' in self.bag_loss:
            logits = self.classifier(features).unsqueeze(0)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(S.squeeze(), dim=1)
        else:
            risk  = self.classifier(features)#.squeeze()
        return risk