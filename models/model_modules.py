import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f
        self.bn1 = nn.BatchNorm1d(size)
        self.bn2 = nn.BatchNorm1d(size)
        self.dropout1 = nn.Dropout(0.7)

    def forward(self, x):
        x = self.bn1(x)
        x = self.dropout1(x)
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        x= self.bn2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self,size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(size,size)
        self.bn1 = nn.BatchNorm1d(size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)

        #if self.downsample:
        #    residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Residual(nn.Module):
    def __init__(self,size, n_layer):
        super(Residual, self).__init__()
        self.n_layer = n_layer
        self.blocks = nn.ModuleList([ResidualBlock(size) for i in range(n_layer)])
    def forward(self,x):
        for i in range(self.n_layer):
            x = self.blocks[i](x)
        return x




def SNN_Block(dim1, dim2, dropout=0.25):
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.SELU(),
            nn.AlphaDropout(p=dropout, inplace=False))
    
class Attn_Net(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class XlinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling
    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=1, use_bilinear=0, gate=1, dim=256, scale_dim=16, num_modalities=4,
                 mmhid1=256, mmhid2=256, dropout_rate=0.25):
        super(XlinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate = gate
        self.num_modalities = num_modalities

        dim_og, dim = dim, dim//scale_dim
        skip_dim = dim_og*self.num_modalities if skip else 0
                
        self.reduce = []
        for i in range(self.num_modalities):
            linear_h = nn.Sequential(nn.Linear(dim_og, dim), nn.ReLU())
            linear_z = nn.Bilinear(dim_og, dim_og, dim) if use_bilinear else nn.Sequential(nn.Linear(dim_og*self.num_modalities, dim))
            linear_o = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(p=dropout_rate))
            
            if self.gate:
                self.reduce.append(nn.ModuleList([linear_h, linear_z, linear_o]))
            else:
                self.reduce.append(nn.ModuleList([linear_h, linear_o]))
                
        self.reduce = nn.ModuleList(self.reduce)

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim+1)**num_modalities, mmhid1), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid1+skip_dim, mmhid2), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, v_list: list):
        v_cat = torch.cat(v_list, axis=1)
        o_list = []
        
        for i, v in enumerate(v_list):
            h = self.reduce[i][0](v)
            z = self.reduce[i][1](v_cat)
            o = self.reduce[i][2](nn.Sigmoid()(z)*h)
            o = torch.cat((o, torch.cuda.FloatTensor(o.shape[0], 1).fill_(1)), 1)
            o_list.append(o)
        
        o_fusion = o_list[0]
        for o in o_list[1:]:
            o_fusion = torch.bmm(o_fusion.unsqueeze(2), o.unsqueeze(1)).flatten(start_dim=1)

        ### Fusion
        out = self.post_fusion_dropout(o_fusion)
        out = self.encoder1(out)
        if self.skip: 
            for v in v_list:
                out = torch.cat((out, v), axis=1)
        out = self.encoder2(out)
        return out
