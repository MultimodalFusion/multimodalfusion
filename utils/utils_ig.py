import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

from utils.file_utils import save_hdf5
from utils.utils import collate_radio_features_mask
from models.resnet_custom import resnet50_baseline
from datasets.dataset_raw import PreprocessDatasetMask
from models.model_modules import Attn_Net, Attn_Net_Gated, XlinearFusion


from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cmapy
import cv2
import nibabel as nib

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_slide_features(args,subject_m, subject_id, top_k_slide):
    print('Get features...')
    transform_f = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    ])  

    transform_orig = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ])  

    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    all_modalities_features = []
    all_modalities_orig_features = []
    all_masks = []
    for m,f in subject_m.items():
        file_path = os.path.join(args.radio_dir,subject_id,f)
        #import pdb;pdb.set_trace()
        pd = PreprocessDatasetMask(file_path =file_path, 
                    transform = True, transform_f = transform_f, plane = 'axial', 
                    cancer_type = args.cancer_type,segment = args.segment)

        loader = DataLoader(dataset = pd, batch_size = 1000, 
            shuffle = False, collate_fn=collate_radio_features_mask, **kwargs)

        all_radio_features_m = []
        all_masks_m = []
        for count,(batch,slice_ids,mask ) in enumerate(loader):
            all_radio_features_m.append(batch.numpy())
            if mask is not None:
                all_masks_m.append(mask.numpy())
            radio_features = batch.to(device)
            slice_ids = slice_ids


        true_top_k_slide = [np.where(slice_ids == t_k)[0].item() for t_k in top_k_slide]
        all_modalities_features.append(radio_features[true_top_k_slide])
        all_modalities_orig_features.append(np.concatenate(all_radio_features_m)[true_top_k_slide])
        #import pdb;pdb.set_trace()

        if all_masks_m != []:
            all_masks.append(np.concatenate(all_masks_m)[true_top_k_slide])
        else:
            mask = np.concatenate(all_radio_features_m)[:,1,:,:].squeeze()
            #import pdb;pdb.set_trace()
            mask = (mask !=  mask[0][0][0]).astype(int)
            all_masks.append(mask)


        """    
        loader = DataLoader(dataset = PreprocessDataset(file_path =file_path, 
                    transform = True, transform_f = transform_orig, plane = 'axial', 
                    cancer_type = args.cancer_type,segment = args.segment), 
                batch_size = 1000, shuffle = False, collate_fn=collate_radio_features, **kwargs)

        for count,(batch,slice_ids) in enumerate(loader):
            orig_radio_features = batch.to(device)   
            all_modalities_orig_features.append(orig_radio_features[top_k_slide])
        """



    return  all_modalities_features,all_modalities_orig_features, all_masks



def initiate_grad_cam_model(ckpt_path):
    print('Initiate model....')
    model_path = os.path.dirname(ckpt_path)
    with open(os.path.join(model_path,f'experiment_{os.path.basename(model_path)}.txt'),'rb') as f:
        experiment_args_dict = eval(f.read())

    model = grad_cam_radio(n_classes = experiment_args_dict['n_classes'], 
        modalities = experiment_args_dict['radio_modality'])
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt, strict=False)
    model.relocate()
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    model.eval()
    model = model.eval()
    return model


def compute_ig(model, img_input, img_orig, other):
    print('Compute IG....')
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(img_input, n_steps=200, internal_batch_size = 1,
                                                    additional_forward_args = other)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(img_input, nt_samples=40, nt_type='smoothgrad_sq',
                                                additional_forward_args = other, nt_samples_batch_size = 1,
                                                stdevs =0.3)
    return attributions_ig, attributions_ig_nt


def vis_ig(attr_ig, img_orig, save_path, mask_bg):
    if mask_bg:
        mask = [img_orig[0][0]==0]
        attr_ig[0][0][mask] = 0
        attr_ig[0][1][mask] = 0
        attr_ig[0][2][mask] = 0
    fig = viz.visualize_image_attr_multiple(np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(img_orig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          ["original_image","heat_map", "blended_heat_map"],
                                          ["all", "absolute_value", "absolute_value"],
                                          show_colorbar=True,
                                         cmap ='Oranges',
                                         alpha_overlay = 0.7,
                                         use_pyplot =False,fig_size = (15,5))
    fig[0].axes[0].set_title(f'Original Highest Attention Slide')
    fig[0].axes[1].set_title('Integrated Gradient Absolute Attribution Heatmap')
    fig[0].axes[2].set_title('Overlayed Heatmap')

    print('Save heatmap...')
    fig[0].savefig(save_path)
    return fig[0]

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      cmap = 'RdYlBu_r') -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cmapy.cmap(cmap))
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = cv2.addWeighted(heatmap,1,img,0.5,0)#heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), np.uint8(255 * mask)

class grad_cam_radio(nn.Module):
    def __init__(self, radio_fusion= 'concat', gate=True, dropout=True,
        model_size_radio: str='small',n_classes=4, modalities = ['T1','T2','T1Gd','FLAIR'], 
        transfer = None, radio_mil_type = 'attention',other =None):
        super(grad_cam_radio, self).__init__()


        self.resnet50 = resnet50_baseline(pretrained=True)
        #self.other = other
        
        
        self.radio_fusion = radio_fusion
        self.n_classes = n_classes
        self.size_dict_radio = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.modalities = modalities
        self.radio_mil_type = radio_mil_type
        print(radio_mil_type)
        
        size_radio = self.size_dict_radio[model_size_radio]

        if len(self.modalities) > 1:
            if self.radio_fusion == 'tensor':
                self.radio_xfusion = XlinearFusion(dim=1024, scale_dim=64,mmhid1 = 1024,mmhid2 = 1024, skip  = 0)

            elif self.radio_fusion == 'concat':
                self.reduce_dim = nn.Linear(size_radio[0]*len(self.modalities), size_radio[0])

        if self.radio_mil_type == 'attention':
            fc_radio = [nn.Linear(size_radio[0], size_radio[1]), nn.ReLU()]
            fc_radio.append(nn.Dropout(0.25))


            if gate:
                attention_net_radio = Attn_Net_Gated(L = size_radio[1], D = size_radio[2], dropout = dropout, n_classes = 1)
            else:
                attention_net_radio = Attn_Net(L = size_radio[1], D = size_radio[2], dropout = dropout, n_classes = 1)

            fc_radio.append(attention_net_radio)
            self.attention_net_radio = nn.Sequential(*fc_radio)

        elif self.radio_mil_type == 'max':
            self.pre_pool = nn.Sequential(*[nn.Linear(size_radio[0], size_radio[1]), nn.ReLU()])


        if transfer:
            classifiers = [nn.Linear(size_radio[1], size_radio[1]), nn.ReLU(),nn.Dropout(0.25), 
            nn.Linear(size_radio[1], size_radio[1]), nn.ReLU(),nn.Dropout(0.25), 
            nn.Linear(size_radio[1], n_classes)]
            self.classifier = nn.Sequential(*classifiers)
        else:
            self.classifier = nn.Linear(size_radio[1], n_classes)   

    def relocate(self):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = self.resnet50.to(device)
        if len(self.modalities) > 1:
            if self.radio_fusion == 'tensor':
                self.xfusion = self.xfusion.to(device)
            elif self.radio_fusion == 'concat':
                self.reduce_dim = self.reduce_dim.to(device)
        if self.radio_mil_type =='attention':
            self.attention_net_radio = self.attention_net_radio.to(device)
        elif self.radio_mil_type =='max':
            self.pre_pool = self.pre_pool.to(device)
        self.classifier = self.classifier.to(device)
    
    def forward(self, x):
        h = self.resnet50(x)
        if self.radio_mil_type == 'attention':
            h = torch.reshape(h,(1,-1))
            if len(self.modalities) >1:
                h = self.reduce_dim(h)
            A, h = self.attention_net_radio(h)  
            A = torch.transpose(A, 1, 0)

            A_raw = A 
            A = F.softmax(A, dim=1) 
            M = torch.mm(A, h)

            #top_A = torch.argmax(A)
            #M = h[top_A:top_A+1]

            logits  = self.classifier(M)        
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            #return hazards
            S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1)

        return [risk]