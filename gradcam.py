import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import sys

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import h5py
import time
import argparse
import pickle
from PIL import Image
from utils.utils_ig import *
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--radio_dir', type=str, default=None)
parser.add_argument('--patches_dir', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--cancer_type', type=str, default='glioma')
parser.add_argument('--modalities', type=str, default='T1,T2,T1Gd,FLAIR')
parser.add_argument('--segment', action='store_true', default=False, help='Segment Lung?')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--subject', type=str, default=None)
parser.add_argument('--all_slices', action='store_true', default=False, help='ALL slices?')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite original files')


args = parser.parse_args()
args.modalities = [m for m in args.modalities.split(',')]



if __name__ == '__main__':
    #read dataset
    dataset_path = args.csv_path
    radio_dir = args.radio_dir
    patches_dir = args.patches_dir
    overwrite = args.overwrite
    all_slices = args.all_slices
    subject = args.subject
    datasets = pd.read_csv(dataset_path)

    selected=['sfb11','sfb47']#['TCGA-02-0033','TCGA-06-0176','TCGA-CS-4941','TCGA-HT-7473']
    datasets =datasets[datasets.subject_id.isin(selected)].reset_index(drop = False)
    #import pdb;pdb.set_trace()
    #initiate model
    model = initiate_grad_cam_model(args.ckpt_path)

    #gradcam
    target_layers = [model.resnet50.layer3[-1]]
    subjects_modalities = datasets[['subject_id']+args.modalities].dropna().set_index('subject_id').T.to_dict()
    
    if args.subject is not None:
        subjects_modalities = {args.subject:subjects_modalities[args.subject]}
    #Loop through all patient
    for subject_id, subject_m in subjects_modalities.items():
        all_subjects = os.listdir(patches_dir)
        if subject_id not in all_subjects:
            continue

        if 'ig_heatmap' in os.listdir(os.path.join(patches_dir, subject_id)) and not overwrite:
            continue
        #if subject_id not in ['sfb11','sfb47']:#'TCGA-50-5044':
        #    continue

        print(f'----------------{subject_id}---------------')
        subject_m = {k:v for k, v, in subject_m.items() if not pd.isna(v)}
        # top k most important
        scores_slices = pd.read_csv(os.path.join(patches_dir,subject_id,'scores.csv'))
        #import pdb;pdb.set_trace()
        if all_slices:
            process_slide = scores_slices.sort_values('ind')['ind']
        else:
            process_slide = scores_slices.iloc[0:args.top]['ind']

         
        all_modalities_features,all_modalities_orig_features, all_masks = get_slide_features(args, subject_m , subject_id, process_slide)

        all_heatmap = {}
        for k, which_slide in enumerate(process_slide):
            print(f'{k} - slide {which_slide}')
            #get images 
            #import pdb;pdb.set_trace()
            img_input = torch.cat([img_slide[[k]] for img_slide in all_modalities_features],dim =0)


            with GradCAMPlusPlus(model=model, target_layers=target_layers, 
                                 use_cuda=torch.cuda.is_available()) as cam:
                cam.batch_size = 4
                targets = [ClassifierOutputTarget(0)]
                grayscale_cam = cam(input_tensor=img_input, targets=targets , aug_smooth=True)
                for i in range(grayscale_cam.shape[0]):
                    grayscale_cam_i = grayscale_cam[i, :]
                    rgb_img = np.transpose(img_input[i].squeeze().cpu().detach().numpy(), (1,2,0))
                    rgb_img = (rgb_img-rgb_img.min())/(rgb_img.max()-rgb_img.min())
                    #rgb_img = np.transpose(img_orig.squeeze()[i].cpu().detach().numpy(), (1,2,0))
                    visualization, heatmap = show_cam_on_image(rgb_img, grayscale_cam_i, use_rgb=True)
                    if not all_slices:
                        os.makedirs(os.path.join(patches_dir, subject_id,'ig_heatmap'),exist_ok = True)
                        save_path = os.path.join(patches_dir, subject_id,'ig_heatmap',f'{args.modalities[i]}_{k}_{which_slide}.png')
                        im = Image.fromarray(visualization)
                        im.save(os.path.join(save_path))
                    else:
                        if i in all_heatmap:
                            all_heatmap[i].append(grayscale_cam_i)
                        else:
                            all_heatmap[i] = [grayscale_cam_i]


        if all_slices:
            #import pdb;pdb.set_trace()
            all_heatmap_max=np.max([np.max(h) for h_m, h in all_heatmap.items()])
            all_heatmap_min=np.min([np.min(h) for h_m, h in all_heatmap.items()])
            all_final_scores_blur = []
            for m_i, modality in enumerate(args.modalities):
                #OBTAIN ATTENTION SCORES
                scores_new = scores_slices.sort_values('ind').scores
                scores_new = (scores_new - scores_new.min())/(scores_new.max()-scores_new.min())
                scores_new = np.concatenate([np.expand_dims(np.ones((224,224))*i,axis = 0) for i in scores_new])#.shape
                m_attr = np.concatenate([ np.expand_dims(i, axis=0) for i in all_heatmap[m_i]])#.shape

                for mask_i in range(len(m_attr)):
                    #import pdb;pdb.set_trace()
                    minimum_thresh = m_attr.min()
                    try:
                        m_attr[mask_i][all_masks[m_i][:,1,:,:][mask_i].squeeze()==0] = minimum_thresh
                    except:
                        m_attr[mask_i][all_masks[m_i][mask_i].squeeze()==0] = minimum_thresh
                    #final_scores[mask_i][grayscale_mri_ct[mask_i]==grayscale_mri_ct[m_i].min()] = 0
                #import pdb;pdb.set_trace()
                
                m_attr = (m_attr-all_heatmap_min)/(all_heatmap_max-all_heatmap_min)
                final_scores = m_attr * scores_new
                final_scores_blur = gaussian_filter(final_scores, sigma=[5,1,1])
                all_final_scores_blur.append(final_scores_blur)


            all_final_scores_blur_max = np.max([np.max(h) for h in all_final_scores_blur])
            all_final_scores_blur_min = np.min([np.min(h) for h in all_final_scores_blur])
            for m_i, modality in enumerate(args.modalities):
                af = np.array([[ -1.,   0.,   0.,  -0.],
                       [  0.,  -1.,   0., 239.],
                       [  0.,   0.,   1.,   0.],
                       [  0.,   0.,   0.,   1.]])
                grayscale_mri_ct = all_modalities_orig_features[m_i][:,1,:,:].squeeze()
                temp = nib.Nifti1Image(np.transpose(grayscale_mri_ct, (2, 1, 0)),af)
                nib.save(temp, os.path.join(patches_dir, subject_id,f'{subject_id}_{modality}_orig.nii.gz'))  

                final_scores_blur = all_final_scores_blur[m_i]
                final_scores_blur = (final_scores_blur -all_final_scores_blur_min)/(all_final_scores_blur_max-all_final_scores_blur_min)
                temp_attr = nib.Nifti1Image(np.transpose(final_scores_blur, (2, 1, 0)),af)
                nib.save(temp_attr, os.path.join(patches_dir, subject_id,f'{subject_id}_{modality}_attr.nii.gz'))  

                os.makedirs(os.path.join(patches_dir, subject_id,'ig_heatmap_all',modality),exist_ok =True)

                #grayscale_mri_ct = all_modalities_orig_features[m_i][:,1,:,:].squeeze()
                for overlay_i in range(len(final_scores_blur)):

                    grayscale_mri_ct_i = grayscale_mri_ct[overlay_i][None,...]
                    grayscale_mri_ct_i = (grayscale_mri_ct_i-grayscale_mri_ct_i.min())/(grayscale_mri_ct_i.max()-grayscale_mri_ct_i.min())
                    grayscale_mri_ct_i = np.transpose(np.repeat(grayscale_mri_ct_i,3,axis=0), (1,2,0))


                    #rgb_img_i = (rgb_img_i-rgb_img_i.min())/(rgb_img_i.max()-rgb_img_i.min())
                    final_scores_i = final_scores_blur[overlay_i]                    
                    visualization, heatmap = show_cam_on_image(grayscale_mri_ct_i, final_scores_i, use_rgb=True)

                    save_path = os.path.join(patches_dir, subject_id,'ig_heatmap_all',modality,'all_'+'{:03}'.format(overlay_i)+'.png')
                    im = Image.fromarray(np.concatenate([np.uint8(255 * grayscale_mri_ct_i),visualization],axis =1))
                    im.save(os.path.join(save_path))



            with open(os.path.join(patches_dir, subject_id,'heatmap.pkl'),'wb') as handle:
                pickle.dump(all_heatmap, handle, protocol=pickle.HIGHEST_PROTOCOL)


            #attr_ig, attr_ig_nt = compute_ig(model, img_input, img_orig, other)
            #vis_ig(attr_ig_nt, img_orig, save_path, True)






