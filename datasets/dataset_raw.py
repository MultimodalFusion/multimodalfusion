import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
from torchvision import transforms
import torch
from torch.utils.data import Dataset

from utils.ct_preprocess_utils import *
from lungmask import mask


class PreprocessDataset(Dataset):
    #def __init__(self,file_path , small, stride, transform, transform_f, plane, consecutive,consec_gap, cancer_type = 'glioma'):
    def __init__(self,file_path , transform, transform_f, plane, cancer_type = 'glioma',segment = False):

        """
        Args
        file_path(string): Path to the MRI scan
        transform(boolean): Whether the MRI scan needs transformation
        tranform_f(function): Transformation function
        """
        #self.small = small
        #self.consecutive = consecutive
        self.cancer_type = cancer_type
        self.transform = transform
        self.transform_f = transform_f
        self.plane = plane
        #self.consec_gap = consec_gap

        if cancer_type == 'glioma':
            self.img = sitk.ReadImage(file_path)
            standard = (0.0, -239.0, 0.0)
            origin = self.img.GetOrigin()
            flip_dim = [False if origin[i] == standard[i] else True for i in range(3) ]
            self.img = sitk.Flip(self.img, flip_dim)
            assert self.img.GetOrigin() == standard
            self.img_arr = sitk.GetArrayFromImage(self.img)

            #remove all black slices
            if self.plane == 'axial':
                slice_id = [ind for ind in range(self.img_arr.shape[0]) if np.count_nonzero(self.img_arr[ind,:,:]) > 0]
                img_selected = self.img_arr[slice_id,:,:]
            
            #normalize to 0 and 1
            img_final =  np.array([normalize(i,img_selected.min(),img_selected.max()) for i in img_selected])

            #crop the image by its max position
            img_final = crop_image(img_final)

        elif cancer_type == 'lung':
            if self.plane == 'axial':
                self.img = load_scan(file_path)
                img_hu = get_pixels_hu(self.img)

                #orientation
                for i, each_img in enumerate(self.img):
                    ori = each_img.ImageOrientationPatient
                    x = np.round(ori[0:3])
                    y = np.round(ori[3:6])

                    if all(x == [-1,0,0]):
                        img_hu[i] = np.flip(img_hu[i],0)
                    if all(y == [0,-1,0]):
                        img_hu[i] = np.flip(img_hu[i],1)    
                    if all(x == [0,-1,0]) and all(y == [1,0,0]):
                        img_hu[i] = np.rot90(img_hu[i])
                    if all(x == [0,-1,0]) and all(y == [-1,0,0]):
                        img_hu[i] = np.rot90(img_hu[i])
                        img_hu[i] = np.flip(img_hu[i],1)  
                    if all(x == [0,1,0]) and all(y == [1,0,0]):
                        img_hu[i] = np.rot90(img_hu[i])
                        img_hu[i] = np.flip(img_hu[i],0)  
                    if all(x == [0,1,0]) and all(y == [-1,0,0]):
                        img_hu[i] = np.rot90(img_hu[i],3)

                img_hu[img_hu< -1000] = -1000
                pix_resampled, spacing = resample(img_hu, self.img, [1,1.5,1.5])
                if segment:
                    print('Segmentation and place bounding box...')
                    segmentation = mask.apply(pix_resampled)
                    segmented_img = [lung_box(pix_resampled[ind], segmentation[ind])[0] for ind, img in enumerate(pix_resampled)]
                else:
                    segmentation = mask.apply(pix_resampled)
                    segmented_img = largest_lung_box(pix_resampled,segmentation)
                cropped = crop_image(np.array(segmented_img))
                normalized = np.array([normalize(i,-1000,400) for i in cropped])
                slice_id = [ind for ind in range(normalized.shape[0]) if np.count_nonzero(normalized[ind,:,:]) > 0]
                img_final = normalized[slice_id,:,:]
                #img_final = padded[~np.all(padded == 0 ,axis = (1,2))]
            else:
                raise NotImplementedError

        self.slice_id = slice_id
        self.img_final = img_final


    def __len__(self):

        if self.plane == 'axial':
            return self.img_final.shape[0] 

    def __getitem__(self, idx):

        if self.plane == 'axial':
            #temp = self.img_final[idx,:,:][...,None]
            #temp =np.repeat(temp,3,axis=2)#.astype(np.uint8)
            temp = self.img_final[idx,:,:][None,...]
            temp = torch.tensor(np.repeat(temp,3,axis=0))

        if self.transform: 
            img_processed = self.transform_f(temp)
        else:
            img_processed = torch.tensor(temp)

        return([img_processed, self.slice_id[idx]])


        


class PreprocessDatasetMask(Dataset):
    #def __init__(self,file_path , small, stride, transform, transform_f, plane, consecutive,consec_gap, cancer_type = 'glioma'):
    def __init__(self,file_path , transform, transform_f, plane, cancer_type = 'glioma',segment = False):

        """
        Args
        file_path(string): Path to the MRI scan
        transform(boolean): Whether the MRI scan needs transformation
        tranform_f(function): Transformation function
        """
        #self.small = small
        #self.consecutive = consecutive
        self.cancer_type = cancer_type
        self.transform = transform
        self.transform_f = transform_f
        self.plane = plane
        #self.consec_gap = consec_gap

        if cancer_type == 'glioma':
            self.img = sitk.ReadImage(file_path)
            standard = (0.0, -239.0, 0.0)
            origin = self.img.GetOrigin()
            flip_dim = [False if origin[i] == standard[i] else True for i in range(3) ]
            self.img = sitk.Flip(self.img, flip_dim)
            assert self.img.GetOrigin() == standard
            self.img_arr = sitk.GetArrayFromImage(self.img)

            #remove all black slices
            if self.plane == 'axial':
                slice_id = [ind for ind in range(self.img_arr.shape[0]) if np.count_nonzero(self.img_arr[ind,:,:]) > 0]
                img_selected = self.img_arr[slice_id,:,:]
            
            #normalize to 0 and 1
            img_final =  np.array([normalize(i,img_selected.min(),img_selected.max()) for i in img_selected])

            #crop the image by its max position
            img_final = crop_image(img_final)

        elif cancer_type == 'lung':
            if self.plane == 'axial':
                self.img = load_scan(file_path)
                img_hu = get_pixels_hu(self.img)

                #orientation
                for i, each_img in enumerate(self.img):
                    ori = each_img.ImageOrientationPatient
                    x = np.round(ori[0:3])
                    y = np.round(ori[3:6])

                    if all(x == [-1,0,0]):
                        img_hu[i] = np.flip(img_hu[i],0)
                    if all(y == [0,-1,0]):
                        img_hu[i] = np.flip(img_hu[i],1)    
                    if all(x == [0,-1,0]) and all(y == [1,0,0]):
                        img_hu[i] = np.rot90(img_hu[i])
                    if all(x == [0,-1,0]) and all(y == [-1,0,0]):
                        img_hu[i] = np.rot90(img_hu[i])
                        img_hu[i] = np.flip(img_hu[i],1)  
                    if all(x == [0,1,0]) and all(y == [1,0,0]):
                        img_hu[i] = np.rot90(img_hu[i])
                        img_hu[i] = np.flip(img_hu[i],0)  
                    if all(x == [0,1,0]) and all(y == [-1,0,0]):
                        img_hu[i] = np.rot90(img_hu[i],3)

                img_hu[img_hu< -1000] = -1000
                pix_resampled, spacing = resample(img_hu, self.img, [1,1.5,1.5])
                if segment:
                    print('Segmentation and place bounding box...')
                    segmentation = mask.apply(pix_resampled)

                    segmented_img = []
                    segmented_mask = []
                    for ind, img in enumerate(pix_resampled):
                        seg_img , seg_mask = lung_box(pix_resampled[ind], segmentation[ind])
                        segmented_img.append(seg_img)
                        segmented_mask.append(seg_mask)
                    #segmented_img = [lung_box(pix_resampled[ind], segmentation[ind])[0] for ind, img in enumerate(pix_resampled)]
                else:
                    segmentation = mask.apply(pix_resampled)
                    segmented_img = largest_lung_box(pix_resampled,segmentation)
                cropped = crop_image(np.array(segmented_img))
                cropped_mask = crop_image(np.array(segmented_mask))

                normalized = np.array([normalize(i,-1000,400) for i in cropped])
                slice_id = [ind for ind in range(normalized.shape[0]) if np.count_nonzero(normalized[ind,:,:]) > 0]
                img_final = normalized[slice_id,:,:]
                self.mask = cropped_mask[slice_id,:,:]
                assert len(cropped) == len(cropped_mask)
                #img_final = padded[~np.all(padded == 0 ,axis = (1,2))]
            else:
                raise NotImplementedError

        self.slice_id = slice_id
        self.img_final = img_final



    def __len__(self):

        if self.plane == 'axial':
            return self.img_final.shape[0] 

    def __getitem__(self, idx):

        if self.plane == 'axial':
            #temp = self.img_final[idx,:,:][...,None]
            #temp =np.repeat(temp,3,axis=2)#.astype(np.uint8)
            temp = self.img_final[idx,:,:][None,...]
            temp = torch.tensor(np.repeat(temp,3,axis=0))

            if self.cancer_type == 'lung':
                temp_mask = self.mask[idx,:,:][None,...]
                temp_mask = torch.tensor(np.repeat(temp_mask,3,axis=0))

        transform_orig = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.CenterCrop(224),
                            transforms.ToTensor()
                        ])  

        if self.transform: 
            img_processed = self.transform_f(temp)
            if self.cancer_type == 'lung':

                mask_processed = transform_orig(temp_mask)
            else:
                mask_processed = None
            
        else:
            img_processed = torch.tensor(temp)
            if self.cancer_type == 'lung':
                mask_processed = torch.tensor(temp_mask)
            else:
                mask_processed = None

        return([img_processed, self.slice_id[idx], mask_processed])


        

    
