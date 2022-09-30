import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from datasets.wsi_dataset import Wsi_Region
import h5py
import SimpleITK as sitk

from utils.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore

from models.model_attention_mil_path import MIL_Attention_fc_path, MIL_Attention_fc_surv_path
from models.model_attention_mil_radio import MIL_Attention_fc_radio, MIL_Attention_fc_surv_radio
from models.model_mm_attention_mil import MM_MIL_Attention_fc, MM_MIL_Attention_fc_surv
from models.model_genomic import MaxNet
from models.resnet_custom import resnet50_baseline
from utils.ct_preprocess_utils import *
from lungmask import mask

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords,  slide_path=None, wsi_object=None, heatmap_vis_args = None, annotation=None ):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    vis_level = heatmap_vis_args['vis_level']
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
        heatmap_vis_args['vis_level'] = vis_level
    if annotation is not None:
        wsi_object.initXML(annotation)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, **heatmap_vis_args)
    return heatmap

def initialize_wsi(wsi_path, mask_file='', seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if os.path.isfile(mask_file):
        wsi_object.initSegmentation(mask_file)
        return wsi_object

    else:
        if seg_params['seg_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            seg_params['seg_level'] = best_level

        wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
        if mask_file is not None:
            wsi_object.saveSegmentation(mask_file)
        return wsi_object

def compute_fineheatmap(wsi_object, label=None, model=None, feature_extractor=None, batch_size=512, seg_params=None, filter_params=None, 
    vis_params=None, save_path=None, mask_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']
    
    # wsi = wsi_object.getOpenSlide()
    # if vis_params['vis_level'] < 0:
    #     best_level = wsi.get_best_level_for_downsample(32)
    #     vis_params['vis_level'] = best_level
    # mask = wsi_object.visWSI(**vis_params)
    # mask.save(mask_path)

    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', len(roi_loader))
    mode = "w"
    for idx, (roi, coords) in enumerate(roi_loader):
        roi = roi.to(device)
        with torch.no_grad():
            features = feature_extractor(roi)
            A = model(x_path=features, attention_only=True)
        if idx % math.ceil(num_batches * 0.05) == 0:
            print('procssed {} / {}'.format(idx, num_batches))
        
        A = A.view(-1, 1).cpu().numpy()
        coords = coords.numpy()
        if ref_scores is not None:
            for score_idx in range(len(A)):
                A[score_idx] = score2percentile(A[score_idx], ref_scores)

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        # save_path = save_attention_hdf5_file(A, coords, save_path, mode=mode)
        asset_dict = {'attention_scores': A, 'coords': coords}
        save_path = save_hdf5(save_path, asset_dict, mode=mode)
        mode = "a"
    return save_path, wsi_object

def compute_from_patches(wsi_object, clam_pred=None, model=None, feature_extractor=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']
    #import pdb;pdb.set_trace()

    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', len(roi_loader))
    mode = "w"
    #genomic_features =genomic_features.to(device)
    for idx, (roi, coords) in enumerate(roi_loader):
        roi = roi.to(device)
        #coords = coords#.numpy()
        
        with torch.no_grad():
            features = feature_extractor(roi)
            if attn_save_path is not None:
                #A = model(x_path=features, x_omic=genomic_features, meta=0, attention_only=True)
                _, _, _, A= model(path_features=features) 
                A = A.view(-1, 1).cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
        if idx % math.ceil(num_batches * 0.05) == 0:
            print('procssed {} / {}'.format(idx, num_batches))

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object

def email_progress(study: str, model: str):
    import smtplib, ssl

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "mahmoodlab.bwh@gmail.com"  # Enter your address
    receiver_email = "richardchen@g.harvard.edu"  # Enter receiver address
    password = 'faisalmahmood1'
    message = """\
    Subject: %s w/ %s Done

    This message is sent from Python.""" % (study, model)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

### PROCESS RADIOLOGY IMAGES
#def mri_normalize(arr):
#    max_value = arr.max()
#    min_value = arr.min()
#    arr = (arr - min_value) / (max_value - min_value)
#    return arr

def radio_img(file_path, modality_files):
    img_arrs={}

    for m, file in modality_files.items():
        if m != 'CT':
            img = sitk.ReadImage(os.path.join(file_path,file))
            standard = (0.0, -239.0, 0.0)
            origin = img.GetOrigin()
            flip_dim = [False if origin[i] == standard[i] else True for i in range(3) ]
            img = sitk.Flip(img, flip_dim)
            assert img.GetOrigin() == standard
            img_arr = sitk.GetArrayFromImage(img)

            #img = sitk.ReadImage(os.path.join(file_path,file))
            #img_arr = sitk.GetArrayFromImage(img)
            #slice_id = [ind for ind in range(img_arr.shape[0]) if np.count_nonzero(img_arr[ind,:,:]) > 0]
            #img_selected = img_arr[slice_id,:,:]
            #img_final = img_arr * (255.0/img_arr.max())
            #img_final = img_arr.astype(np.uint8)
            #img_final = crop_image(img_final)
            #slice_id = [ind for ind in range(img_arr.shape[0]) if np.count_nonzero(img_arr[ind,:,:]) > 0]
            #img_selected = img_arr[slice_id,:,:]
            #img_final =  np.array([normalize(i,img_selected.min(),img_selected.max()) for i in img_selected])
            #crop the image by its max position
            #img_final = crop_image(img_final)
            img_arrs[m] = img_arr


        else:
            img = load_scan(os.path.join(file_path,file))
            img_hu = get_pixels_hu(img)
            img_hu[img_hu< -1000] = -1000
            pix_resampled, spacing = resample(img_hu, img, [1,1.5,1.5])
            if True:
                print('Segmentation and place bounding box...')
                segmentation = mask.apply(pix_resampled)
                segmented_img = [lung_box(pix_resampled[ind], segmentation[ind])[0] for ind, img in enumerate(pix_resampled)]
                #cropped = crop_image(np.array(segmented_img))
            else:
                segmentation = mask.apply(pix_resampled)
                segmented_img = largest_lung_box(pix_resampled,segmentation)

            cropped = crop_image(np.array(segmented_img))
            normalized = np.array([normalize(i,-1000,400) for i in cropped])
            #slice_id = [ind for ind in range(normalized.shape[0]) if np.count_nonzero(normalized[ind,:,:]) > 0]
            #img_arr = normalized[slice_id,:,:]
            img_arrs[m] = normalized


    return img_arrs

def process_mri(radio_h5_folder, subject_id, modality_files, modalities):
    radio_features = {}
    slices_index = {}
    if all(pd.isna((list(modality_files.values())))):
        for m in modalities:
            radio_features[m] = torch.zeros((1,1))
    else:
        for m in modalities:
            radio_path = os.path.join(radio_h5_folder,m,f'{subject_id}.h5')
            file = h5py.File(radio_path, "r")
            features = file['features'][:]
            slice_id = file['slice_index'][:]
            radio_features[m] = features
            slices_index[m] = slice_id 
        intersect = list(set.intersection(*[set(v) for k, v in slices_index.items()]))
        for m in modalities:
            radio_features[m] = torch.tensor(radio_features[m][np.in1d(slices_index[m], intersect),:])

    return radio_features,intersect


def infer_patient(model, features,bins=None, label=None):
    with torch.no_grad():
        if isinstance(model,(MIL_Attention_fc_surv_path)):
            path_features = features.to(device)
            hazards, survival, Y_hat_model, A = model(path_features=path_features) 
            risk = -torch.sum(survival, dim=1).cpu().numpy()
            #import pdb;pdb.set_trace()
            if bins is not None:
                Y_hat = np.digitize(risk, np.array(bins))[0] - 1
            else:
                Y_hat = None
            A_final = A.view(-1, 1).cpu().numpy()
        elif isinstance(model,(MIL_Attention_fc_surv_radio)):
            hazards, survival, Y_hat_model, A = model(**features) 
            risk = -torch.sum(survival, dim=1).cpu().numpy()
            if bins is not None:
                Y_hat = np.digitize(risk, np.array(bins))[0] - 1
            else:
                Y_hat = None
            A_final = A.view(-1, 1).cpu().numpy()
        else:
            raise NotImplementedError

        Y_hat_model = Y_hat_model.cpu().numpy()[0][0]
        print('Y_hat: {}, Y: {}, risk: {}, hazards: {}'.format(Y_hat, label, risk, ["{:.4f}".format(p) for p in hazards.cpu().flatten()]))    

    return Y_hat_model, risk, A_final

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            if not np.isnan(val):
                params[key] = dtype(val)

    return params


def process_single_slide(slide_entry, r_case_save_dir, patch_args, data_args, heatmap_args, exp_args, feature_extractor,inference_mode=False):
    slide_name = slide_entry['slide_id']
    slide_id = slide_name.replace(data_args.slide_ext, '')
    if data_args.slide_ext not in slide_name:
        slide_name+=data_args.slide_ext
    
    if False: print('\nprocessing: ', slide_name)   

    if isinstance(data_args.data_dir, str):
        slide_path = os.path.join(data_args.data_dir, slide_name)
    elif isinstance(data_args.data_dir, dict):
        data_dir_key = process_stack[data_args.data_dir_key]
        slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
    else:
        raise NotImplementedError

    h5_path = os.path.join(r_case_save_dir, slide_id+'.h5')
    if not os.path.isfile(h5_path):
        os.system('cp %s %s' % (os.path.join(data_args.data_h5_dir, slide_id+'.h5'), h5_path))

    mask_file = os.path.join(r_case_save_dir, slide_id+'_mask.pkl')
    if not os.path.isfile(mask_file):
        os.system('cp %s %s' % (os.path.join(data_args.mask_dir,'pkl_files', slide_id+'_mask.pkl'), mask_file))

    mask_path = os.path.join(r_case_save_dir, '{}_mask.jpg'.format(slide_id))
    if not os.path.isfile(mask_path):
        os.system('cp %s %s' % (os.path.join(data_args.mask_dir,'jpg_files', slide_id+'_mask.jpg'), mask_path))

    flag = False

    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False}#, 'keep_ids': 0,'exclude_ids':0}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}


    if (not os.path.isfile(mask_file)) or (not os.path.isfile(mask_path)):
        flag = True
        ### Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(slide_entry, seg_params)
        filter_params = load_params(slide_entry, filter_params)
        vis_params = load_params(slide_entry, vis_params)

        if False: print('Initializing WSI object')

        wsi_object = initialize_wsi(slide_path, mask_file = mask_file, seg_params=seg_params, filter_params=filter_params)
        if False: print('Done!')

        # wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        # vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_case_save_dir, '{}_blockmap.h5'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        
        mask = wsi_object.visWSI(**vis_params)
        mask.save(mask_path)

    ##### Check if segmentation_file exists ######
        wsi_object.saveSegmentation(mask_file)
    
    try:
        return_patch_args= None
        #import pdb;pdb.set_trace()
        if any([param_i not in slide_entry.index for param_i in ['patch_size','custom_downsample','patch_level','step_size'] ]):
            seg_params = def_seg_params.copy()
            filter_params = def_filter_params.copy()
            vis_params = def_vis_params.copy()

            seg_params = load_params(slide_entry, seg_params)
            filter_params = load_params(slide_entry, filter_params)

            wsi_object = initialize_wsi(slide_path,  mask_file = mask_file, seg_params=seg_params, 
                filter_params=filter_params)
            level0_mag, new_patch_level, new_patch_size, new_step_size, mag_downsample = wsi_object.fetch_mag_patching_params(mag_level=20)
            return_patch_args = {'patch_size': new_patch_size,'custom_downsample':mag_downsample,
            'patch_level':new_patch_level,'step_size':new_step_size}

        file = h5py.File(h5_path, "r")
    except OSError:
        ### Load segmentation and filter parameters
        blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 
            'patch_size': patch_args.patch_size, 'step_size': patch_args.patch_size, 
            'custom_downsample':patch_args.custom_downsample, 
            'level': patch_args.patch_level, 
            'use_center_shift': heatmap_args.use_center_shift}
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()
        blocky_wsi_kwargs_params = blocky_wsi_kwargs.copy()

        seg_params = load_params(slide_entry, seg_params)
        filter_params = load_params(slide_entry, filter_params)
        vis_params = load_params(slide_entry, vis_params)
        
        #blocky_wsi_kwargs_params = load_params(slide_entry, blocky_wsi_kwargs_params)

        wsi_object = initialize_wsi(slide_path,  mask_file = mask_file, seg_params=seg_params, 
            filter_params=filter_params)
        level0_mag, new_patch_level, new_patch_size, new_step_size, mag_downsample = wsi_object.fetch_mag_patching_params(mag_level=20)
        return_patch_args = {'patch_size': new_patch_size,'custom_downsample':mag_downsample,
        'patch_level':new_patch_level,'step_size':new_step_size}

        blocky_wsi_kwargs_params['patch_size'] =tuple([new_patch_size]*2)
        blocky_wsi_kwargs_params['step_size'] =tuple([new_step_size]*2)
        blocky_wsi_kwargs_params['custom_downsample'] = mag_downsample if mag_downsample is not None else 1
        blocky_wsi_kwargs_params['level'] = new_patch_level

    

        _, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
                                        model=None, feature_extractor=feature_extractor, 
                                        batch_size=exp_args.batch_size, **blocky_wsi_kwargs_params, 
                                        attn_save_path=None, feat_save_path=h5_path, #genomic_features=None,
                                        ref_scores=None)



    return slide_id, slide_path, return_patch_args, h5_path, mask_file, flag

from PIL import Image

def getConcatImage(imgs, how='horizontal', gap=32):
    gap_dist = (len(imgs)-1)*gap
    
    if how == 'vertical':
        w, h = np.min([img.width for img in imgs]), np.sum([img.height for img in imgs])
        curr_h = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))
        for img in imgs:
            dst.paste(img, (0, curr_h))
            curr_h += img.height

    elif how == 'horizontal':
        w, h = np.sum([img.width for img in imgs]), np.min([img.height for img in imgs])
        w += gap_dist
        curr_w = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))

        for idx, img in enumerate(imgs):
            dst.paste(img, (curr_w, 0))
            curr_w += img.width + gap

    return dst


def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict
