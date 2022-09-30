from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from tqdm import tqdm
import sys
import warnings
import logging
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import shap
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns
import argparse
from math import floor
import matplotlib.pyplot as plt
import itertools
import nibabel as nib
from captum.attr import IntegratedGradients, GradientShap
from captum.attr import LayerConductance
from utils.utils import *
from utils.eval_utils import initiate_model as initiate_model
from models.model_attention_mil_path import MIL_Attention_fc_path, MIL_Attention_fc_surv_path
from models.model_attention_mil_radio import MIL_Attention_fc_radio, MIL_Attention_fc_surv_radio
from models.model_mm_attention_mil import MM_MIL_Attention_fc, MM_MIL_Attention_fc_surv
from models.model_genomic import MaxNet

from models.resnet_custom import resnet50_baseline
from utils.batch_process_utils import initialize_df#_heatmap
from utils.wsi_utils import sample_rois#,to_percentiles_na
from utils.file_utils import save_hdf5
from utils.heatmap_utils import *

#from utils.shap_utils import *
from utils_analysis.evaluation import load_risk_df, load_genomic_df, getIndividualSHAP, getSHAPLocalExplanationPlot, getGlobalShap

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file',  type=str, default="config_path.yaml")
parser.add_argument('--sampling',  action='store_true', default=False)
parser.add_argument('--heatmap', action='store_true', default=False)
#parser.add_argument('--silent', '-s', action='store_true', default=False)
#parser.add_argument('--exp', '-exp', type=str, default='./heatmaps/wsi/')

args = parser.parse_args()


def interpret_patient_omic(x_omic):
    return model.captum(x_omic)


if __name__ == '__main__':
    ### Initializing Config File
    #inference_mode = True
    sampling_mode = args.sampling
    heatmap_mode = args.heatmap
    #config_path = os.path.join('./heatmaps', args.exp, 'configs', args.config_file)
    config_path = os.path.join(args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    #model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    model_args.ckpt_path=os.path.join(model_args.model_path,f's_{ str(model_args.cv) }_minloss_checkpoint.pt')
    #model_args.pkl_path=os.path.join(model_args.pkl_path,f'split_train_val_{str(model_args.cv)}_results.pkl')
    #model_args.pkl_path=os.path.join(model_args.model_path,f'split_train_test_{str(model_args.cv)}_results.pkl')
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    preset = pd.read_csv(os.path.join('./heatmaps', data_args.cancer_type, 'presets',data_args.preset))
    df = pd.read_csv(os.path.join('./heatmaps', data_args.cancer_type, 'process_lists', data_args.process_list))
    df = df.merge(preset,how = 'left',on = 'slide_id')

    #mask = df['process'] == 1
    process_stack = df.set_index('subject_id')
    process_stack['bag_size'] = -1234
    process_stack['risk_hat'] = -1234
    process_stack['Y_hat'] = -1234
    total = len(process_stack)

    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False}#, 'keep_ids': 0,'exclude_ids':0}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if data_args.preset is not None:
        preset = pd.read_csv(os.path.join('./heatmaps', data_args.cancer_type, 'presets',data_args.preset))
        for key in def_seg_params.keys():
            def_seg_params[key] = preset.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset.loc[0, key]


    if "omic" in model_args.mode: 
        train_data_df, test_data_df, test_data_orig = load_genomic_df(process_stack = process_stack, 
            dataset_path = os.path.join('./dataset_csv/', data_args.cancer_type,f'{model_args.task}.csv'),
            splits_path = os.path.join('./splits', data_args.cancer_type,
                exp_args.split,f'splits_{model_args.cv}.csv' ),
            split_mode = model_args.split_mode,
            modalities = model_args.modalities) 
        #import pdb;pdb.set_trace()
        model_args.omic_input_dim= test_data_df.shape[1]

    if model_args.initiate_fn == 'initiate_model':
        model =  initiate_model(model_args, model_args.ckpt_path)
        #print(model)
    else:
        raise NotImplementedError

    feature_extractor = resnet50_baseline(pretrained=True)
    feature_extractor.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

    bins = exp_args.bins
    if bins is not None:
        bins.reverse()

    ######################################
    #START GENERATING HEATMAPS
    ######################################

    #PLOT GLOBAL ATTRIBUTION SHAP PLOT
    if "omic" in model_args.mode:
        import matplotlib as mpl

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['axes.linewidth'] = 2

        sns.set(font="Helvetica")
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("white")

        os.makedirs(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code), exist_ok=True)


        test_data_df.columns = test_data_df.columns.str.replace('_mut', ' [MUT]')
        test_data_df.columns = test_data_df.columns.str.replace('_cnv', ' [CNV]')

        train_data_df.columns = train_data_df.columns.str.replace('_mut', ' [MUT]')
        train_data_df.columns = train_data_df.columns.str.replace('_cnv', ' [CNV]')

        explainer = shap.GradientExplainer(model,torch.tensor(train_data_df.values).to(device).float())
        test_shap_attr = explainer.shap_values(torch.tensor(test_data_df.values).to(device).float())
        train_shap_attr = explainer.shap_values(torch.tensor(train_data_df.values).to(device).float())
        #shap.summary_plot(shap_attr,test_data_df,show = False, plot_type = 'dot',max_display = 20)

        _ = getGlobalShap(train_data_df,test_data_df,train_shap_attr,test_shap_attr, save_path = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code,'global_shap.eps'))

        #fig.delaxes(fig.axes[1])

        """
        plt.figure(figsize = (30,20))
        shap.dependence_plot('1p_cnv',shap_attr,test_data_df,show = False)
        plt.savefig(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code,'dependence_shap.png'))
        """
        for ind ,subject_id in tqdm(enumerate(test_data_orig.index)):
            if model_args.pkl_path is not None:
                rg_df = load_risk_df(model_args.pkl_path)
                if subject_id not in rg_df.index:
                    #continue
                    risk_group = -1
                else:
                    risk_group = rg_df.loc[subject_id]['strat']
                    if isinstance(risk_group, pd.Series):
                        risk_group = risk_group.iloc[0]

                if risk_group not in [-1, 0, 1]:
                    continue

                if risk_group == 1:
                    risk2str = 'IH'
                elif risk_group == 0:
                    risk2str = 'IL'
                else:
                    risk2str = 'unknown'
            else:
                risk2str = 'unknown'

            r_case_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, risk2str, subject_id)
            os.makedirs(r_case_save_dir, exist_ok=True)
            r_shap_save_dir = os.path.join(r_case_save_dir, 'shap')
            os.makedirs(r_shap_save_dir, exist_ok=True)
            _ = getIndividualSHAP(test_shap_attr=test_shap_attr[[ind]], test_data_df=test_data_df, 
                train_data_df = train_data_df, case_id=subject_id, save_path=r_shap_save_dir)

            with open(os.path.join(r_shap_save_dir, '%s_shap.pkl' % (subject_id)), 'wb') as handle:
                    pickle.dump(test_shap_attr[ind], handle, protocol=pickle.HIGHEST_PROTOCOL)

        sys.exit()

    skip = []
    pbar = tqdm(enumerate(process_stack.index.unique()))
    selected_samples =['TCGA-60-2723']#['TCGA-51-4079','TCGA-50-5044','C3N-01799',
    #['C3N-00959','C3N-01799','TCGA-60-2723','TCGA-34-5928']#['TCGA-06-0176','TCGA-CS-4941','TCGA-HT-7473']
    #['TCGA-50-5044','TCGA-38-4629','TCGA-34-5231','TCGA-51-4079']#['C3N-00959','C3N-01799','TCGA-60-2723','TCGA-34-5928'] 

    #['sfb42','sfb47','sfb48','sfb11']
    #['TCGA-38-A44F','TCGA-50-6673','TCGA-J2-A4AD','TCGA-34-2605','TCGA-51-4079','C3N-01846']
    #['TCGA-02-0033','TCGA-27-2526','TCGA-HT-A5RC','TCGA-DU-6405'] #os.listdir('/mnt/sdb1/For_Drew/LGG/low risk') + os.listdir('/mnt/sdb1/For_Drew/LGG/high risk')

    pbar = tqdm(enumerate(selected_samples)) 


    for idx, subject_id in pbar:
        if idx < 0:
            continue
        #import pdb;pdb.set_trace()


        slide_df = process_stack.loc[subject_id]
        if isinstance(slide_df, pd.core.series.Series):
            slide_df = slide_df.to_frame().T
        else:
            slide_df = slide_df.reset_index()
        try:
            label = slide_df['disc_label'].values[0]
        except KeyError:
            label = 'Unspecified'


        grouping = label

        logging.basicConfig(filename=os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code,'error.log'), 
            filemode='w', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
        logger=logging.getLogger(__name__)
        #GET PATHOLOGY FEATURES
        if "path" in model_args.mode:

            p_case_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping), subject_id)
            r_case_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  subject_id)
            os.makedirs(p_case_save_dir, exist_ok=True)
            os.makedirs(r_case_save_dir, exist_ok=True)

            slide_ids, f_paths, h_paths, m_paths, s_paths = [], [], [], [], []
            slides_patch_args = {}
            flags = []
            for slide_idx in range(len(slide_df)):
                slide_entry = slide_df.loc[slide_df.index[slide_idx]]

                pbar.set_description("%s: Generating Features for Slide %s..." % (subject_id, slide_entry['slide_id'][:24]))
                slide_id, slide_path, return_patch_args, h5_path, mask_file, flag = process_single_slide(slide_entry, r_case_save_dir, 
                    patch_args, data_args, heatmap_args, exp_args, feature_extractor,inference_mode=True)
                slide_ids.append(slide_id)
                h_paths.append(h5_path)
                m_paths.append(mask_file)
                s_paths.append(slide_path)
                flags.append(flag)
                try:
                    slides_patch_args[slide_id] = slide_entry[['patch_size','step_size','patch_level','custom_downsample']].to_dict()
                except:
                    slides_patch_args[slide_id] = return_patch_args

            counts, path_features, coords = [], [], []

            for h5_path in h_paths:
                file = h5py.File(h5_path, "r")
                path_features.append(file['features'][:])
                coords.append(file['coords'][:])
                counts.append(len(file['coords']))
                file.close()
            path_features = torch.tensor(np.vstack(path_features))
            coords = np.vstack(coords)
            counts = np.array(counts)
            cum_counts = np.append(np.array([0]), np.cumsum(counts))
            #import pdb;pdb.set_trace()
            path_features = path_features.to(device)    


            #RESULT
            Y_hat, risk, A_path  = infer_patient(model,path_features, bins, label)

            #Histopathology heatmap

            pbar.set_description("%s: Generating Blockmap" % subject_id)
            block_map_save_path = os.path.join(r_case_save_dir, '{}_blockmap.h5'.format(subject_id))
            if not os.path.isfile(block_map_save_path): 
                asset_dict = {'attention_scores': A_path, 'coords': coords, 'counts': counts}
                block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
                        
            scores = A_path
            # scores = to_percentiles(scores)

            heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 
                        'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample,# 'cmap': 'jet',
                        'alpha':heatmap_args.alpha,'binarize':heatmap_args.binarize,'blank_canvas': heatmap_args.blank_canvas,
                        'thresh':heatmap_args.binary_thresh,'overlap':patch_args.overlap}

            if heatmap_args.use_ref_scores:
                heatmap_vis_args['convert_to_percentiles'] = False
            
            slide_df = slide_df.set_index('slide_id')

            if heatmap_mode:

                h5_all_path = []
                for idx, slide_id in enumerate(slide_ids):
                    pbar.set_description("%s: Sampling Patches - %s" % (subject_id, slide_id[:23]))
                    slide_patch_args = slides_patch_args[slide_id]

                    slide_path = s_paths[idx]
                    mask_file = m_paths[idx]
                    # Load segmentation and filter parameters
                    if False: print('\nDrawing heatmap for {}'.format(slide_id))
                    #import pdb;pdb.set_trace()
                    wsi_object = initialize_wsi(slide_path, mask_file=mask_file)
                    wsi_ref_downsample = wsi_object.level_downsamples[slide_patch_args['patch_level']]
                    vis_patch_size = tuple((np.array(slide_patch_args['patch_size']) * np.array(wsi_ref_downsample) * slide_patch_args['custom_downsample']).astype(int))
                    heatmap_vis_args['patch_size']= vis_patch_size
                    slide_scores = scores[cum_counts[idx]:cum_counts[idx+1]]  
                    slide_coords = coords[cum_counts[idx]:cum_counts[idx+1]]

                    bag_size = len(slide_scores)
            
                    pbar.set_description("%s: Generating LR Heatmap - %s" % (subject_id, slide_id[:23]))
                    
                    #if not production:
                    #    ### Heatmap Generation
                    #    heatmap_save_name = '{}_blockmap.png'.format(slide_id)
                    #    
                    #    if os.path.isfile(os.path.join(r_case_save_dir, heatmap_save_name)):
                    #        pass
                    #    else:
                    #        heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, heatmap_vis_args =heatmap_vis_args)
                    #        heatmap.save(os.path.join(r_case_save_dir, heatmap_save_name))
                    #        del heatmap
                    #else:
                        ##import pdb;pdb.set_trace()

                    heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
                                                                                                    int(heatmap_args.blur), 
                                                                                                    int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
                                                                                                    float(heatmap_args.alpha), int(heatmap_args.vis_level), 
                                                                                                    int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)



                    if heatmap_args.use_roi:
                        x1, x2 = slide_df.loc[slide_id, 'x1'], slide_entry.loc[slide_id, 'x2']
                        y1, y2 = slide_df.loc[slide_id, 'y1'], slide_entry.loc[slide_id, 'y2']
                        top_left = (int(x1), int(y1))
                        bot_right = (int(x2), int(y2))
                    else:
                        top_left = None
                        bot_right = None

                    patch_size = tuple([slide_patch_args['patch_size']]*2)
                    step_size = tuple([slide_patch_args['step_size']]*2)
                    step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))

                    wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
                                  'custom_downsample':slide_patch_args['custom_downsample'], 'level': slide_patch_args['patch_level'], 
                                  'use_center_shift': heatmap_args.use_center_shift}
                    
                    ### --> Seems like roi.h5 is continously being updated w/ attention scores + coordinates from 0.75 overlap!,
                    ### rois are then loaded from the save_path
                    save_path = os.path.join(r_case_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))
                    h5_all_path.append(save_path)
                    if heatmap_args.use_ref_scores:
                        ref_scores = scores
                    else:
                        ref_scores = None

                    pbar.set_description("%s: Generating HR Heatmap - %s" % (subject_id, slide_id[:24]))
                    if heatmap_args.calc_heatmap and not os.path.isfile(os.path.join(save_path)):
                        compute_from_patches(wsi_object=wsi_object, clam_pred=None, model=model, 
                            feature_extractor=feature_extractor, batch_size=exp_args.batch_size, **wsi_kwargs, 
                            attn_save_path=save_path,  ref_scores=ref_scores)

                    if not os.path.isfile(save_path):
                        if heatmap_args.use_roi:
                            save_path_full = os.path.join(r_case_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                            if os.path.isfile(save_path_full):
                                save_path = save_path_full
                            else:
                                continue
                        else:
                            continue


                    if os.path.isfile(os.path.join(p_case_save_dir, heatmap_save_name)):
                        continue
                    
                    file = h5py.File(save_path, 'r')
                    dset = file['attention_scores']
                    coord_dset = file['coords']

                    slide_scores = dset[:]
                    slide_coords = coord_dset[:]

                    file.close()


                    
                    heatmap_vis_args.update({'top_left':top_left, 'bot_right' : bot_right})
                    heatmap = drawHeatmap(slide_scores, slide_coords, slide_path, wsi_object=wsi_object,  
                                          annotation=None, heatmap_vis_args = heatmap_vis_args)#, 
                                          #binarize=heatmap_args.binarize, 
                                          #blank_canvas=heatmap_args.blank_canvas,
                                          #thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size, line_thickness=28, 
                                          #overlap=patch_args.overlap, 
                                          #top_left=top_left, bot_right = bot_right)
                    if heatmap_args.save_ext == 'jpg':
                        heatmap.save(os.path.join(p_case_save_dir, heatmap_save_name), quality=100)
                    else:
                        heatmap.save(os.path.join(p_case_save_dir, heatmap_save_name))
                    
                    if heatmap_args.save_orig:
                        #import pdb;pdb.set_trace
                        if heatmap_args.vis_level >= 0:
                            vis_level = heatmap_args.vis_level
                        else:
                            vis_params = def_vis_params.copy()
                            vis_params = load_params(slide_entry, vis_params)
                            vis_level = vis_params['vis_level']

                        heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
                        if os.path.isfile(os.path.join(p_case_save_dir, heatmap_save_name)):
                            pass
                        else:
                            # wsi_object = WholeSlideImage(slide_path)
                            heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
                            if heatmap_args.save_ext == 'jpg':
                                heatmap.save(os.path.join(p_case_save_dir, heatmap_save_name), quality=100)
                            else:
                                heatmap.save(os.path.join(p_case_save_dir, heatmap_save_name))


                counts, scores, coords = [], [], []
                #import pdb;pdb.set_trace()
                for save_path in h5_all_path:
                    file = h5py.File(save_path, 'r')
                    dset = file['attention_scores']
                    coord_dset = file['coords']

                    slide_scores = dset[:]
                    slide_coords = coord_dset[:]

                    counts.append(len(slide_coords))
                    coords.append(slide_coords)
                    scores.append(slide_scores)

                    file.close()

                scores = np.vstack(scores)
                coords = np.vstack(coords)
                counts = np.array(counts)
                cum_counts = np.append(np.array([0]), np.cumsum(counts))

                #SAMPLE TOPK MOST IMPORTANT PATCHES
                if sampling_mode:

                    patch_root = './patches/'
                    if model_args.pkl_path is not None:
                        rg_df = load_risk_df(model_args.pkl_path)
                        if subject_id not in rg_df.index:
                            #continue
                            risk_group = -1
                        else:
                            risk_group = rg_df.loc[subject_id]['strat']
                            if isinstance(risk_group, pd.Series):
                                risk_group = risk_group.iloc[0]

                        if risk_group not in [-1, 0, 1]:
                            continue

                        if risk_group == 1:
                            risk2str = 'IH'
                        elif risk_group == 0:
                            risk2str = 'IL'
                        else:
                            risk2str = 'unknown'
                    else:
                        risk2str = 'unknown'

                    patch_save_dir = os.path.join(patch_root, 'raw_patches', exp_args.save_exp_code, risk2str)
                    pkl_save_dir = os.path.join(patch_root, 'pkl_files', exp_args.save_exp_code, risk2str)
                    os.makedirs(patch_save_dir, exist_ok=True)
                    os.makedirs(pkl_save_dir, exist_ok=True)
                    bag_size = cum_counts[-1]

                    dynamic_k = round(bag_size*0.005)
                    if dynamic_k < 100:
                        dynamic_k = 200

                    samples = sample_args.samples
                    pbar.set_description("%s: Sampling Patches - %s" % (subject_id, slide_id[:23]))
                    if subject_id in os.listdir(patch_save_dir):
                        continue
                    os.makedirs(os.path.join(patch_save_dir,subject_id ,'high_attention'), exist_ok=True)
                    os.makedirs(os.path.join(patch_save_dir,subject_id ,'low_attention'), exist_ok=True)
                    for sample in samples:
                        if sample['sample']:
                            #import pdb;pdb.set_trace()
                            sample_results = sample_rois(scores, coords, k=dynamic_k, mode=sample['mode'], seed=sample['seed'], 
                                score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                            
                            patches = []
                            patches_sn = []
                            for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'] , sample_results['sampled_scores'])):
                                s_ids = sample_results['sampled_ids'][idx]
                                for which_slide in range(0,len(counts)):
                                    #print(which_slide)
                                    if  s_ids >= cum_counts[which_slide] and s_ids < cum_counts[which_slide + 1]:
                                        s_slide_ids = which_slide

                                slide_path = s_paths[s_slide_ids]
                                mask_file = m_paths[s_slide_ids]
                                slide_id = slide_ids[s_slide_ids]
                                slide_patch_args = slides_patch_args[slide_id]

                                wsi_object = initialize_wsi(slide_path, mask_file=mask_file)

                                if patch_args.custom_downsample == 1:
                                    patch = wsi_object.wsi.read_region(tuple(s_coord), slide_patch_args['patch_level'], 
                                        (slide_patch_args['patch_size'], slide_patch_args['patch_size'])).convert('RGB')
                                elif patch_args.custom_downsample == 2:
                                    patch = wsi_object.wsi.read_region(tuple(s_coord), slide_patch_args['patch_level'], 
                                        (slide_patch_args['patch_size']*2, slide_patch_args['patch_size']*2)).convert('RGB')

                                if sample['name'] == 'topk_high_attention':
                                    patch.save(os.path.join(patch_save_dir,subject_id,'high_attention','%s_%d_%d_%d.png' % (slide_id[:23], idx, dynamic_k, bag_size)))
                                if sample['name'] == 'topk_low_attention':
                                    patch.save(os.path.join(patch_save_dir,subject_id ,'low_attention','%s_%d_%d_%d.png' % (slide_id[:23], idx, dynamic_k, bag_size)))

                                patches.append(patch)
        #SAMPLE RADIOLOGY SLICES
        elif "radio" in model_args.mode:
            
            radio_h5_folder = data_args.data_h5_dir
            radio_dict = df[['subject_id']+model_args.modalities].set_index('subject_id').T.to_dict()
            modality_files = radio_dict[subject_id]
            
            if sampling_mode:

                radio_features,intersect_all = process_mri(radio_h5_folder, subject_id, modality_files, 
                    modalities = model_args.modalities)
                all_features = {i: r.to(device) for i, r in radio_features.items() }

                Y_hat, risk, A_radio  = infer_patient(model, all_features, bins, label)
                A_radio_softmax = np.exp(A_radio) / sum(np.exp(A_radio))
                radio_img_path = os.path.join(data_args.data_dir,subject_id)
                #radio_img_path = '/mnt/sda1/Lottie/histopathology_gbmlgg/PORPOISE-master/all_radio_data/'+subject_id+'/flair.nii.gz'
                img_arrs = radio_img(radio_img_path, modality_files)
                #for radio_m in radio_features:
                #    try:
                #        assert radio_features[radio_m].shape[0] == img_arrs[radio_m].shape[0]
                #    except:
                #        import pdb;pdb.set_trace()
                #img_arrs = {m:img_arr[intersect_all] for m , img_arr in img_arrs.items()}
                scores_dict = dict(zip(intersect_all,A_radio.flatten()))

                patch_root = './patches/'
                if model_args.pkl_path is not None:

                    rg_df = load_risk_df(model_args.pkl_path)
                    if subject_id not in rg_df.index:
                        #continue
                        risk_group = -1
                    else:
                        risk_group = rg_df.loc[subject_id]['strat']
                        if isinstance(risk_group, pd.Series):
                            risk_group = risk_group.iloc[0]

                    if risk_group not in [-1, 0, 1]:
                        continue

                    if risk_group == 1:
                        risk2str = 'IH'
                    elif risk_group == 0:
                        risk2str = 'IL'
                    else:
                        risk2str = 'unknown'
                else:
                    risk2str = 'unknown'

                patch_save_dir = os.path.join(patch_root, 'raw_patches', exp_args.save_exp_code, risk2str)
                pkl_save_dir = os.path.join(patch_root, 'pkl_files', exp_args.save_exp_code, risk2str)
                os.makedirs(patch_save_dir, exist_ok=True)
                os.makedirs(pkl_save_dir, exist_ok=True)



                samples = sample_args.samples
                pbar.set_description("%s: Sampling Patches" % (subject_id))

                high_patch_save_dir = os.path.join(patch_save_dir,subject_id ,'high_attention')
                low_patch_save_dir = os.path.join(patch_save_dir,subject_id ,'low_attention')

                os.makedirs(high_patch_save_dir, exist_ok=True)
                os.makedirs(low_patch_save_dir, exist_ok=True)

                #import pdb;pdb.set_trace()
                for sample in samples:
                    if sample['sample']:
                        dynamic_k = round(len(A_radio)*0.1)
                        if dynamic_k < 20:
                            dynamic_k = 20
                    temp_s = pd.DataFrame({'scores':A_radio.flatten(),'scores_softmax':A_radio_softmax.flatten(),
                        'ind':intersect_all})
                    temp_s.sort_values('scores',ascending = False,inplace = True)
                    temp_s.reset_index(drop = True, inplace = True)
                    temp_s.to_csv(os.path.join(patch_save_dir,subject_id,"scores.csv"))

                    for m, img_arr in img_arrs.items():
                        high_patch_save_dir_m = os.path.join(high_patch_save_dir,m)
                        low_patch_save_dir_m = os.path.join(low_patch_save_dir,m)
                        os.makedirs(high_patch_save_dir_m, exist_ok=True)
                        os.makedirs(low_patch_save_dir_m, exist_ok=True)
                        for i in range(dynamic_k):
                            top_10_ind = temp_s.loc[i,'ind']
                            fig, ax = plt.subplots( nrows=1, ncols=1 ) 
                            try:
                                plt.imsave(os.path.join(high_patch_save_dir_m,f'top_{str(i)}_axial_{str(top_10_ind)}.png'),
                                    img_arr[top_10_ind,:,:],cmap = 'gray')
                            except:
                                import pdb;pdb.set_trace()
                            plt.close(fig)
                        
                        for i in range(dynamic_k):
                            temp_s_reverse = temp_s.sort_values('scores').reset_index()
                            tail_10_ind = temp_s_reverse.loc[i,'ind']
                            fig, ax = plt.subplots( nrows=1, ncols=1 ) 
                            try:
                                plt.imsave(os.path.join(low_patch_save_dir_m,f'tail_{str(i)}_axial_{str(tail_10_ind)}.png'),
                                    img_arr[tail_10_ind,:,:],cmap = 'gray')
                            except:
                                import pdb;pdb.set_trace()
                            plt.close(fig)







