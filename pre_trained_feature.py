from __future__ import print_function
import torch

import argparse
import pdb
import os
import math
import sys

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from utils.utils import *
from datasets.dataset_survival import Generic_Survival_Dataset, Generic_MIL_Survival_Dataset
from models.model_genomic import MaxNet
from models.model_attention_mil_path import MIL_Attention_fc_path, MIL_Attention_fc_surv_path
from models.model_attention_mil_radio import MIL_Attention_fc_radio, MIL_Attention_fc_surv_radio



parser = argparse.ArgumentParser(description='Pre-trained Unimodal Model Feature Extraction')
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='pretrained_feature')
parser.add_argument('--which_k', type=int, default=None, help='n-fold-cv')
parser.add_argument('--extraction_csv_path', type=str, default=None, help='what subjects')

args = parser.parse_args()

with open( os.path.join(args.checkpoint_path,f'experiment_{os.path.basename(args.checkpoint_path)}.txt'),'rb') as f:
    experiment_args_dict = eval(f.read())

args.data_root_dir = experiment_args_dict['data_root_dir']
args.csv_path = experiment_args_dict['csv_path']
args.split_dir = experiment_args_dict['split_dir']
args.radio_modality = experiment_args_dict['radio_modality']
args.cancer_type = experiment_args_dict['cancer_type']
args.seed = experiment_args_dict['seed']
args.n_classes = experiment_args_dict['n_classes']
args.bag_loss = experiment_args_dict['bag_loss']
args.which_modality = experiment_args_dict['mode']
args.original_csv_file = experiment_args_dict['csv_path']
output_dir = os.path.join(args.output_dir, args.cancer_type)

dataset = Generic_MIL_Survival_Dataset(csv_path =  args.csv_path,
                            data_dir = args.data_root_dir,
                            mode = args.which_modality,
                            modalities = args.radio_modality,
                            n_bins = args.n_classes,
                            label_col = 'survival_months',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            ignore = []
                            )
whole_dataset = dataset.return_whole_splits(csv_file = os.path.join(args.split_dir,f'splits_{str(args.which_k)}.csv'))
collate_function = collate_MIL_survival(args.radio_modality)
loader = DataLoader(whole_dataset, batch_size=1, sampler = SequentialSampler(whole_dataset), collate_fn = collate_function,shuffle=False)


if __name__ == '__main__':
    #read dataset
    csv_path = args.csv_path
    path = True if args.which_modality == 'path' else False
    radio =  True if args.which_modality == 'radio' else False
    omic = True if args.which_modality == 'omic' else False

    #make new directory
    os.makedirs(output_dir, exist_ok = True)

    print(f'Load checkpoint from {args.which_k} fold result')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    extraction_subject_ids = np.unique(pd.read_csv(args.extraction_csv_path).subject_id)

    if radio:
        os.makedirs(os.path.join(output_dir,  'radio_pt_files' ), exist_ok=True)
        print('Initiate radio model ...')
        model_dict = {"dropout": True, 'n_classes': args.n_classes,  'modalities': args.radio_modality,
            'n_classes' : args.n_classes,'gate_radio' :True}
        radio_model = MIL_Attention_fc_surv_radio(**model_dict)
        print(radio_model)
        radio_model = radio_model.to(device)
        print('Load pretrained weights ...')
        radio_ckpt = torch.load(os.path.join(args.checkpoint_path, f"s_{str(args.which_k)}_minloss_checkpoint.pt"))
        #import pdb;pdb.set_trace()
        radio_model.load_state_dict(radio_ckpt)
        #for name, param in radio_model.named_parameters():
        #    print(name, param.data)
        radio_model.eval()

    if path:
        os.makedirs(os.path.join(output_dir,  'path_pt_files' ), exist_ok=True)
        print('Initiate path model ...')
        model_dict = {"dropout": True, 'n_classes': args.n_classes}
        path_model = MIL_Attention_fc_surv_path(**model_dict)
        path_model = path_model.to(device)
        print('Load pretrained weights ...')
        path_ckpt = torch.load(os.path.join(args.checkpoint_path, f"s_{str(args.which_k)}_minloss_checkpoint.pt"))
        path_model.load_state_dict(path_ckpt)
        path_model.eval()

    if omic:
        os.makedirs(os.path.join(output_dir,  'omic_pt_files' ), exist_ok=True)
        print('Initiate omic model ...')
        model_dict = {'input_dim': whole_dataset.genomic_features.shape[1], 
            'model_size_omic': 'small', 'n_classes': args.n_classes,'bag_loss':args.bag_loss}
        omic_model = MaxNet(**model_dict)
        omic_model = omic_model.to(device)
        print('Load pretrained weights ...')
        omic_ckpt = torch.load(os.path.join(args.checkpoint_path, f"s_{str(args.which_k)}_minloss_checkpoint.pt"))
        omic_model.load_state_dict(omic_ckpt)
        omic_model.eval()


    for batch_idx, (radio_features ,path_features, genomic_features, label, event_time, c) in enumerate(loader):
        subject_id= dataset.slides_radio_data.subject_id[batch_idx]
        print(f'-----pretrained feature extraction for {subject_id} ------')
        if subject_id not in extraction_subject_ids:
            print('skipped')
            continue
        if path and not torch.equal(path_features,torch.zeros((1,1))):
            path_features = path_features.to(device)
            output_pt_path = os.path.join(output_dir, 'path_pt_files', f'{subject_id}.pt')
            if not os.path.isfile(output_pt_path):
                print('Extracting pre-trained features for pathology features of size', path_features.shape)
                with torch.no_grad():
                    path_f_extracted = path_model(path_features=path_features, return_features = True ) 
                path_f_extracted = path_f_extracted.detach().cpu()
                print(f'Saving pre-trained features to {output_pt_path}')
                torch.save(path_f_extracted, output_pt_path)
            else:
                print('skipped')

        if radio and not all([torch.equal(r,torch.zeros((1,1))) for i, r in radio_features.items()]):
            all_args = {i: r.to(device) for i, r in radio_features.items() }
            all_args['return_features'] = True
            output_pt_path = os.path.join(output_dir, 'radio_pt_files', f'{subject_id}.pt')
            if not os.path.isfile(output_pt_path):
                print('Extracting pre-trained features for radiology features of size', list(radio_features.values())[0].size(0))
                with torch.no_grad():
                    radio_f_extracted = radio_model(**all_args) 
                radio_f_extracted = radio_f_extracted.detach().cpu()

                print(f'Saving pre-trained features to {output_pt_path}')
                torch.save(radio_f_extracted, output_pt_path)
            else:
                print('skipped')


        if omic and not torch.equal(genomic_features.float(),torch.zeros((1,1))):
            genomic_features = genomic_features.to(device)
            output_pt_path = os.path.join(output_dir, 'omic_pt_files', f'{subject_id}.pt')
            if not os.path.isfile(output_pt_path):
                print('Extracting pre-trained features for omics features of size', genomic_features.shape)
                with torch.no_grad():
                    omic_f_extracted = omic_model(genomic_features =genomic_features.float(),return_features = True ) 
                omic_f_extracted = omic_f_extracted.detach().cpu()
                print(f'Saving pre-trained features to {output_pt_path}')
                torch.save(omic_f_extracted, output_pt_path)
            else:
                print('skipped')

