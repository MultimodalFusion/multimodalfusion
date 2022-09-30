from __future__ import print_function

import argparse
import pdb
import os
import math
import sys

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from timeit import default_timer as timer

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_survival import Generic_Survival_Dataset, Generic_MIL_Survival_Dataset


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    val_cindex = []
    test_cindex = []
    folds = np.arange(start, end)

    for i in folds:
      start = timer()
      seed_torch(args.seed)

      if args.split_mode == 'train_val':
        train_dataset, val_dataset = dataset.return_train_val_splits(from_id=False, 
          csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        args.omic_input_dim = train_dataset.genomic_features.shape[1]
        print("Genomic Dimension", args.omic_input_dim)
        datasets = (train_dataset, val_dataset)
        val_df, val_c = train(datasets, i, args)
        val_cindex.append(val_c)

      elif args.split_mode == 'train_val_test':
        train_dataset, val_dataset, test_dataset = dataset.return_train_val_test_splits(from_id=False, 
          csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        print('training: {}, validation: {}, test: {}'.format(len(train_dataset), len(val_dataset),len(test_dataset)))
        if 'omic' in args.mode:
          args.omic_input_dim = train_dataset.genomic_features.shape[1]
          print("Genomic Dimension", args.omic_input_dim)
        datasets = (train_dataset, val_dataset, test_dataset)
        val_df, val_c, test_df, test_c = train(datasets, i, args)
        val_cindex.append(val_c)
        test_cindex.append(test_c)

      else:
        raise NotImplementedError

      #write results to pkl
      save_pkl(os.path.join(args.results_dir, 'split_train_val_{}_results.pkl'.format(i)), val_df)

      if args.split_mode == 'train_val_test':
        save_pkl(os.path.join(args.results_dir, 'split_train_test_{}_results.pkl'.format(i)), test_df)
      end = timer()
      print('Fold %d Time: %f seconds' % (i, end - start))

    print(f'Average validation c_index: {np.mean(val_cindex)}')
    if args.split_mode == 'train_val_test':
      print(f'Average test c_index: {np.mean(test_cindex)}')
    
    if len(folds) != args.k: save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else: save_name = 'summary.csv'

    if args.split_mode == 'train_val':
      results_df = pd.DataFrame({'folds': folds, 'val_cindex': val_cindex})
    elif args.split_mode == 'train_val_test':
      results_df = pd.DataFrame({'folds': folds, 'val_cindex': val_cindex, 'test_cindex': test_cindex})
    results_df.to_csv(os.path.join(args.results_dir, 'summary.csv'))

# Training settings
parser = argparse.ArgumentParser(description='Configurations for MMF Training')
parser.add_argument('--data_root_dir', type=str, default='./features', help='data directory')
parser.add_argument('--which_splits', type=str, default='10foldcv', help='Path to splits directory.')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--mode', type=str, default='radio')
parser.add_argument('--modality', type = str, default = 'T1,T2,T1Gd,FLAIR')
parser.add_argument('--task', type=str, default='survival', help='Which csv.')
parser.add_argument('--cancer_type', choices=['brain','lung'], type=str, default='brain', help='which cancer type')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--split',type = str, default = None)

#MODEL
parser.add_argument('--model_type', type=str, default=None, help='type of model (radio_attention_mil | path_attention_mil | max_net | mm_attention_mil)')
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--split_mode', type=str, choices = ['train_val', 'train_val_test'], default = 'train_val',help='train_val/train_val_test')
parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight', type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None, help='instance-level clustering loss function (default: None)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv','ranking_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None','all','omic_mm'], default='None', help='Reg Type (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='Regularization Strength')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size_wsi', type=str, default='small', help='Size of wsi AMIL model.')
parser.add_argument('--model_size_radio', type=str, default='small', help='Size of radio AMIL model.')
parser.add_argument('--model_size_omic', type=str, default='small', help='Size of SNN Model.')
parser.add_argument('--gc', type=int, default=1, help='gradient accumulation step')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
parser.add_argument('--gate_path', action='store_true', default=False, help='Enable feature gating in MMF layer.')
parser.add_argument('--gate_omic', action='store_true', default=False, help='Enable feature gating in MMF layer.')
parser.add_argument('--gate_radio', action='store_true', default=False, help='Enable feature gating in MMF layer.')
parser.add_argument('--fusion', type=str, default=None, help='Which fusion mechanism to use.')
parser.add_argument('--radio_fusion', type=str, default = None, help='Which fusion mechanism to use.')
parser.add_argument('--radio_mil_type',type = str, default = None)

parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--overwrite', action='store_true', default=False, help='Current experiment results already exists. Overwrite?')
parser.add_argument('--apply_mad', action='store_true', default=True, help='Use genes with median absolute deviation.')
parser.add_argument('--test',type = str, default = '')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#specific cancer type folder
dataset_path = os.path.join('dataset_csv',args.cancer_type)
args.results_dir = os.path.join(args.results_dir,args.cancer_type)
args.split_dir = os.path.join('./splits',args.cancer_type,args.which_splits)
args.data_root_dir = os.path.join(args.data_root_dir,args.cancer_type)

args.modality = [m for m in args.modality.split(',')]

#experiment name
param_code = ''
if args.model_type == 'path_attention_mil':
  param_code += 'PATH'
elif args.model_type == 'radio_attention_mil':
  param_code += 'RADIO'
elif args.model_type == 'max_net':
  param_code += 'OMICS'
elif args.model_type == 'mm_attention_mil':
  param_code += 'MMF'
  if 'radio' in args.mode:
    param_code += '_RADIO'
  if 'path' in args.mode:
    param_code += '_PATH'
  if 'omic' in args.mode:
    param_code += '_OMICS'
    
else:
  raise NotImplementedError

param_code += '_a%s' % str(args.alpha_surv)

if args.lr != 2e-4:
  param_code += '_lr%s' % format(args.lr, '.0e')

if args.reg_type != 'None':
  param_code += '_reg%s' % format(args.lambda_reg, '.0e')

if args.gc != 1:
  param_code += '_gc%s' % str(args.gc)

param_code += '_s%s' % str(args.seed)

if args.test != '':
  param_code += f'_{args.test}'

args.exp_code = param_code

print("Experiment Name:", param_code)

#set seed
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)


#encoding_size = 1024
settings = {'data_root_dir':args.data_root_dir,
            'csv_path':os.path.join(dataset_path,f'{args.task}.csv'),
            'split_dir': os.path.join('./splits',args.cancer_type ,args.which_splits),
            'cancer_type':args.cancer_type,
            'mode':args.mode,
            'num_splits': args.k, 
            'n_classes':args.n_classes,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'inst_loss': args.inst_loss,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            'model_size_radio': args.model_size_radio,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt,
            'fusion':args.fusion,
            'radio_fusion':args.radio_fusion,
            'radio_mil_type':args.radio_mil_type,
            'radio_modality': args.modality}
print('\nLoad Dataset')

dataset = Generic_MIL_Survival_Dataset(csv_path =  os.path.join(dataset_path,f'{args.task}.csv'),
                            data_dir = args.data_root_dir,
                            mode = args.mode,
                            modalities = args.modality,
                            n_bins = args.n_classes,
                            label_col = 'survival_months',
                            k = args.k,
                            split = args.split,
                            split_dir = args.split_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            ignore = []
                            )
args.bins = dataset.bins

#create result dir
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, args.which_splits, param_code)
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
  print("Exp Code <%s> already exists! Exiting script. set overwrite or rename using test" % args.exp_code)
  sys.exit()


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":

    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))


