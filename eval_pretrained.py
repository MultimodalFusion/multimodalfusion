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
from utils.core_utils_pretrained import eval_model
from datasets.dataset_survival import Generic_Survival_Dataset, Generic_MIL_Survival_Dataset_Pretrained


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    val_cindex = []
    test_cindex = []
    val_ibs = []
    test_ibs = []
    folds = np.arange(args.k)

    for i in folds:
      start = timer()
      seed_torch(args.seed)

      if args.split_mode == 'train_val':
        train_dataset, val_dataset = dataset.return_train_val_splits(from_id=False, 
          csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)
        val_df, val_c, val_i = eval_model(datasets, i, args)
        val_cindex.append(val_c)
        val_ibs.append(val_i)

      elif args.split_mode == 'train_val_test':
        train_dataset, val_dataset, test_dataset = dataset.return_train_val_test_splits(from_id=False, 
          csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        print('training: {}, validation: {}, test: {}'.format(len(train_dataset), len(val_dataset),len(test_dataset)))
        #if 'omic' in args.mode:
        #  args.omic_input_dim = train_dataset.genomic_features.shape[1]
        #  print("Genomic Dimension", args.omic_input_dim)
        datasets = (train_dataset, val_dataset, test_dataset)
        val_df, val_c, val_i, test_df, test_c, test_i = eval_model(datasets, i, args)
        val_cindex.append(val_c)
        test_cindex.append(test_c)

        val_ibs.append(val_i)
        test_ibs.append(test_i)

      else:
        raise NotImplementedError

      #write results to pkl
      save_pkl(os.path.join(args.results_dir, 'split_train_val_{}_results.pkl'.format(i)), val_df)

      if args.split_mode == 'train_val_test':
        save_pkl(os.path.join(args.results_dir, 'split_train_test_{}_results.pkl'.format(i)), test_df)
      end = timer()
      print('Fold %d Time: %f seconds' % (i, end - start))

    print(f'Average validation c_index: {np.mean(val_cindex)}')
    print(f'Average validation ibs: {np.mean(val_ibs)}')

    if args.split_mode == 'train_val_test':
      print(f'Average test c_index: {np.mean(test_cindex)}')
      print(f'Average test ibs: {np.mean(test_ibs)}')

    
    if len(folds) != args.k: save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else: save_name = 'summary.csv'

    if args.split_mode == 'train_val':
      results_df = pd.DataFrame({'folds': folds, 'val_cindex': val_cindex, 'val_ibs': val_ibs})
    elif args.split_mode == 'train_val_test':
      results_df = pd.DataFrame({'folds': folds, 'val_cindex': val_cindex, 'test_cindex': test_cindex,
         'val_ibs': val_ibs, 'test_ibs': test_ibs})
    results_df.to_csv(os.path.join(args.results_dir, 'summary.csv'))

# Training settings
parser = argparse.ArgumentParser(description='Configurations for MMF Evaluating')
parser.add_argument('--which_splits', type=str, default='10foldcv', help='Path to splits directory.')
parser.add_argument('--model_path', type=str, default=None, help='Path to model.')
parser.add_argument('--split_mode', type=str, default = 'train_val_test',help='train_val/train_val_test')
parser.add_argument('--batch_size', type=int, default = 32, help='Batch Size')
parser.add_argument('--overwrite', action='store_true', default=False, help='Current experiment results already exists. Overwrite?')
args = parser.parse_args()

with open( os.path.join(args.model_path,f'experiment_{os.path.basename(args.model_path)}.txt'),'rb') as f:
    experiment_args_dict = eval(f.read())

args.data_root_dir = experiment_args_dict['data_root_dir']
args.csv_path = experiment_args_dict['csv_path']
args.cancer_type = experiment_args_dict['cancer_type']
args.seed = experiment_args_dict['seed']
args.n_classes = experiment_args_dict['n_classes']
args.bag_loss = experiment_args_dict['bag_loss']
args.mode = experiment_args_dict['mode']
args.k = experiment_args_dict['num_splits']
args.task = experiment_args_dict['task']
args.model_type = experiment_args_dict['model_type']
args.train_type = experiment_args_dict['train_type']
args.n_layers = experiment_args_dict['n_layers']
args.split_dir = os.path.join('./splits',args.cancer_type,args.which_splits)
args.modalities = experiment_args_dict['radio_modality']
args.results_dir = os.path.join(experiment_args_dict['results_dir'], args.which_splits, experiment_args_dict['exp_code'])

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

print('\nLoad Dataset')
dataset = Generic_MIL_Survival_Dataset_Pretrained(csv_path = args.csv_path,
                            data_dir = args.data_root_dir,
                            mode = args.mode,
                            modalities = args.modalities,
                            n_bins = args.n_classes,
                            label_col = 'survival_months',
                            k = args.k,
                            split = None,
                            split_dir = args.split_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            ignore = [])
args.bins = dataset.bins

### GET RID OF WHICH_SPLITS IF U WANT TO MAKE THE RESULTS FOLDER LESS CLUTTERRED
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
  print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
  sys.exit()
  

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))


