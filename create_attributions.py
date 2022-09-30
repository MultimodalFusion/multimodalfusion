import torch
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from utils.eval_utils import initiate_pretrained_model
from datasets.dataset_survival import Generic_Survival_Dataset, Generic_MIL_Survival_Dataset_Pretrained
from utils.utils import *


parser = argparse.ArgumentParser(description='Attribution script')
parser.add_argument('--model_path',  type=str, default=None)
args = parser.parse_args()

#LOAD MODEL ARGUMENTS
with open(os.path.join(args.model_path,f'experiment_{os.path.basename(args.model_path)}.txt'),'rb') as f:
    experiment_args_dict = eval(f.read())

args.data_root_dir = experiment_args_dict['data_root_dir']
args.csv_path = experiment_args_dict['csv_path']
args.split_dir = experiment_args_dict['split_dir']
args.modality = experiment_args_dict['radio_modality']
args.cancer_type = experiment_args_dict['cancer_type']
args.seed = experiment_args_dict['seed']
args.n_classes = experiment_args_dict['n_classes']
args.which_modality = experiment_args_dict['mode']
args.original_csv_file = experiment_args_dict['csv_path']
args.k = experiment_args_dict['num_splits']
args.mode = experiment_args_dict['mode']
args.task = experiment_args_dict['task']
args.bag_loss = experiment_args_dict['bag_loss']
args.train_type = experiment_args_dict['train_type']
args.n_layers = 1#experiment_args_dict['n_layers']


#output_dir = os.path.join(args.output_dir, args.cancer_type)

#CAPTUM FUNCTIONS
def interpret_patient_radio_path(x_radio,x_path):
    return model.captum_radio_path(x_radio,x_path)
def interpret_patient_path_omic(x_omic,x_path):
    return model.captum_path_omic(x_omic,x_path)
def interpret_patient_radio_omic(x_radio,x_omic):
    return model.captum_radio_omic(x_radio,x_omic)
def interpret_patient_radio_path_omic(x_radio, x_path, x_omic):
    return model.captum(x_radio, x_path, x_omic)

def boxplot_attr(attr,path):
    attr = attr.reset_index()
    attr_result_long = attr.melt(id_vars = 'subject_id',var_name = 'modalities',value_name = 'attribution_percentage')
    fig,ax = plt.subplots(1,1)
    sns.boxplot(data = attr_result_long, x= 'modalities', y ='attribution_percentage', ax= ax)
    sns.swarmplot(data = attr_result_long, x= 'modalities', y ='attribution_percentage', ax= ax, color = 'black')
    ax.set_ylabel('attribution percentage')
    plt.savefig(os.path.join(path,'attribution_plot.png'))

dataset = Generic_MIL_Survival_Dataset_Pretrained(csv_path =  args.csv_path,
                            data_dir = args.data_root_dir,
                            mode = args.mode,
                            modalities = args.modality,
                            n_bins = args.n_classes,
                            label_col = 'survival_months',
                            k = args.k,
                            split = None,
                            split_dir = args.split_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            ignore = [])

if __name__ == "__main__":
    all_attr = []
    all_attr_orig = []
    for split in range(args.k):
        print('-------------- split ',split,'----------------')
        train_dataset, val_dataset = dataset.return_train_val_splits(from_id=False, 
                  csv_path=os.path.join(args.split_dir, f'splits_{split}.csv'))
        val_loader = get_pretrained_split_loader(val_dataset, batch_size=1)
        ckpt_path = os.path.join(args.model_path, f's_{split}_minloss_checkpoint.pt')
        model = initiate_pretrained_model(args, ckpt_path)
        subject_count = 0
        for batch_idx, (radio_features ,path_features, genomic_features, label, event_time, c,_) in enumerate(val_loader):
            subject_id = val_loader.dataset.slides_radio_data['subject_id'][subject_count:(subject_count+len(label))]
            subject_count += len(label)
            radio_features= radio_features
            path_features =path_features
            genomic_features = genomic_features

            if 'radio' in args.mode and 'path' in args.mode and 'omic' in args.mode:
                ig = IntegratedGradients(interpret_patient_radio_path_omic)
                ig_attr = ig.attribute((radio_features,path_features,genomic_features), n_steps=20)
                ig_attr_radio = ig_attr[0].detach().cpu().data.numpy()
                ig_attr_wsi = ig_attr[1].detach().cpu().data.numpy()
                ig_attr_omic = ig_attr[2].detach().cpu().data.numpy()
                
                radio_attr = np.sum(np.abs(ig_attr_radio),axis = 1).item()
                wsi_attr = np.sum(np.abs(ig_attr_wsi),axis = 1).item()
                omic_attr = np.sum(np.abs(ig_attr_omic),axis = 1).item()

                all_attr.append(pd.DataFrame({'subject_id':subject_id,'radio_attr':radio_attr,
                    'path_attr': wsi_attr, 'omic_attr': omic_attr}))

                radio_attr_orig = np.sum(ig_attr_radio,axis = 1).item()
                wsi_attr_orig = np.sum(ig_attr_wsi,axis = 1).item()
                omic_attr_orig = np.sum(ig_attr_omic,axis = 1).item()

                all_attr_orig.append(pd.DataFrame({'subject_id':subject_id,'radio_attr':radio_attr_orig,
                    'path_attr': wsi_attr_orig, 'omic_attr': omic_attr_orig}))

            elif 'radio' in args.mode and 'path' in args.mode and 'omic' not in args.mode:
                ig = IntegratedGradients(interpret_patient_radio_path)
                ig_attr = ig.attribute((radio_features,path_features), n_steps=20)
                ig_attr_radio = ig_attr[0].detach().cpu().data.numpy()
                ig_attr_wsi = ig_attr[1].detach().cpu().data.numpy()
                
                radio_attr = np.sum(np.abs(ig_attr_radio),axis = 1).item()
                wsi_attr = np.sum(np.abs(ig_attr_wsi),axis = 1).item()
                all_attr.append(pd.DataFrame({'subject_id':subject_id,'radio_attr':radio_attr,
                    'path_attr': wsi_attr}))

                radio_attr_orig = np.sum(ig_attr_radio,axis = 1).item()
                wsi_attr_orig = np.sum(ig_attr_wsi,axis = 1).item()

                all_attr_orig.append(pd.DataFrame({'subject_id':subject_id,'radio_attr':radio_attr_orig,
                    'path_attr': wsi_attr_orig}))

            if 'radio' in args.mode and 'path' not in args.mode and 'omic' in args.mode:
                ig = IntegratedGradients(interpret_patient_radio_omic)
                #import pdb;pdb.set_trace()
                ig_attr = ig.attribute((radio_features,genomic_features), n_steps=20)
                ig_attr_radio = ig_attr[0].detach().cpu().data.numpy()
                ig_attr_omic = ig_attr[1].detach().cpu().data.numpy()
                
                radio_attr = np.sum(np.abs(ig_attr_radio),axis = 1).item()
                omic_attr = np.sum(np.abs(ig_attr_omic),axis = 1).item()
                all_attr.append(pd.DataFrame({'subject_id':subject_id,'radio_attr':radio_attr,
                    'omic_attr': omic_attr}))

                radio_attr_orig = np.sum(ig_attr_radio,axis = 1).item()
                omic_attr_orig = np.sum(ig_attr_omic,axis = 1).item()

                all_attr_orig.append(pd.DataFrame({'subject_id':subject_id,'radio_attr':radio_attr_orig,
                     'omic_attr': omic_attr_orig}))

            if 'radio' not in args.mode and 'path' in args.mode and 'omic' in args.mode:
                ig = IntegratedGradients(interpret_patient_path_omic)
                ig_attr = ig.attribute((genomic_features ,path_features), n_steps=20)
                ig_attr_omic = ig_attr[0].detach().cpu().data.numpy()
                ig_attr_wsi = ig_attr[1].detach().cpu().data.numpy()
                
                wsi_attr = np.sum(np.abs(ig_attr_wsi),axis = 1).item()
                omic_attr = np.sum(np.abs(ig_attr_omic),axis = 1).item()
                all_attr.append(pd.DataFrame({'subject_id':subject_id,
                    'path_attr': wsi_attr, 'omic_attr': omic_attr}))

                wsi_attr_orig = np.sum(ig_attr_wsi,axis = 1).item()
                omic_attr_orig = np.sum(ig_attr_omic,axis = 1).item()

                all_attr_orig.append(pd.DataFrame({'subject_id':subject_id,
                    'path_attr': wsi_attr_orig, 'omic_attr': omic_attr_orig}))
    #import pdb;pdb.set_trace()
    all_attr = pd.concat(all_attr)
    all_attr = pd.DataFrame(all_attr.groupby('subject_id').mean())
    #.set_index('subject_id')
    save_path = os.path.join('./attributions', args.cancer_type, os.path.basename(args.split_dir), os.path.basename(args.model_path))
    os.makedirs(save_path, exist_ok = True)
    all_attr.to_csv(os.path.join(save_path, 'attr.csv'))

    all_attr_orig = pd.concat(all_attr_orig)
    all_attr_orig = pd.DataFrame(all_attr_orig.groupby('subject_id').mean())
    all_attr_orig.to_csv(os.path.join(save_path, 'attr_orig.csv'))

    #attr_percent = all_attr.div(all_attr.sum(axis = 1),axis = 0)
    #boxplot_attr(attr_percent,save_path)

    

