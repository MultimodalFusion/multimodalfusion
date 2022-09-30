import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms


import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import h5py
import time
import argparse
import pickle


from utils.file_utils import save_hdf5
from utils.utils import collate_radio_features
from models.resnet_custom import resnet50_baseline
from datasets.dataset_raw import PreprocessDataset




def compute_features(file_path, output_path, model, transform_f, 
    #small = False,stride = 60,
    batch_size = 8,verbose = 0, print_every=4, transform = True, planes = ['axial','sagittal','coronal'],
    #consecutive = False, consec_gap = 1, 
    cancer_type = 'glioma',segment = False):
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}

    mode = 'w'
    
    for plane in planes:
        print(f'Processing {plane} plane')
        loader = DataLoader(dataset = PreprocessDataset(file_path =file_path, 
            transform = True, transform_f = transform_f, plane = plane, 
            cancer_type = cancer_type,segment = segment), 
        batch_size = batch_size, shuffle = False, collate_fn=collate_radio_features, **kwargs)
        if verbose > 0:
            print('processing {}: total of {} batches'.format(file_path,len(loader)))

        

        for count,(batch,slice_ids) in enumerate(loader):
            if batch is not None:
                with torch.no_grad():
                    if count % print_every == 0:
                        print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))

                    batch = batch.to(device, non_blocking = True)
                    #import pdb;pdb.set_trace()
                    features = model(batch)
                    features = features.cpu().numpy()
                    slice_ids = slice_ids#.numpy()

                    asset_dict = {'features': features, 'slice_index': slice_ids}

                    #print(features)
                    save_hdf5(output_path, asset_dict, mode=mode)
                    mode = 'a'
    return output_path

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--radio_dir', type=str, default=None)
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--planes',type = str, default = 'axial,sagittal,coronal')
parser.add_argument('--cancer_type', type=str, default='glioma')
parser.add_argument('--segment', action='store_true', default=False, help='Segment Lung?')

args = parser.parse_args()


if __name__ == '__main__':
    #read dataset
    dataset_path = args.csv_path
    radio_dir = args.radio_dir
    output_dir = args.output_dir

    args.planes = [p for p in args.planes.split(',')]

    #if args.small:
    #    output_dir = output_dir + '_small'

    #make new directory
    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(os.path.join(output_dir, 'radio_pt_files'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'radio_h5_files'), exist_ok=True)

    #model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = resnet50_baseline(pretrained=True)
    model = model.to(device)
    model.eval()


    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)


    transform_f = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])


    datasets = pd.read_csv(dataset_path)
    if args.cancer_type == 'glioma':
        subjects_modalities = datasets[['subject_id', 'FLAIR', 'T1', 'T1Gd', 'T2']].dropna().set_index('subject_id').T.to_dict()
        modalities = ['FLAIR', 'T1', 'T1Gd', 'T2']

        for i in modalities:
            os.makedirs(os.path.join(output_dir, args.cancer_type, 'radio_pt_files',i), exist_ok = True)
            os.makedirs(os.path.join(output_dir, args.cancer_type, 'radio_h5_files',i), exist_ok = True)


        #total = len(subjects_modalities)
        for subject, subject_m in subjects_modalities.items():
            subject_m = {k:v for k, v, in subject_m.items() if not pd.isna(v)}

            #print('\nprogress: {}/{}'.format(ind, total))
            print(subject)

            datasets[datasets['subject_id']== subject][modalities]
            
            for modality, modality_file_name in subject_m.items():
                print(modality)
                
                output_h5_path = os.path.join(output_dir, args.cancer_type, 'radio_h5_files', modality, subject+'.h5')
                output_pt_path = os.path.join(output_dir, args.cancer_type, 'radio_pt_files', modality, subject+'.pt')
                
                input_path = os.path.join(radio_dir,subject,modality_file_name)
                time_start = time.time()
                output_file_path = compute_features(input_path, output_h5_path, model, #small = args.small, 
                    #stride = args.stride, 
                    transform_f = transform_f, 
                    batch_size = args.batch_size, transform = True,
                    planes = args.planes,
                    #consecutive = args.consecutive,
                    #consec_gap = args.consec_gap,
                    cancer_type = args.cancer_type)
                time_elapsed = time.time() - time_start
                print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
                
                file = h5py.File(output_file_path, "r")

                features = file['features'][:]
                print('features size: ', features.shape)

                print('coordinates: ', file['slice_index'].shape)
                features = torch.from_numpy(features)
                torch.save(features, output_pt_path)




    elif args.cancer_type == 'lung':
        subjects_modalities = datasets[['subject_id', 'CT']].dropna().set_index('subject_id').T.to_dict()
        
        modality = 'CT'
        os.makedirs(os.path.join(output_dir, args.cancer_type, 'radio_pt_files',modality), exist_ok = True)
        os.makedirs(os.path.join(output_dir, args.cancer_type, 'radio_h5_files',modality), exist_ok = True)

        sub_total = 0
        total = len(subjects_modalities)
        not_processed_subject = []
        for subject, subject_m in subjects_modalities.items():
            sub_total +=1
            print('\nprogress: {}/{}'.format(sub_total, total))

            try:
                subject_m = {k:v for k, v, in subject_m.items() if not pd.isna(v)}

                print(subject,modality)
                subject = str(subject)

                output_h5_path = os.path.join(output_dir, args.cancer_type ,  'radio_h5_files', modality, subject+'.h5')
                output_pt_path = os.path.join(output_dir, args.cancer_type, 'radio_pt_files', modality, subject+'.pt')

                if os.path.isfile(output_pt_path):
                    print('skipped')
                    continue
                
                input_path = os.path.join(radio_dir,subject,subject_m['CT'])

                time_start = time.time()
                output_file_path = compute_features(input_path, output_h5_path, model, 
                    transform_f = transform_f, 
                    batch_size = args.batch_size, transform = True,
                    planes = args.planes,
                    cancer_type = args.cancer_type,
                    segment = args.segment)
                time_elapsed = time.time() - time_start
                print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
                
                file = h5py.File(output_file_path, "r")

                features = file['features'][:]
                print('features size: ', features.shape)

                print('coordinates: ', file['slice_index'].shape)
                features = torch.from_numpy(features)
                torch.save(features, output_pt_path)
            except Exception as e:
                print(e)
                print('Fails to process; save subject')
                not_processed_subject.append(subject)

        with open(os.path.join(output_dir, "not_processed.pkl"), "wb") as fp:
            pickle.dump(not_processed_subject, fp)

