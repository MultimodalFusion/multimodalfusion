import os
import torch
import numpy as np
import pandas as pd
import h5py
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,LeaveOneOut
from collections import Counter

from torch.utils.data import Dataset

class Generic_Survival_Dataset(Dataset):
    def __init__(self, csv_path, mode = 'radio', modalities = ['T1','T2','T1Gd','FLAIR'], 
        shuffle = False, seed = 7,  print_info = True, n_bins = 5, ignore=[], label_col = 'survival_months',
        filter_dict = {}, eps=1e-6, k = 5, split = None, split_dir = 'splits/5foldcv'):
        
        self.seed = seed
        self.print_info = print_info
        self.label_col = label_col
        self.mode = mode
        self.modalities = modalities
        self.data_dir = None
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.k = k
        self.split_dir = split_dir
        self.split = split

    
        slides_radio_data = pd.read_csv(csv_path, low_memory=False)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slides_radio_data)
           
        #assign labels
        patients_df = slides_radio_data.drop_duplicates(['subject_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]
        disc_labels, q_bins = pd.qcut(uncensored_df[uncensored_df.train == 1][label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slides_radio_data[label_col].max() + eps
        q_bins[0] = slides_radio_data[label_col].min() - eps
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(1, 'disc_label', disc_labels.values.astype(int))


        slides_dict = {}
        slides_radio_data = slides_radio_data.set_index('subject_id')
        for patient in patients_df['subject_id']:
            slide_ids = slides_radio_data.loc[patient, 'slide_id']
            
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            elif isinstance(slide_ids, float):
                continue
            else:
                slide_ids = slide_ids.values

            slides_dict.update({patient:slide_ids})
        self.slides_dict = slides_dict
        #import pdb;pdb.set_trace()

        slides_radio_data = patients_df
        slides_radio_data.reset_index(drop = True, inplace = True)


        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:                
                label_dict.update({(i, c):key_count})
                key_count+=1         
        self.label_dict = label_dict
        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        slides_radio_data.insert(1,'label',[label_dict[i] for i in list(zip(slides_radio_data.disc_label,slides_radio_data.censorship))])
        
        #patient-modalities
        patient_data = {'subject_id':slides_radio_data.subject_id, 'label': slides_radio_data.label}
        self.patient_data = patient_data

        radio_dict = slides_radio_data[['subject_id']+ self.modalities].set_index('subject_id').T.to_dict()
        self.radio_dict = radio_dict
    
        self.slides_radio_data = slides_radio_data
        self.metadata = ['subject_id','label', 'disc_label', 'slide_id'] + self.modalities + ['oncotree_code','is_female','age','survival_months','censorship','train']
        self.patient_cls_ids = [np.where(self.patient_data['label'] == i)[0] for i in range(self.num_classes)]
        self.slide_cls_ids = [np.where(self.slides_radio_data['label'] == i)[0] for i in range(self.num_classes)]
        
        if print_info:
            self.summarize()

        if split is not None:
            self.do_split()


    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))


    def get_list(self, ids):
        return self.slides_radio_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slides_radio_data['label'][ids]
    

    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slides_radio_data['subject_id'].isin(split.tolist())
            df_slice = self.slides_radio_data[mask].reset_index(drop=True)
            split = Split(df_slice, modalities = self.modalities, mode=self.mode, 
                data_dir=self.data_dir, label_col=self.label_col, radio_dict = self.radio_dict,
                slides_dict=self.slides_dict, num_classes=self.num_classes,metadata = self.metadata)
        else:
            split = None
        
        return split

    def return_whole_splits(self,csv_file: str=None):

        df_slice = self.slides_radio_data.reset_index(drop=True)
        split = Split(df_slice, modalities = self.modalities, mode=self.mode, 
                data_dir=self.data_dir, label_col=self.label_col, radio_dict = self.radio_dict,
                slides_dict=self.slides_dict, num_classes=self.num_classes,metadata = self.metadata)

        if csv_file is not None:
            all_splits = pd.read_csv(csv_file)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')

            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            split.apply_scaler(scalers=scalers)

        return split


    def return_train_val_splits(self, from_id: bool=True, csv_path: str=None):
        assert csv_path 
        all_splits = pd.read_csv(csv_path)
        train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
        val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')

        try:
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
        except AttributeError:
            pass

        return train_split, val_split

    def return_train_val_test_splits(self, from_id: bool=True, csv_path: str=None):
        assert csv_path 
        all_splits = pd.read_csv(csv_path)
        train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
        val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
        test_split = self.get_split_from_df(all_splits = all_splits, split_key = 'test')
        try:
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            test_split.apply_scaler(scalers=scalers)
        except AttributeError:
            pass        
        return train_split, val_split, test_split

    def do_split(self):
        df = self.slides_radio_data.reset_index(drop = True)
        df_train = df[df.train == 1]
        omics_features = [i for i in self.slides_radio_data.columns if '_cnv' in i or '_mut' in i]
        threemod = df_train.dropna(subset = ['slide_id'] + self.modalities + omics_features).reset_index(drop = True)
        threemod_subject = np.unique(threemod.subject_id.values)
        if sum(df.train == 0) !=0:
            df_test = df[df.train == 0]
            test_threemod = df_test.dropna(subset = ['slide_id'] + self.modalities + omics_features).reset_index(drop = True)
            test_threemod_subject = np.unique(test_threemod.subject_id.values)
        os.makedirs( self.split_dir,exist_ok = True)
        replace_split = 'y'
        if len(os.listdir( self.split_dir)) != 0:
            replace_split = input('splits already exist! Replace split? (y/n)')

        if replace_split == 'y':
            if self.split == 'threemod':
                if len(threemod) < 120:
                    how = 'shuffle_split'
                    test_size = 0.2
                else:
                    how = 'k_fold'
                    test_size = None

                if sum(df.train == 0) !=0:
                    train_test_splits = self.train_val_split(threemod, self.k, self.split_dir , seed = self.seed, save = False, how = how, test_size = test_size)#[0]
                    df_test_subject = df_test.subject_id
                    for i, split_i in enumerate(train_test_splits):
                        split_i['test'] = list(test_threemod_subject)+ [np.nan] * (len(split_i)-len(test_threemod_subject))
                        split_i.to_csv(os.path.join(self.split_dir ,f'splits_{i}.csv'), index = False)
                else:
                    self.train_val_split(threemod, self.k, self.split_dir, seed = self.seed, how = how, test_size = test_size)
                #self.train_val_split(threemod, self.k, self.split_dir, seed = self.seed, how = 'k_fold')
            elif self.mode == 'radio' and self.split == 'pre_trained':
                radio_df = df.dropna(subset=self.modalities)
                radio_df = radio_df[~radio_df.subject_id.isin(threemod_subject)].reset_index(drop = True)
                how = 'shuffle_split'
                test_size = 0.1
                #if len(radio_df) < 150:
                #    how = 'shuffle_split'
                #    test_size = 0.1
                #else:
                #    how = 'k_fold'
                #    test_size = None
                self.train_val_split(radio_df, self.k, self.split_dir, seed = self.seed, how = how, test_size = test_size)

            elif self.mode == 'omic' and self.split == 'pre_trained':
                omic_df = df.dropna(subset=omics_features)
                omic_df = omic_df[~omic_df.subject_id.isin(threemod_subject)].reset_index(drop = True)
                how = 'shuffle_split'
                test_size = 0.1
                #if len(omic_df) < 150:
                #    how = 'shuffle_split'
                #    test_size = 0.1
                #else:
                #    how = 'k_fold'
                #    test_size = None
                self.train_val_split(omic_df, self.k, self.split_dir, seed = self.seed, how = how, test_size = test_size)
            elif self.mode == 'path' and self.split == 'pre_trained':
                path_df = df.dropna(subset=['slide_id'])
                path_df = path_df[~path_df.subject_id.isin(threemod_subject)].reset_index(drop = True)
                how = 'shuffle_split'
                test_size = 0.1
                #if len(path_df) < 150:
                #    how = 'shuffle_split'
                #    test_size = 0.1
                #else:
                #    how = 'k_fold'
                #    test_size = None
                self.train_val_split(path_df, self.k, self.split_dir, seed = self.seed, how = how, test_size = test_size)


    @staticmethod
    def train_val_split(df,cv,path, seed = 0, save = True, how = None, test_size = None):
        
        all_splits = []
        try:
            if how == 'k_fold':
                sss = StratifiedKFold(n_splits=cv, shuffle = True, random_state = seed)
            elif how == 'shuffle_split':
                sss = StratifiedShuffleSplit(n_splits=cv, test_size = test_size, random_state = seed)
            i = 0
            print(Counter(df.label))
            for train_index, val_index in sss.split(df.subject_id.values,df['label'].values):
                train_ids = df.subject_id.values[train_index]
                val_ids = df.subject_id.values[val_index]
                final_val_ids = np.append(val_ids,np.repeat(np.nan,len(train_ids) - len(val_ids)))
                
                split_temp = pd.DataFrame({'train':train_ids,'val':final_val_ids})
                if save:
                    split_temp.to_csv(os.path.join(path,f'splits_{i}.csv'), index = False)
                else:
                    all_splits.append(split_temp)

                i+=1
        except ValueError:
            count_1 = [i for i,c in Counter(df['label'].values).items() if c ==1]
            subject_id_1 = np.array(df[df['label'].isin(count_1)].subject_id.values)
            df = df[~df['label'].isin(count_1)].reset_index(drop = True)
            i = 0
            if how == 'k_fold':
                sss = StratifiedKFold(n_splits=cv, shuffle = True, random_state = seed)
            elif how == 'shuffle_split':
                sss = StratifiedShuffleSplit(n_splits=cv, test_size = test_size, random_state = seed)
            for train_index, val_index in sss.split(df.subject_id.values,df['label'].values):
                train_ids = df.subject_id.values[train_index]
                if i == 0:
                    val_ids = df.subject_id.values[val_index]
                    val_ids = np.append(val_ids,subject_id_1)
                    final_val_ids = np.append(val_ids,np.repeat(np.nan,len(train_ids) - len(val_ids)))
                else:
                    train_ids = np.append(train_ids,subject_id_1)
                    val_ids = df.subject_id.values[val_index]
                    final_val_ids = np.append(val_ids,np.repeat(np.nan,len(train_ids) - len(val_ids)))
                
                split_temp = pd.DataFrame({'train':train_ids,'val':final_val_ids})
                if save:
                    split_temp.to_csv(os.path.join(path,f'splits_{i}.csv'), index = False)
                else:
                    all_splits.append(split_temp)
                i+=1


        if not save:
            return all_splits

    def train_val_test_split(self,df,cv,path,seed = 0, how = None, test_size = None):
        os.makedirs(path,exist_ok = True)
        #import pdb;pdb.set_trace()

        train_test_splits = self.train_val_split(df, cv, path , seed, save = False, how = how, test_size = test_size)[0]
        train_temp_ids = train_test_splits.train.values
        test_temp_ids = train_test_splits.dropna().val
        train_temp_df = df[df.subject_id.isin(train_temp_ids)]
        train_val_splits = self.train_val_split(train_temp_df, cv, path , seed, save = False, how = how, test_size = test_size)

        for i , s in enumerate(train_val_splits):
            s['test'] = list(test_temp_ids) + [np.nan] * (len(s) - len(test_temp_ids))
            s.to_csv(os.path.join(path,f'splits_{i}.csv'), index = False)

    
    def __getitem__(self,idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_Survival_Dataset):
    def __init__(self,data_dir,**kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        
    def __getitem__(self, idx):
        subject_id = self.slides_radio_data['subject_id'][idx]
        label = self.slides_radio_data['disc_label'][idx]
        event_time = self.slides_radio_data[self.label_col][idx]
        c = self.slides_radio_data['censorship'][idx]
        modality_files = self.radio_dict[subject_id]
        genomic_features = np.array(self.genomic_features)[idx]

        if self.data_dir:
            radio_features = {}
            slices_index = {}
            if "radio" in self.mode:
                if all(pd.isna((list(modality_files.values())))):
                    for m in self.modalities:
                        radio_features[m] = torch.zeros((1,1))
                else:
                    for m in self.modalities:
                        radio_path = os.path.join(self.data_dir,'radio_h5_files',m,f'{subject_id}.h5')
                        file = h5py.File(radio_path, "r")
                        features = file['features'][:]
                        slice_id = file['slice_index'][:]
                        radio_features[m] = features
                        slices_index[m] = slice_id 
                    intersect = list(set.intersection(*[set(v) for k, v in slices_index.items()]))
                    for m in self.modalities:
                        radio_features[m] = torch.tensor(radio_features[m][np.in1d(slices_index[m], intersect),:])
                            
            else:
                for m in self.modalities:
                    radio_features[m] = torch.zeros((1,1))


            if "path" in self.mode and subject_id in self.slides_dict:
                slide_ids = self.slides_dict[subject_id]
                #if pathology is missing
                if len(slide_ids) == 0:
                    path_features = torch.zeros((1,1))
                #pathology not missing
                else:
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(self.data_dir, 'path_pt_files', slide_id.replace('.svs','.pt'))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
            else:
                path_features = torch.zeros((1,1))

            if "omic" in self.mode:
                if any(np.isnan(genomic_features)):
                    genomic_features = torch.zeros((1,1))
                else:
                    genomic_features = torch.tensor(genomic_features)
                    
            else:
                genomic_features = torch.zeros((1,1))


            if "radio" not in self.mode and "path" not in self.mode and "omic" not in self.mode:
                raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            return radio_features, path_features, genomic_features, label, event_time,c




class Generic_MIL_Survival_Dataset_Pretrained(Generic_Survival_Dataset):
    def __init__(self,data_dir,**kwargs):
        super(Generic_MIL_Survival_Dataset_Pretrained, self).__init__(**kwargs)
        self.data_dir = data_dir

    def __getitem__(self, idx):
        subject_id = self.slides_radio_data['subject_id'][idx]
        other_var = torch.tensor([[0]])#torch.tensor(self.slides_radio_data[['subject_id']][idx].values)
        label = self.slides_radio_data['disc_label'][idx]
        event_time = self.slides_radio_data[self.label_col][idx]
        c = self.slides_radio_data['censorship'][idx]

        if self.data_dir:
            try:
                radio_path = os.path.join(self.data_dir, 'radio_pt_files', '{}.pt'.format(subject_id))
                radio_features = torch.reshape(torch.load(radio_path),(1,256))
            except:
                radio_features = torch.zeros((1,256))
            try:
                wsi_path = os.path.join(self.data_dir, 'path_pt_files', '{}.pt'.format(subject_id))
                path_features = torch.reshape(torch.load(wsi_path),(1,256))
                #print(path_features)
            except:
                path_features = torch.zeros((1,256))

            try:
                omic_path = os.path.join(self.data_dir, 'omic_pt_files', '{}.pt'.format(subject_id))
                genomic_features = torch.reshape(torch.load(omic_path),(1,256))
                genomic_features = (genomic_features - genomic_features.min())/(genomic_features.max() - genomic_features.min())
            except:
                genomic_features = torch.zeros((1,256))


            #if "radio" not in self.mode and "path" not in self.mode and "omic" not in self.mode:
            #    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            #print(radio_features.min(), path_features.min(),genomic_features.min())
            return radio_features, path_features, genomic_features, label, event_time, c, other_var

    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slides_radio_data['subject_id'].isin(split.tolist())
            df_slice = self.slides_radio_data[mask].reset_index(drop=True)
            split = Split_Pretrained(df_slice, modalities = self.modalities, mode=self.mode, 
                data_dir=self.data_dir, label_col=self.label_col, radio_dict = self.radio_dict,
                slides_dict=self.slides_dict, num_classes=self.num_classes,metadata = self.metadata)
        else:
            split = None
        
        return split

class Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slides_radio_data,metadata, mode,  modalities, data_dir=None, label_col=None,
        radio_dict =None, slides_dict=None,
        #radio_fusion_after = False, 
        num_classes=2):
        #self.use_h5 = False
        self.slides_radio_data = slides_radio_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.slides_dict = slides_dict
        self.radio_dict = radio_dict
        self.modalities = modalities
        self.genomic_features = self.slides_radio_data.drop(self.metadata, axis=1)
        self.slide_cls_ids = [np.where(self.slides_radio_data['label'] == i)[0] for i in range(self.num_classes)]
    def __len__(self):
        return len(self.slides_radio_data)
        
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return scaler_omic

    def apply_scaler(self, scalers: tuple=None):
        self.genomic_features = scalers.transform(self.genomic_features)


class Split_Pretrained(Generic_MIL_Survival_Dataset_Pretrained):
    def __init__(self, slides_radio_data,metadata, mode,  modalities, data_dir=None, label_col=None,
        radio_dict =None, slides_dict=None,num_classes=4):
        #self.use_h5 = False
        self.slides_radio_data = slides_radio_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.slides_dict = slides_dict
        self.radio_dict = radio_dict
        self.modalities = modalities
        self.slide_cls_ids = [np.where(self.slides_radio_data['label'] == i)[0] for i in range(self.num_classes)]
    def __len__(self):
        return len(self.slides_radio_data)

