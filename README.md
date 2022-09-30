# AI-based multimodal integration of radiology, pathology and genomics for outcome prediction

### Prerequisite
<br/><br>
### Data Preparation
---
#### Public Dataset
Whole slide images (WSI) from can be downloaded from [GDC Data Portal](https://portal.gdc.cancer.gov/).Radiology images, including MRI scans and CT scans, are available on [TCIA](https://www.cancerimagingarchive.net/).  More brain MRI scans are avilable on [BRATS2020](https://www.med.upenn.edu/cbica/brats2020/data.html). Pre-processed genomic data can be downloaded from [cBioPortal](https://www.cbioportal.org/). [National Lung Screening Trial (NLST)](https://cdas.cancer.gov/nlst/) contain both CT scans and WSIs for lung cancer. People can submit a request to access the data. The survival information, including censorship and survival time, is available in all dataset. We can match the patients from TCGA to TCIA using patient identifier.

#### Cleaned CSV File
We cleaned a csv file which includes:
- **subject_id**: patient identifier
- **slide_id**: WSI svs file name 
- radiology modalities(**FLAIR**,**T2**,**T1**,**T1Gd** or **CT**): Name of the folder containing corresponding modalities in radiology images
- **oncotree_code**: type of cancer
- **is_female**: whether the patient is female (1: female; 0:male)
- **age**: age of the patient at diagnosis
- **survival_months**: survival time in months
- **censorship**: whether the patient censored (1: censored or survived; 0: uncensored or died )
- **train**: whether the patient is in the test cohort (1: not in test cohort; 0: in test cohort)
- Genomics features (e.g. **IDH1_mut**): each column indicates one genomic features

This csv file will be used in feature extractions, training, evaluation, and interpretability.
<br/><br>
### Preprocessing
---
#### Histology 
The first step is to segment and patch WSIs.
```bash
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 
```
It will output:
- `masks/`: Directory of segmented tissue regions
- `patches/`: Directory of extracted image patches in .h5 file
- `stitches/`: Directory of downsampled visualizations of stitched tissue patches
- `process_list_autogen.csv`: A csv file that contains segmentation and patching parameters for all slides processed. This file will be used in generating interpretability in WSIs later.

After patching, `extracted_features_fp.py` passed the patches into ResNet50 for feature extraction and output 1024-dimensional feature vector for each patch. Those features are saved in .h5 file and .pt file.
```bash
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 256 --slide_ext .svs
```

#### Radiology
--cancer_type glioma
--batch_size 128 
```bash
CUDA_VISIBLE_DEVICES=0 python feature_extraction.py --radio_dir <RADIO_DATA_DIR> --csv_path <PATH_TO_CSV_FILE> \
  --output_dir <PATH_TO_FEATURES_DIR>  
```

<br/><br>
### Pretraining models
---
Following data preprocessing, we pre-trained unimodal models on three modalities respectively to address the issue of data scarcity. 
```bash
#Pretrain Radio AMIL
CUDA_VISIBLE_DEVICES=0 python main.py --k <N_FOLD> --which_splits <SPLIT_DIR_NAME> --split_mode train_val \
  --mode radio --model_type radio_attention_mil --task <NAME_CSV_FILE> --data_root_dir <PATH_TO_FEATURES_DIR> 

#Pretrain Histology AMIL
CUDA_VISIBLE_DEVICES=0 python main.py --k <N_FOLD> --which_splits <SPLIT_DIR_NAME> --split_mode train_val \
  --mode omic --model_type max_net --bag_loss cox_surv --task <NAME_CSV_FILE> --data_root_dir <PATH_TO_FEATURES_DIR> 

#Pretrain Omics SNN
CUDA_VISIBLE_DEVICES=0 python main.py --k <N_FOLD> --which_splits <SPLIT_DIR_NAME> --split_mode train_val \
  --mode path --model_type path_attention_mil --task <NAME_CSV_FILE> --data_root_dir <PATH_TO_FEATURES_DIR>
```
#### Train val split
If the splits have been created, `--which_splits` needs to be a folder name under `./splits`; if the splits have not been created, `--split pre_trained` could be used to do stratified random k-fold (`--k`) train validation split based on N discrete time intervals (`--n_classes`). The splits will be saved under `./splits` with the name specified in `--which_splits`.

#### Training
The cancer type should be specified in `--cancer_type`: glioma or lung. It should match the name of the subfolder in the folders where you store data, features, splits, and results. In radiology models, users need to specify the name each radiology modality with `--modality`. This should match the name of the folder where features and original files were stored. 

There are some key hyperparameters:
* `--gate_radio` and `--gate_path` (bool): whether to use gated attention modules.
* `--max_epochs` (int):the number of epochs for training.
* `--early_stopping` (bool): save the model with the lowest validation loss and therefore prevent overfitting
* `--drop_out` (bool): whether to use drop out in AMIL
* `--bag_loss` (str): what loss function to be used. Note: Histology AMIL and Radiology AMIL can only use nll or ranking-nll loss functions.
* `--batch_size` (int): the number of samples in each batch. The batch_size should be 1 in Histology AMIL and Radiology AMIL.
* `--seed` (int): set seed for reproducibility.
* `--reg_type` (str): L1 regularization
* `--alpha_surv` (float): the hyperparameter for nll or ranking-nll loss to indicate the weights assigned to uncensored patients.

The model will save model checkpoints for each cross validation in .pt files. The predicted risks of the validation set will be saved in pickle files. `summary.csv` will store all validation c-index for each cv. All the hyperparamters used will be saved in a .txt file.
<br/><br>
### Extracting features embeddings
---
We extracted the pre-trained features from the last layer of the pre-trained models before predictions. The default output from the models will be a 256-dimensional feature vector. We can use the following command to extract features:
```bash
CUDA_VISIBLE_DEVICES=0 python pre_trained_feature.py --output_dir <PRETRAINED_FEATURE_DIR> --which_k <CHOSEN_CV> \
  --checkpoint_path <MODEL_DIR> --extraction_csv_path <CSV_FILE_CONTAINING_PATIENTS_FOR_EXTRACTION>
```
Users need to specify the name of output directory with `--output_dir`, the path to model with `--checkpoint_path`, the cv chose for feature extraction with `--which_k`, and also the subset of patients with `--extraction_csv_path`. The python file will automatically load the .txt file where all the hyperparameters were stored.
<br/><br>
### Multimodal fusion
---
There are three types of multimodal fusion implemented: early concatenation, late concatenation, and kronecker. For early and late concatenation, users can select from feed-forward neural network or highway network. The type of fusion model should be specified with `--train_type`. To change the number of layers in the highway network, users need to add `--n_layers`.

```bash
CUDA_VISIBLE_DEVICES=0 python main_pretrained.py --k <N_FOLD> --which_splits <SPLIT_DIR_NAME> --split_mode train_val \
  --mode path_omic_radio --model_type mm_attention_mil --task <NAME_CSV_FILE> --data_root_dir <PRETRAINED_FEATURE_DIR>
```
The hyperparameters, including `cancer_type`, `bag_loss`, `early_stopping`, `batch_size`, `max_epochs`, `modality`, `n_classes`, `reg_type`, and so on, can also be used to fine-tune the multimodal fusion model.
<br/><br>
### Evaluation
---
To evaluate the pre-trained model on an external dataset, users need to add a test column or replace the validation column with all patients in splits files, and run the following command line:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_pretrained.py --which_splits <SPLIT_DIR_NAME> --split_mode train_val_test \
  --model_path <PATH_TO_FUSION_MODEL_DIRECTORY>
```
The evaluation script will save predicted risks of patients in both the validation set and test set. `summary.csv` will store c-index, and ibs (if using nll or ranking-nll loss) for each cv. 
<br/><br>
### Interpretability
---
Our multimodal fusion models are interpretable at both the population-level and patient-level.

1. Multimodal Attributions
By running the python script `create_attributions.py` with the input of `--model_path`, which is the model directory, users are able to calculate the attribution of each modality.
```bash
CUDA_VISIBLE_DEVICES=0 python create_attributions.py --model_path <PATH_TO_MODEL_DIRECTORY>
```

2. WSI
The attention heatmap on WSI utilized the attentions scores from AMIL. Users need to modify the configuration file to set the input parameters for heatmap. A sample file, `config_path.yaml`, can be found in . Users can add `--heatmap` to plot the heatmap and `--sampling` to sample the the most important or least important patches.
```bash
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config_file <PATH_TO_PATH_CONFIG_FILE> --heatmap --sampling
```

3. MRI and CT
The interpretability for radiology images include two steps. The first step is sampling the high-attention and low-attention patches using `create_heatmaps.py`. The second step is to run `gradcam.py`

```bash
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config_file <PATH_TO_RADIO_CONFIG_FILE> --sampling

CUDA_VISIBLE_DEVICES=0 python gradcam.py --radio_dir <PATH_TO_MRI_CT> \
  --radio_pt_dir <PATH_TO_PRETRAINED_RADIO_FEATURES> --patches_dir <PATH_TO_PATCHES_DIRECTORY> \
  --csv_path <PATH_TO_CSV_FILE> --ckpt_path <PATH_TO_MODEL_CHECKPOINT>
```

4. Omics
We utilized SHAP package here to calculate the integrated gradients attributions for each genomic features in each patient. Run the following command line with the omics configuration file (config_omics.yaml) to compute the attributions and plot the SHAP plots.
```bash
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config_file <PATH_TO_OMIC_CONFIG_FILE>
```
The script will save a global SHAP summary plot and local SHAP plot for each of the patient. The attributions will be saved in pickle files.


### License



