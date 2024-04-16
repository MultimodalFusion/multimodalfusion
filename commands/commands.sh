# EXAMPLE COMMANDLINE FOR TRAINING THE MODELS


# PATH UNIMODAL
CUDA_VISIBLE_DEVICES=0 python main.py --k 5 --which_splits <SPLIT_DIR_NAME> --split_mode train_val --mode path \
	--model_type path_attention_mil --task <NAME_CSV_FILE> --max_epochs 200 --seed 0 --early_stopping --gate_path \
	--data_root_dir <PATH_TO_FEATURES_DIR> --n_classes 8 --alpha_surv 0.6 --cancer_type brain --drop_out --split pre_trained

# OMICS UNIMODAL
CUDA_VISIBLE_DEVICES=1 python main.py --k 5 --which_splits <SPLIT_DIR_NAME> --split_mode train_val \
	--task <NAME_CSV_FILE> --data_root_dir <PATH_TO_FEATURES_DIR> --mode omic --model_type max_net \
    --max_epochs 200 --bag_loss cox_surv --batch_size 128 --seed 0 --early_stopping --n_classes 4 --drop_out --alpha_surv 0.6 \
    --cancer_type brain --reg_type all --split pre_trained 

# RADIO UNIMODAL
CUDA_VISIBLE_DEVICES=0 python main.py --k 5 --split_mode train_val --which_splits <SPLIT_DIR_NAME> --mode radio \
    --model_type radio_attention_mil --task <NAME_CSV_FILE> --max_epochs 200 --seed 0 --early_stopping \
    --data_root_dir <PATH_TO_FEATURES_DIR> --n_classes 8 --alpha_surv 0.6 --radio_mil_type attention --cancer_type brain \
    --drop_out --radio_fusion concat --gate_radio --split pre_trained

# FEATURE EXTRACTION
CUDA_VISIBLE_DEVICES=0 python pre_trained_feature.py --output_dir <PRETRAINED_FEATURE_DIR> --which_k <CHOSEN_CV> \
	--checkpoint_path <MODEL_DIR> \
	--extraction_csv_path <CSV_FILE_CONTAINING_PATIENTS_FOR_EXTRACTION> \
	--data_root_dir <PATH_TO_FEATURES_DIR>

#MULTIMODAL
#FINE-TUNING UNIMODAL
#PATH; FCNN; NLL SURVIVAL LOSS
CUDA_VISIBLE_DEVICES=1 python main_pretrained.py --k 10 --which_splits <SPLIT_DIR_NAME> --split_mode train_val_test \
    --mode path --model_type path_attention_mil --bag_loss nll_surv --batch_size 32 --task <NAME_CSV_FILE>  \
    --max_epochs 200 --seed 0 --early_stopping --data_root_dir <PRETRAINED_FEATURE_DIR> --n_classes 4 --reg_type all \
    --cancer_type brain --modality FLAIR,T1,T2,T1Gd --train_type fcnn --results_dir <RESULT_DIRECTORY>

#RADIO; HIGHWAY (1 LAYER); COX SURVIVAL LOSS
CUDA_VISIBLE_DEVICES=1 python main_pretrained.py --k 10 --which_splits <SPLIT_DIR_NAME> --split_mode train_val_test \
    --mode radio --model_type radio_attention_mil --bag_loss cox_surv --batch_size 32 --task <NAME_CSV_FILE>  \
    --max_epochs 200 --seed 0 --early_stopping --data_root_dir <PRETRAINED_FEATURE_DIR> --n_classes 4 --reg_type all \
    --cancer_type brain --modality FLAIR,T1,T2,T1Gd --train_type highway --n_layers 1 --results_dir <RESULT_DIRECTORY>

#OMICS; HIGHWAY (3 LAYERS); RANKING SURVIVAL LOSS
CUDA_VISIBLE_DEVICES=1 python main_pretrained.py --k 10 --which_splits <SPLIT_DIR_NAME> --split_mode train_val_test \
    --mode omic --model_type max_net --bag_loss ranking_surv  --batch_size 32 --task <NAME_CSV_FILE>  \
    --max_epochs 200 --seed 0 --early_stopping --data_root_dir <PRETRAINED_FEATURE_DIR> --n_classes 4 --reg_type all \
    --cancer_type brain --modality FLAIR,T1,T2,T1Gd --train_type highway --n_layers 3 --results_dir <RESULT_DIRECTORY>

#MULTIMODAL FUSION
#PATH+RADIO; EARLY-HIGHWAY (8 LAYERS); NLL-RANKING SURVIVAL LOSS
CUDA_VISIBLE_DEVICES=0 python main_pretrained.py --k 10 --which_splits <SPLIT_DIR_NAME> --split_mode train_val_test --mode radio_path \
    --model_type mm_attention_mil --bag_loss ranking_nll_surv --batch_size 32 --task <NAME_CSV_FILE> --max_epochs 200 --seed 0 \
    --early_stopping --data_root_dir <PRETRAINED_FEATURE_DIR> --n_classes 4 --reg_type all --train_type early-highway \
    --cancer_type brain --modality FLAIR,T1,T2,T1Gd --n_layers 8 --results_dir <RESULT_DIRECTORY>

#PATH+OMICS; EARLY-FCNN; NLL SURVIVAL LOSS
CUDA_VISIBLE_DEVICES=0 python main_pretrained.py --k 10 --which_splits <SPLIT_DIR_NAME> --split_mode train_val_test --mode path_omic \
    --model_type mm_attention_mil --bag_loss nll_surv --batch_size 32 --task <NAME_CSV_FILE> --max_epochs 200 --seed 0 \
    --early_stopping --data_root_dir <PRETRAINED_FEATURE_DIR> --n_classes 4 --reg_type all --train_type early-fcnn \
    --cancer_type brain --modality FLAIR,T1,T2,T1Gd --results_dir <RESULT_DIRECTORY>


#RADIO+OMICS; LATE-HIGHWAY (4 LAYERS); COX SURVIVAL LOSS
CUDA_VISIBLE_DEVICES=0 python main_pretrained.py --k 10 --which_splits <SPLIT_DIR_NAME> --split_mode train_val_test --mode radio_omic \
    --model_type mm_attention_mil --bag_loss cox_surv --batch_size 32 --task <NAME_CSV_FILE> --max_epochs 200 --seed 0 \
    --early_stopping --data_root_dir <PRETRAINED_FEATURE_DIR> --n_classes 4 --reg_type all --train_type late-highway \
    --cancer_type brain --modality FLAIR,T1,T2,T1Gd --n_layers 4 --results_dir <RESULT_DIRECTORY>


#PATH+OMICS+RADIO; KRONNECKER RPODUCT; RANKING SURVIVAL LOSS
CUDA_VISIBLE_DEVICES=0 python main_pretrained.py --k 10 --which_splits <SPLIT_DIR_NAME> --split_mode train_val_test \
    --mode path_omic_radio --model_type mm_attention_mil --bag_loss ranking_surv --batch_size 32 --task <NAME_CSV_FILE> \
    --max_epochs 200 --seed 0 --early_stopping --data_root_dir <PRETRAINED_FEATURE_DIR> --n_classes 4 --reg_type all \
    --train_type kronecker --cancer_type brain --modality FLAIR,T1,T2,T1Gd --results_dir <RESULT_DIRECTORY>
