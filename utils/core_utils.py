import numpy as np
import torch
import pickle 
import os
from collections import OrderedDict

#from datasets.dataset_generic import save_splits
from sklearn.metrics import roc_auc_score
from models.model_attention_mil_path import MIL_Attention_fc_path, MIL_Attention_fc_surv_path
from models.model_attention_mil_radio import MIL_Attention_fc_radio, MIL_Attention_fc_surv_radio
from models.model_mm_attention_mil import MM_MIL_Attention_fc, MM_MIL_Attention_fc_surv
from models.model_genomic import MaxNet
from utils.utils import *
from utils.loss_utils import *

from argparse import Namespace

from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored, integrated_brier_score,cumulative_dynamic_auc
from sksurv.util import Surv
def train(datasets: tuple, cur: int, args: Namespace, eval_mode: bool = False):
    """
        train for a single fold
        eval_mode: evaluation if true
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None


    print('\nInit train/val/test splits...', end=' ')
    if args.split_mode == 'train_val_test':
        train_split, val_split, test_split = datasets
    elif args.split_mode == 'train_val':
        train_split, val_split = datasets

    #save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    if args.split_mode == 'train_val_test':
        print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    elif args.bag_loss == 'ranking_surv':
        loss_fn = RankingSurvLoss()
    elif args.bag_loss == 'ranking_nll_surv':
        loss_fn = RankingNLLSurvLoss(alpha=args.alpha_surv, nll_ratio = args.nll_ratio)
    else:
        raise NotImplementedError


    if args.reg_type == 'all':
        reg_fn = l1_reg_all
    elif args.reg_type == 'omic_mm':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type =='path_attention_mil':
        model_dict.update({'gate_path': args.gate_path,'model_size_wsi': args.model_size_wsi,})

        model = MIL_Attention_fc_surv_path(**model_dict)
    elif args.model_type =='radio_attention_mil':
        model_dict.update({'radio_fusion': args.radio_fusion,'modalities':args.modality,'gate_radio': args.gate_radio})
        model = MIL_Attention_fc_surv_radio(**model_dict)
    elif args.model_type =='mm_attention_mil':
        model_dict.update({ 'input_dim': args.omic_input_dim, 'fusion': args.fusion, 'radio_fusion':args.radio_fusion,
            'model_size_wsi':args.model_size_wsi, 'model_size_radio':args.model_size_radio,'model_size_omic':args.model_size_omic,
            'gate_radio': args.gate_radio, 'gate_path': args.gate_path, 'gate':gate,
            'n_classes': args.n_classes, 'mode':args.mode})
        model = MM_MIL_Attention_fc_surv(**model_dict)
    elif args.model_type =='max_net':
        model_dict = {'input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic,'bag_loss': args.bag_loss}
        model = MaxNet(**model_dict)
    else:
        raise NotImplementedError
    
    model.relocate()
    print('Done!')
    print_network(model)
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, weighted = args.weighted_sample, batch_size=args.batch_size, radio_modality = args.modality)
    val_loader = get_split_loader(val_split, batch_size=args.batch_size, radio_modality = args.modality)
    if args.split_mode == 'train_val_test':
        test_loader = get_split_loader(test_split,   batch_size=args.batch_size, radio_modality = args.modality)
    print('Done!')


    if eval_mode:
        ckpt = torch.load(os.path.join(args.results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        model.load_state_dict(ckpt, strict=False)
        model.eval()

        results = {}
        results_val_dict, val_c_index = summary_survival(model,train_loader, args.n_classes, args.mode, args.bins)

        if args.split_mode == 'train_val_test':
            results_val_dict, val_c_index = summary_survival(model,val_loader, args.n_classes, args.mode, args.bins)
            results_test_dict, test_c_index = summary_survival(model,test_loader, args.n_classes, args.mode, args.bins)
            print('Val c-Index: {:.4f}'.format(val_c_index))
            print('Test c-Index: {:.4f}'.format(test_c_index))
            return results_val_dict, val_c_index, results_test_dict, test_c_index

        elif args.split_mode == 'train_val':
            results_val_dict, val_c_index = summary_survival(model,val_loader, args.n_classes, args.mode, args.bins)
            print('Val c-Index: {:.4f}'.format(val_c_index))
            return results_val_dict, val_c_index

    else:
        print('\nInit optimizer ...', end=' ')
        optimizer = get_optim(model, args)
        print('Done!')

        print('\nSetup EarlyStopping...', end=' ')
        if args.early_stopping:
            early_stopping = EarlyStopping(warmup=0, patience=20, stop_epoch=100, verbose = True)

        else:
            early_stopping = None
        print('Done!')

        for epoch in range(args.max_epochs):
            train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, args.mode, writer, 
                loss_fn, reg_fn, args.lambda_reg, args.gc, args.bins)
            stop = False
            if not stop:
                stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, args.mode, 
                    early_stopping, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args.bins)
            else:
                break

        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
        final_results_val_dict, final_val_cindex= summary_survival(model, val_loader, args.n_classes, args.mode, args.bins, loss_fn=loss_fn)
        if args.early_stopping:
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_minloss_checkpoint.pt".format(cur))))
        results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes, args.mode, args.bins, loss_fn=loss_fn)
        if args.split_mode == 'train_val_test':
            results_test_dict, test_cindex = summary_survival(model,test_loader, args.n_classes, args.mode, args.bins, loss_fn=loss_fn)


        print('Final Val c-Index: {:.4f}'.format(final_val_cindex))
        print('EarlyStoppig Val c-Index: {:.4f}'.format(val_cindex))
        if args.split_mode == 'train_val_test':
            print('EarlyStoppig Test c-Index: {: .4f}'.format(test_cindex))
        writer.close()

        if args.split_mode == 'train_val':
            return results_val_dict, val_cindex
        elif args.split_mode == 'train_val_test':
            return results_val_dict, val_cindex, results_test_dict, test_cindex

def train_loop_survival(epoch, model, loader, optimizer, n_classes, mode, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, t_bin = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_pred_survival = []

    for batch_idx, (radio_features ,path_features, genomic_features, label, event_time, c) in enumerate(loader):
        if 'omic' in mode and torch.equal(genomic_features.float(),torch.zeros((1,1))):
            continue

        if 'path' in mode and torch.equal(path_features,torch.zeros((1,1))):
            continue

        if 'radio' in mode and all([torch.equal(r,torch.zeros((1,1))) for i, r in radio_features.items()]):
            continue
 
        all_features = {i: r.to(device) for i, r in radio_features.items() }
        all_features['path_features'] = path_features.to(device)
        all_features['genomic_features'] = genomic_features.to(device).float()
        label = label.to(device)
        c = c.to(device)

        hazards, S, Y_hat, _ = model(**all_features) # return hazards, S, Y_hat, A_raw, results_dict

        if isinstance(loss_fn ,CoxSurvLoss) or isinstance(loss_fn, RankingSurvLoss):
            risk = hazards
            loss = loss_fn(risks=risk, times =torch.tensor(event_time).to(device), c=c)

        elif isinstance(loss_fn , NLLSurvLoss):
            risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        #elif isinstance(loss_fn , RankingNLLSurvLoss):
            #risk = -torch.sum(S, dim=1)
            #loss = loss_fn(hazards=hazards, risks =risk, S=S, Y=label, c=c)

        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        #import pdb;pdb.set_trace()
        all_risk_scores.append(risk.detach().cpu().numpy().flatten())
        all_censorships.append(c.detach().cpu().numpy())#.item()
        all_event_times.append(event_time)
        #all_pred_survival.append(S.detach().cpu().numpy())
        #print(f'-------batch {batch_idx}-------')


        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        bag_size = ''
        if 'path' in mode:
            bag_size = bag_size + 'Path ' + str(path_features.size(0)) + ' '
        #import pdb;pdb.set_trace()
        if 'radio' in mode:
            bag_size = bag_size + 'Radio ' + str(list(radio_features.values())[0].size(0))


        if (batch_idx + 1) % 50 == 0:
            print('batch {}, loss: {:.4f}, risk:{:4f} bag_size: {}'.format(batch_idx, loss_value + loss_reg, risk.detach().cpu().numpy()[0], bag_size))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    all_risk_scores = np.concatenate(all_risk_scores).flatten()
    all_event_times = np.concatenate(all_event_times).flatten()
    all_censorships = np.concatenate(all_censorships).flatten()


    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival(cur, epoch, model, loader, n_classes, mode, early_stopping=None, writer=None, 
    loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, t_bin = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # loader.dataset.update_mode(True)
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_pred_survival = []
    # model.alpha.requires_grad=True

    for batch_idx, (radio_features ,path_features, genomic_features, label, event_time, c) in enumerate(loader):
        if 'omic' in mode and torch.equal(genomic_features.float(),torch.zeros((1,1))):
            continue

        if 'path' in mode and torch.equal(path_features,torch.zeros((1,1))):
            continue

        if 'radio' in mode and all([torch.equal(r,torch.zeros((1,1))) for i, r in radio_features.items()]):
            continue

        all_features = {i: r.to(device) for i, r in radio_features.items() }
        all_features['path_features'] = path_features.to(device)
        all_features['genomic_features'] = genomic_features.to(device).float()
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            hazards, S, Y_hat, _ = model(**all_features)

        if isinstance(loss_fn ,CoxSurvLoss) or isinstance(loss_fn, RankingSurvLoss):
            risk = hazards
            loss = loss_fn(risks=risk, times =torch.tensor(event_time).to(device), c=c)
            #import pdb;pdb.set_trace()

        elif isinstance(loss_fn , NLLSurvLoss):
            risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        #elif isinstance(loss_fn , RankingNLLSurvLoss):
            #risk = -torch.sum(S, dim=1)
        #    loss = loss_fn(hazards=hazards, risks =risk, S=S, Y=label, c=c)

        loss_value = loss.item()
        #import pdb; pdb.set_trace()
        all_risk_scores.append(risk.detach().cpu().numpy().flatten())
        all_censorships.append(c.detach().cpu().numpy())#.item()
        all_event_times.append(event_time)


        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        #risk = -torch.sum(S, dim=1).cpu().numpy()

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    all_risk_scores = np.concatenate(all_risk_scores).flatten()
    all_event_times = np.concatenate(all_event_times).flatten()
    all_censorships = np.concatenate(all_censorships).flatten()

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]


    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if epoch == 10:
        torch.save(model.state_dict(), os.path.join(results_dir, 's_%d_mid_checkpoint.pt' % cur))
    print('\nVal Set, val_loss_surv: {:.4f}, val_loss: {:.4f}, val c-index: {:.4f}'.format(val_loss_surv, val_loss, c_index))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    # model.alpha.requires_grad=False
    return False


def summary_survival(model, loader, n_classes,mode, t_bin, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_subject_ids = []
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_pred_survival = []
    all_labels = []
        

    subject_ids = loader.dataset.slides_radio_data['subject_id']
    subject_count = 0

    for batch_idx, (radio_features ,path_features, genomic_features, label, event_time, c) in enumerate(loader):
        subject_id = loader.dataset.slides_radio_data['subject_id'][subject_count:(subject_count+len(label))]
        all_subject_ids.append(subject_id)
        subject_count += len(label)

        if 'omic' in mode and torch.equal(genomic_features.float(),torch.zeros((1,1))):
            continue

        if 'path' in mode and torch.equal(path_features,torch.zeros((1,1))):
            continue

        if 'radio' in mode and all([torch.equal(r,torch.zeros((1,1))) for i, r in radio_features.items()]):
            continue

        all_features = {i: r.to(device) for i, r in radio_features.items() }
        all_features['path_features'] = path_features.to(device)
        all_features['genomic_features'] = genomic_features.to(device).float()
        label = label.to(device)
        c = c.to(device)
        
        subject_id = subject_ids.iloc[batch_idx]


        with torch.no_grad():
            hazards, survival, Y_hat, _= model(**all_features)

        if isinstance(loss_fn ,CoxSurvLoss) or isinstance(loss_fn, RankingSurvLoss):
            risk = hazards

        elif isinstance(loss_fn , NLLSurvLoss):
            risk = -torch.sum(survival, dim=1)

        elif isinstance(loss_fn , RankingNLLSurvLoss):
            risk = -torch.sum(survival, dim=1)

        all_risk_scores.append(risk.detach().cpu().numpy().flatten())
        all_censorships.append( c.detach().cpu().numpy())#.item()
        all_event_times.append(event_time)
        #all_pred_survival.append( S.detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())


    all_risk_scores = np.concatenate(all_risk_scores).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_event_times = np.concatenate(all_event_times).flatten()
    all_censorships = np.concatenate(all_censorships).flatten()
    all_subject_ids = np.concatenate(all_subject_ids).flatten()


    patient_results={'subject_id':all_subject_ids,'risk': all_risk_scores, 'disc_label': all_labels, 
        'survival': all_event_times, 'censorship': all_censorships}

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]


    return patient_results, c_index

"""
def transfer_and_train(datasets: tuple, cur: int, args):

    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None


    print('\nInit train/val/test splits...', end=' ')
    if args.split_mode == 'train_val_test':
        train_split, val_split, test_split = datasets
    elif args.split_mode == 'train_val':
        train_split, val_split = datasets

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    if args.split_mode == 'train_val_test':
        print("Testing on {} samples".format(len(test_split)))


    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError

    if args.reg_type == 'all':
        reg_fn = l1_reg_all
    elif args.reg_type == 'omic_mm':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'transfer': True}

    if args.model_type =='path_attention_mil':
        model = MIL_Attention_fc_surv_path(**model_dict)

    elif args.model_type =='radio_attention_mil':
        model_dict.update({'radio_mil_type':args.radio_mil_type})
        model = MIL_Attention_fc_surv_radio(**model_dict)


    elif args.model_type =='mm_attention_mil':
        model_dict.update({ 'input_dim': args.omic_input_dim, 'fusion': args.fusion, 'radio_fusion':args.radio_fusion,
            'model_size_wsi':args.model_size_wsi, 'model_size_radio':args.model_size_radio,'model_size_omic':args.model_size_omic,
            'gate_radio': args.gate_radio, 'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 
            'n_classes': args.n_classes, 'mode':args.mode})

        model = MM_MIL_Attention_fc_surv(**model_dict)

    elif args.model_type =='max_net':
        model_dict = {'input_dim': args.omic_input_dim, 
        'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        if args.radio_mil_type == 'attention':
            model_dict.update({'transfer': True})
            model = MaxNet(**model_dict)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    
    ### Obtain the weights from the pre-trained models
    print('\nLoad Weights from pre-trained unimodal models .... ', end=' ')


    if 'radio' in args.mode:
        radio_weights = torch.load(os.path.join(args.checkpoints,'radio_checkpoint.pt'))
        if args.radio_mil_type == 'attention':
            radio_weights = {i:v for i, v in radio_weights.items() if 'attention_net' in i or 'reduce_dim' in i or 'radio_xfusion' in i}
        elif args.radio_mil_type == 'max':
            radio_weights = {i:v for i, v in radio_weights.items() if 'pre_pool' in i}
            print(radio_weights)
        elif args.radio_mil_type == 'avg':
            radio_weights = {i:v for i, v in radio_weights.items() if 'pre_pool' in i}

        model.load_state_dict(radio_weights, strict=False)

    if 'path' in args.mode:
        path_weights = torch.load(os.path.join(args.checkpoints,'wsi_checkpoint.pt'))
        path_weights = {i:v for i, v in path_weights.items() if 'attention_net' in i}
        model.load_state_dict(path_weights, strict=False)
    
    if 'omic' in args.mode:
        omic_weights = torch.load(os.path.join(args.checkpoints,'omics_checkpoint.pt'))
        omic_weights = {i:v for i, v in omic_weights.items() if 'fc_omic' in i}

        model.load_state_dict(omic_weights, strict=False)

    print('Done!')
    
    #Freeze pre-trained layers and only optimize fusion layers
    print('\nFreeze layers ...', end=' ')
    layer_names = list(model.state_dict().keys())
    fusion_layers = [ i for i, n in enumerate(layer_names) if 'classifier' in n or 'mm' in n]
    for i , p in enumerate(model.parameters()):
        if i not in fusion_layers:
            p.requires_grad = False
    print('Done!')

    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
                                    weighted = args.weighted_sample,batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing,  batch_size=args.batch_size)
    if args.split_mode == 'train_val_test':
        test_loader = get_split_loader(test_split,  testing = args.testing,  batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=20, stop_epoch=100, verbose = True)

    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    print('Done!')


    for epoch in range(args.max_epochs):
        train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, args.mode, writer, 
            loss_fn, reg_fn, args.lambda_reg, args.gc, args.bins)
        stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, args.mode, 
            early_stopping, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args.bins)

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    final_results_val_dict, final_val_cindex= summary_survival(model, val_loader, args.n_classes, args.mode, args.bins)
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_minloss_checkpoint.pt".format(cur))))
    results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes, args.mode, args.bins)
    if args.split_mode == 'train_val_test':
        results_test_dict, test_cindex = summary_survival(model,test_loader, args.n_classes, args.mode, args.bins)
    print('Final Val c-Index: {:.4f}'.format(final_val_cindex))
    print('EarlyStoppig Val c-Index: {:.4f}'.format(val_cindex))
    if args.split_mode == 'train_val_test':
        print('EarlyStoppig Test c-Index: {: .4f}'.format(test_cindex))
    writer.close()

    if args.split_mode == 'train_val':
        return results_val_dict, val_cindex
    elif args.split_mode == 'train_val_test':
        return results_val_dict, val_cindex, results_test_dict, test_cindex
"""