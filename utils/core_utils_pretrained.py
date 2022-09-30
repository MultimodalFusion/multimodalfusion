import numpy as np
import torch
import pickle 
from utils.utils import *
from utils.loss_utils import *
import os
from collections import OrderedDict

from sklearn.metrics import roc_auc_score
import models.coxranking_models_pretrained as coxranking_models
import models.nll_models_pretrained as nll_models

from argparse import Namespace

from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored, integrated_brier_score,cumulative_dynamic_auc,concordance_index_ipcw
from utils.utils import EarlyStopping

from sksurv.util import Surv


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
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
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'bag_loss': args.bag_loss,
    'train_type':args.train_type,'mode':args.mode, 'n_layers': args.n_layers}
    if args.model_type =='path_attention_mil' or args.model_type =='radio_attention_mil' or args.model_type =='max_net':
        if 'nll_surv' in args.bag_loss:
            model = nll_models.unimonal_pretrained(**model_dict)
        else:
            model = coxranking_models.unimonal_pretrained(**model_dict)
    elif args.model_type =='mm_attention_mil':
        if 'nll_surv' in args.bag_loss:
            model = nll_models.multimodal_pretrained(**model_dict)
        else:
            model = coxranking_models.multimodal_pretrained(**model_dict)
    else:
        raise NotImplementedError
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)

    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_pretrained_split_loader(train_split, training=True,weighted = args.weighted_sample, batch_size=args.batch_size)
    val_loader = get_pretrained_split_loader(val_split, batch_size=args.batch_size)
    if args.split_mode == 'train_val_test':
        test_loader = get_pretrained_split_loader(test_split, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None

    print('Done!')
    stop = False
    for epoch in range(args.max_epochs):
        train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, args.mode, writer, 
            loss_fn, reg_fn, args.lambda_reg, args.gc, args.bins, args.train_type)
        if not stop:    
            stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, args.mode, 
                early_stopping, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args.bins,  args.train_type)
        else:
            break

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    #import pdb; pdb.set_trace()
    final_results_val_dict, final_val_cindex= summary_survival(model, val_loader, args.n_classes, args.mode, 
        args.bins, train_type =args.train_type, loss_fn=loss_fn)
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_minloss_checkpoint.pt".format(cur))))
    results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes, 
        args.mode, args.bins,train_type = args.train_type,loss_fn=loss_fn)
    if args.split_mode == 'train_val_test':
        results_test_dict, test_cindex = summary_survival(model,test_loader, args.n_classes, args.mode, 
            args.bins,train_type =args.train_type,loss_fn=loss_fn)


    print('Final Val c-Index: {:.4f}'.format(final_val_cindex))
    print('EarlyStoppig Val c-Index: {:.4f}'.format(val_cindex))
    if args.split_mode == 'train_val_test':
        print('EarlyStoppig Test c-Index: {: .4f}'.format(test_cindex))
    writer.close()

    if args.split_mode == 'train_val':
        return results_val_dict, val_cindex
    elif args.split_mode == 'train_val_test':
        return results_val_dict, val_cindex, results_test_dict, test_cindex

def train_loop_survival(epoch, model, loader, optimizer, n_classes, mode, writer=None, loss_fn=None, reg_fn=None, 
    lambda_reg=0., gc=16, t_bin = None, train_type = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_pred_survival = []

    for batch_idx, (radio_features ,path_features, genomic_features, label, event_time, c,masks) in enumerate(loader):
        if train_type == 'multimodal-dropout':
            for _, param in model.named_parameters():
                param.requires_grad = True
            if 'omic' in mode and torch.equal(genomic_features.float(),torch.zeros((1,256))):
                for name, param in model.named_parameters():
                    if 'omic' in name and param.requires_grad:
                        param.requires_grad = False

            if 'path' in mode and torch.equal(path_features,torch.zeros((1,256))):
                for name, param in model.named_parameters():
                    if 'WSI' in name and param.requires_grad:
                        param.requires_grad = False        

            if 'radio' in mode and torch.equal(radio_features,torch.zeros((1,256))):
                for name, param in model.named_parameters():
                    if 'MRI' in name and param.requires_grad:
                        param.requires_grad = False        

        radio_features = radio_features.to(device)
        path_features = path_features.to(device)
        genomic_features = genomic_features.to(device)
        label = label.to(device)

        c = c.to(device)

        risk, hazards, S = model(h_radio=radio_features,h_path = path_features,h_omic = genomic_features )
        if isinstance(loss_fn ,CoxSurvLoss) or isinstance(loss_fn, RankingSurvLoss):
            loss = loss_fn(risks=risk, times =torch.tensor(event_time).to(device), c=c)

        elif isinstance(loss_fn , NLLSurvLoss):
            #risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        elif isinstance(loss_fn , RankingNLLSurvLoss):
            #risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, risks =risk, S=S, Y=label, c=c)

        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
            loss_reg = loss_reg.detach().cpu().numpy().item()

        print(f'-------batch {batch_idx}-------')

        #import pdb; pdb.set_trace()
        all_risk_scores.append(risk.detach().cpu().numpy().squeeze())

        all_censorships.append(c.detach().cpu().numpy())#.item()
        all_event_times.append(event_time)
        #all_pred_survival.append(S.detach().cpu().numpy())


        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

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


    #print(all_censorships,all_event_times,all_risk_scores)
    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))
    #print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}, train_ibs: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index, ibs))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
        #writer.add_scalar('train/ibs', ibs, epoch)

def validate_survival(cur, epoch, model, loader, n_classes, mode, early_stopping=None,writer=None, 
    loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, t_bin = None, train_type = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # loader.dataset.update_mode(True)
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_pred_survival = []
    # model.alpha.requires_grad=True

    for batch_idx, (radio_features ,path_features, genomic_features, label, event_time, c,masks) in enumerate(loader):
        radio_features = radio_features.to(device)
        path_features = path_features.to(device)
        genomic_features = genomic_features.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            risk, hazards, S = model(h_radio=radio_features,h_path = path_features,h_omic = genomic_features )

        if isinstance(loss_fn ,CoxSurvLoss) or isinstance(loss_fn, RankingSurvLoss):
            loss = loss_fn(risks=risk, times =torch.tensor(event_time).to(device), c=c)
            #import pdb;pdb.set_trace()

        elif isinstance(loss_fn , NLLSurvLoss):
            #risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        elif isinstance(loss_fn , RankingNLLSurvLoss):
            #risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, risks =risk, S=S, Y=label, c=c)

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
            loss_reg = loss_reg.detach().cpu().numpy().item()
        loss_value = loss.item()
        #import pdb; pdb.set_trace()
        all_risk_scores.append(risk.detach().cpu().numpy().squeeze())
        all_censorships.append(c.detach().cpu().numpy())#.item()
        all_event_times.append(event_time)
        #all_pred_survival.append(S.detach().cpu().numpy())


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

    return False


def summary_survival(model, loader, n_classes,mode, t_bin,train_type = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.
    all_subject_ids = []
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_pred_survival = []
    all_labels = []

    #subject_ids = loader.dataset.slides_radio_data['subject_id']
    #patient_results = {}
    subject_count = 0
    for batch_idx, (radio_features ,path_features, genomic_features, label, event_time, c,masks) in enumerate(loader):
        subject_id = loader.dataset.slides_radio_data['subject_id'][subject_count:(subject_count+len(label))]
        all_subject_ids.append(subject_id)
        subject_count += len(label)
        
        radio_features = radio_features.to(device)
        path_features = path_features.to(device)
        genomic_features = genomic_features.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            risk, hazards, S = model(h_radio=radio_features,h_path = path_features,h_omic = genomic_features )

        if isinstance(loss_fn ,CoxSurvLoss) or isinstance(loss_fn, RankingSurvLoss):
            loss = loss_fn(risks=risk, times =torch.tensor(event_time).to(device), c=c)

        elif isinstance(loss_fn , NLLSurvLoss):
            risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        elif isinstance(loss_fn , RankingNLLSurvLoss):
            risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, risks =risk, S=S, Y=label, c=c)

        all_risk_scores.append(risk.detach().cpu().numpy().squeeze())
        all_censorships.append( c.detach().cpu().numpy())#.item()
        all_event_times.append(event_time)
        #all_pred_survival.append(hazards.detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())


    #import pdb; pdb.set_trace()
    all_risk_scores = np.concatenate(all_risk_scores).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_event_times = np.concatenate(all_event_times).flatten()
    all_censorships = np.concatenate(all_censorships).flatten()
    all_subject_ids = np.concatenate(all_subject_ids).flatten()

    patient_results={'subject_id':all_subject_ids,'risk': all_risk_scores, 'disc_label': all_labels, 
        'survival': all_event_times, 'censorship': all_censorships}
    #survival_test = Surv.from_arrays(1-all_censorships, all_event_times)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    #ibs = brier_score(survival_train , survival_test, all_hazards, times)

    return patient_results, c_index




def eval_model(datasets: tuple, cur: int, args):
    """   
        train for a single fold
    """
    print('\nEvaluating Fold {}!'.format(cur))

    print('\nInit train/val/test splits...', end=' ')
    if args.split_mode == 'train_val_test':
        train_split, val_split, test_split = datasets
    elif args.split_mode == 'train_val':
        train_split, val_split = datasets    

    print('Done!')
    print("Validating on {} samples".format(len(val_split)))
    if args.split_mode == 'train_val_test':
        print("Testing on {} samples".format(len(test_split)))


    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss()
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss()
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    elif args.bag_loss == 'ranking_surv':
        loss_fn = RankingSurvLoss()
    elif args.bag_loss == 'ranking_nll_surv':
        loss_fn = RankingNLLSurvLoss()
    else:
        raise NotImplementedError
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = { 'n_classes': args.n_classes, 'bag_loss': args.bag_loss,
    'train_type':args.train_type,'mode':args.mode, 'n_layers': args.n_layers}
    if args.model_type =='path_attention_mil' or args.model_type =='radio_attention_mil' or args.model_type =='max_net':
        if 'nll_surv' in args.bag_loss:
            model = nll_models.unimonal_pretrained(**model_dict)
        else:
            model = coxranking_models.unimonal_pretrained(**model_dict)
    elif args.model_type =='mm_attention_mil':
        if 'nll_surv' in args.bag_loss:
            model = nll_models.multimodal_pretrained(**model_dict)
        else:
            model = coxranking_models.multimodal_pretrained(**model_dict)
    else:
        raise NotImplementedError
    
    model.relocate()
    print('Done!')
    print_network(model)
    ckpt = torch.load(os.path.join(args.model_path, "s_{}_minloss_checkpoint.pt".format(cur)))
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    print('\nInit Loaders...', end=' ')
    train_loader = get_pretrained_split_loader(train_split, batch_size=args.batch_size)
    val_loader = get_pretrained_split_loader(val_split, batch_size=args.batch_size)
    if args.split_mode == 'train_val_test':
        test_loader = get_pretrained_split_loader(test_split, batch_size=args.batch_size)
    print('Done!')
    if 'nll_surv' in args.bag_loss:
        survival_train = summary_survival_ibs(model, train_loader, args.n_classes, 
            args.mode, args.bins,train_type = args.train_type,loss_fn=loss_fn, bins = args.bins)
    else:
        survival_train = None

    results_val_dict, val_cindex, val_ibs = summary_survival_ibs(model, val_loader, args.n_classes, 
        args.mode, args.bins,train_type = args.train_type,loss_fn=loss_fn, bins = args.bins, survival_train = survival_train)
    if args.split_mode == 'train_val_test':
        results_test_dict, test_cindex, test_ibs = summary_survival_ibs(model,test_loader, args.n_classes, args.mode, 
            args.bins,train_type =args.train_type,loss_fn=loss_fn, bins = args.bins, survival_train = survival_train)

    print('EarlyStoppig Val c-Index: {:.4f}'.format(val_cindex))
    if args.split_mode == 'train_val_test':
        print('EarlyStoppig Test c-Index: {: .4f}'.format(test_cindex))

    if args.split_mode == 'train_val':
        return results_val_dict, val_cindex, val_ibs
    elif args.split_mode == 'train_val_test':
        return results_val_dict, val_cindex,val_ibs, results_test_dict, test_cindex,test_ibs


def summary_survival_ibs(model, loader, n_classes,mode, t_bin,train_type = None, loss_fn = None, bins = None, survival_train = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.
    all_subject_ids = []
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_surv = []
    all_labels = []

    #subject_ids = loader.dataset.slides_radio_data['subject_id']
    #patient_results = {}
    subject_count = 0
    for batch_idx, (radio_features ,path_features, genomic_features, label, event_time, c,masks) in enumerate(loader):
        subject_id = loader.dataset.slides_radio_data['subject_id'][subject_count:(subject_count+len(label))]
        all_subject_ids.append(subject_id)
        subject_count += len(label)
        
        radio_features = radio_features.to(device)
        path_features = path_features.to(device)
        genomic_features = genomic_features.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            risk, hazards, S = model(h_radio=radio_features,h_path = path_features,h_omic = genomic_features )

        if isinstance(loss_fn ,CoxSurvLoss) or isinstance(loss_fn, RankingSurvLoss):
            loss = loss_fn(risks=risk, times =torch.tensor(event_time).to(device), c=c)

        elif isinstance(loss_fn , NLLSurvLoss):
            risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        elif isinstance(loss_fn , RankingNLLSurvLoss):
            risk = -torch.sum(S, dim=1)
            loss = loss_fn(hazards=hazards, risks =risk, S=S, Y=label, c=c)

        all_risk_scores.append(risk.detach().cpu().numpy().squeeze())
        all_censorships.append( c.detach().cpu().numpy())#.item()
        all_event_times.append(event_time)
        if S is not None:
            all_surv.append(S.detach().cpu().numpy())
        else:
            all_surv.append([np.nan])
        all_labels.append(label.detach().cpu().numpy())


    #import pdb; pdb.set_trace()
    all_risk_scores = np.concatenate(all_risk_scores).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_event_times = np.concatenate(all_event_times).flatten()
    all_censorships = np.concatenate(all_censorships).flatten()
    all_subject_ids = np.concatenate(all_subject_ids).flatten()
    all_surv = np.concatenate(all_surv,axis =0)#.flatten()


    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    if isinstance(loss_fn , NLLSurvLoss) or isinstance(loss_fn , RankingNLLSurvLoss):
        if survival_train is None:
            survival_train = Surv.from_arrays(1-all_censorships, all_event_times)    
            return survival_train

        all_event_times[all_event_times> max([j for i,j in survival_train]) ] = max([j for i,j in survival_train])
        survival_test = Surv.from_arrays(1-all_censorships, all_event_times)
        times = bins[1:]
        #print(times, all_event_times.min(),all_event_times.max())
        if times[0] <= all_event_times.min():
            times[0] = all_event_times.min()+ 0.001
        if times[-1] >= all_event_times.max():
            times[-1] = all_event_times.max() -0.001

        ibs = integrated_brier_score(survival_train , survival_test, all_surv, times)
    else:
        ibs = np.nan

    patient_results={'subject_id':all_subject_ids,'risk': all_risk_scores, 'disc_label': all_labels, 
        'survival': all_event_times, 'censorship': all_censorships,'prob':all_surv,'train':survival_train, 'times':times}


    return patient_results, c_index, ibs


class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)