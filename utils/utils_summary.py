import torch
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored
from lifelines.plotting import add_at_risk_counts


rows = ['RADIO','PATH','OMICS','RADIO_OMICS','RADIO_PATH','OMICS_PATH','RADIO_OMICS_PATH']
    
def clean_summary(summary):
    loss_function = []
    modalities = []
    concat = []
    a = []
    n = []
    for i in summary.index:
        if 'nll' in i and 'ranking' not in i:
            loss_function.append('nll')
        elif 'cox' in i:
            loss_function.append('cox')
            
        elif 'ranking' in i and 'nll' not in i:
            loss_function.append('ranking')  
            
        elif 'ranking' in i and 'nll' in i:
            loss_function.append('ranking-nll')    
        m = ''
        if 'RADIO' in i:
            m += 'RADIO_'
        if 'OMICS' in i:
            m += 'OMICS_'
        if 'PATH' in i:
            m += 'PATH_'

        modalities.append(m[:-1])
        

        if 'early-fcnn' in i:
            concat.append('early_fcnn')
        elif 'early-highway' in i:
            if 'nl4' in i:
                concat.append('early_highway_4')
            elif 'nl8' in i:
                concat.append('early_highway_8')
            elif 'nl1' in i:
                concat.append('early_highway_1')            
            else:
                concat.append(np.nan)                
        elif 'late-highway' in i:
            if 'nl4' in i:
                concat.append('late_highway_4')
            elif 'nl8' in i:
                concat.append('late_highway_8')
            else:
                concat.append('late_highway_1') 
        elif 'late-fcnn' in i:
            concat.append('late_fcnn')  
        elif 'fcnn' in i:
            concat.append('fcnn')
        elif 'highway' in i:
            if 'nl4' in i:
                concat.append('highway_4')
            elif 'nl8' in i:
                concat.append('highway_8')
            elif 'nl1' in i:
                concat.append('highway_1')   
        elif 'kronecker' in i:
            concat.append('kronecker')

    summary['loss'] = loss_function
    summary['modalities'] = modalities
    summary['concat'] = concat
    return summary

def summary_result(result_dir,which_c ='val_cindex',verbose = False):
    experiments = os.listdir(result_dir)
    experiments = [i for i in experiments if 'n8' not in i]
    experiments_c = {}
    for e in experiments:
        try:
            experiments_c[e] = pd.read_csv(os.path.join(result_dir,e,'summary.csv'))[which_c].values
        except:
            if verbose:
                print(e, 'not included')
    final_val = pd.DataFrame(experiments_c)
    final_val_long = pd.DataFrame(final_val.unstack()).reset_index().rename(columns = {'level_0':'modality','level_1':'cv',0:which_c})
    summary = final_val_long.groupby('modality')[which_c].agg(['mean','std'])
    summary = clean_summary(summary)
    summary.sort_values('mean')
    return final_val_long,summary


def result_plot(result_dir, which_c = 'val_cindex', overall_func = 'mean',k = 15, plot = True, verbose = False):
    final_val_long,summary = summary_result(result_dir,which_c,verbose)
    if plot:
        summary_graph = final_val_long.groupby('modality')[which_c].agg(['mean','std'])
        #summary_graph.sort_values('mean')
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,len(summary_graph)*0.3),sharey = True)
        sns.barplot(data = final_val_long, x=which_c , y ='modality',order = summary.sort_values('mean').index, ax= ax1)
        sns.boxplot(data = final_val_long, x=which_c, y = 'modality',order = summary.sort_values('mean').index ,ax = ax2)
        #sns.swarmplot(x="val_c_index", y="modality", data=final_val_long,order = summary.sort_values('mean').index ,ax = ax2, color=".2")
        plt.xticks(rotation = 90)
        fig.suptitle('10-Fold CV Val/Test C-index for All Experiments')
        plt.show()    
    
    brain_cox= pd.concat([summary[summary.loss.isin(['cox'])].pivot(index ='modalities',columns = 'concat',values = 'mean' )], axis =1,keys = ['cox_loss'])
    brain_ranking= pd.concat([summary[summary.loss.isin(['ranking'])].pivot(index ='modalities',columns = 'concat',values = 'mean' )], axis =1,keys = ['ranking_loss'])
    brain_nll= pd.concat([summary[summary.loss.isin(['nll'])].pivot(index ='modalities',columns = 'concat',values = 'mean' )], axis =1,keys = ['nll_loss'])
    brain_ranking_nll= pd.concat([summary[summary.loss.isin(['ranking-nll'])].pivot(index ='modalities',columns = 'concat',values = 'mean' )], axis =1,keys = ['ranking_nll_loss'])
    all_exp_tbl = pd.concat([brain_cox,brain_ranking,brain_nll,brain_ranking_nll],axis = 1).round(4)
    try:
        all_exp_tbl = all_exp_tbl.loc[rows,:]
    except:
        pass


    
    experiments = os.listdir(result_dir)
    summary['overall_c'] = np.nan
    for e in experiments:
        try:
            if 'val' in which_c:
                all_result = [pd.DataFrame(pd.read_pickle(os.path.join(result_dir,e,f'split_train_val_{i}_results.pkl'))) for i in range(k)]
            elif 'test' in which_c:
                all_result = [pd.DataFrame(pd.read_pickle(os.path.join(result_dir,e,f'split_train_test_{i}_results.pkl'))) for i in range(k)]
            a = pd.concat(all_result,axis = 0)
            if overall_func == 'mean':
                a = pd.DataFrame(a.groupby('subject_id').risk.mean()).reset_index().merge(a[['subject_id','censorship','survival']].drop_duplicates(),on = 'subject_id',how = 'left')
            elif overall_func == 'max':
                a = pd.DataFrame(a.groupby('subject_id').risk.max()).reset_index().merge(a[['subject_id','censorship','survival']].drop_duplicates(),on = 'subject_id',how = 'left')
            elif overall_func == 'median':
                a = pd.DataFrame(a.groupby('subject_id').risk.median()).reset_index().merge(a[['subject_id','censorship','survival']].drop_duplicates(),on = 'subject_id',how = 'left')
            #a = a.merge(dataset.slides_radio_data[['subject_id','disc_label']],on = 'subject_id',how = 'left')

            c = concordance_index_censored((1-a.censorship).astype(bool), a.survival, a.risk, tied_tol=1e-08)[0]
            if e in summary.index:
                summary.loc[e,'overall_c'] = c
                final_val_long.loc[final_val_long.modality == e,'overall_c'] = c
        except:
            pass
        
    brain_cox= pd.concat([summary[summary.loss.isin(['cox'])].pivot(index ='modalities',columns = 'concat',values = 'overall_c' )], axis =1,keys = ['cox_loss'])
    brain_ranking= pd.concat([summary[summary.loss.isin(['ranking'])].pivot(index ='modalities',columns = 'concat',values = 'overall_c' )], axis =1,keys = ['ranking_loss'])
    brain_nll= pd.concat([summary[summary.loss.isin(['nll'])].pivot(index ='modalities',columns = 'concat',values = 'overall_c' )], axis =1,keys = ['nll_loss'])
    brain_ranking_nll= pd.concat([summary[summary.loss.isin(['ranking-nll'])].pivot(index ='modalities',columns = 'concat',values = 'overall_c' )], axis =1,keys = ['ranking_nll_loss'])
    all_exp_tbl_overall = pd.concat([brain_cox,brain_ranking,brain_nll,brain_ranking_nll],axis = 1).round(4)
    try:
        all_exp_tbl_overall = all_exp_tbl_overall.loc[rows,:]#.T#
    except:
        pass
    
    if plot:
        summary_graph = summary.groupby('modality').overall_c.agg(['mean','std'])
        #summary_graph.sort_values('mean')
        fig, ax1 = plt.subplots(1,1,figsize=(4,len(summary_graph)*0.2),sharex = True)
        sns.barplot(data = final_val_long, x= 'overall_c', y ='modality',order = summary_graph.sort_values('mean').index, ax= ax1)
        plt.xticks(rotation = 90)
        fig.suptitle('Overall Val/Test C-index for All Experiments')
        plt.show()

    return final_val_long, summary, all_exp_tbl, all_exp_tbl_overall


def kmplot_orig(result_dir,which_c = 'val_cindex', overall_func = 'mean', k = 15, thresh=0.05):
    experiments = os.listdir(result_dir)
    for e in experiments:
        try:

            if 'val' in which_c:
                all_result = [pd.DataFrame(pd.read_pickle(os.path.join(result_dir,e,f'split_train_val_{i}_results.pkl'))) for i in range(k)]
            elif 'test' in which_c:
                all_result = [pd.DataFrame(pd.read_pickle(os.path.join(result_dir,e,f'split_train_test_{i}_results.pkl'))) for i in range(k)]
            a = pd.concat(all_result,axis = 0)
            if overall_func == 'mean':
                a = pd.DataFrame(a.groupby('subject_id').risk.mean()).reset_index().merge(a[['subject_id','censorship','survival']].drop_duplicates(),on = 'subject_id',how = 'left')
            elif overall_func == 'max':
                a = pd.DataFrame(a.groupby('subject_id').risk.max()).reset_index().merge(a[['subject_id','censorship','survival']].drop_duplicates(),on = 'subject_id',how = 'left')
            elif overall_func == 'median':
                a = pd.DataFrame(a.groupby('subject_id').risk.median()).reset_index().merge(a[['subject_id','censorship','survival']].drop_duplicates(),on = 'subject_id',how = 'left')

            high_surv =a[a.risk >=a.risk.median()].survival/12
            high_event = 1-a[a.risk >=a.risk.median()].censorship
            low_surv = a[a.risk < a.risk.median()].survival/12
            low_event =1-a[a.risk < a.risk.median()].censorship

            results = logrank_test(high_surv, low_surv, event_observed_A=high_event, event_observed_B=low_event)
            if results.p_value < thresh:
                #print(e)
                plt.figure()

                kmf = KaplanMeierFitter(label='high')
                kmf.fit(high_surv, high_event)
                kmf.plot()

                kmf = KaplanMeierFitter(label='low')
                kmf.fit(low_surv,low_event)
                kmf.plot()

                c = concordance_index_censored((1-a.censorship).astype(bool), a.survival, a.risk, tied_tol=1e-08)[0]
                plt.title(f'{e} \nlog-rank p-value {results.p_value:.4f}; overall c-index {c:.4f}')
                plt.xlabel('year')
                plt.ylabel('survival')

                plt.show()
        except Exception as ex:
            pass
            #print(e, 'not included')

def clean_table(all_exp_tbl,unimodal = ['RADIO','PATH','OMICS']):
    unimodal_tbl = all_exp_tbl.T[unimodal].dropna(how = 'all').T
    unimodal_tbl.columns = pd.MultiIndex.from_tuples([(i, 'early_'+j) for i, j in unimodal_tbl.columns])
    early_all_exp_tbl= all_exp_tbl.T.drop(unimodal,axis = 1).dropna(how = 'all').T
    early_all_exp_tbl = early_all_exp_tbl.iloc[:,[ 'early' in j for i, j in early_all_exp_tbl.columns]]
    early = pd.concat([unimodal_tbl,early_all_exp_tbl])#.style.highlight_max(color = 'lightgreen', axis = 0)

    unimodal_tbl = all_exp_tbl.T[unimodal].dropna(how = 'all').T
    unimodal_tbl.columns = pd.MultiIndex.from_tuples([(i, 'late_'+j) for i, j in unimodal_tbl.columns])
    late_all_exp_tbl= all_exp_tbl.T.drop(unimodal,axis = 1).dropna(how = 'all').T
    late_all_exp_tbl = late_all_exp_tbl.iloc[:,[ 'late' in j for i, j in late_all_exp_tbl.columns]]
    late = pd.concat([unimodal_tbl,late_all_exp_tbl])

    unimodal_tbl = all_exp_tbl.T[unimodal].dropna(how = 'all').T.xs('fcnn',axis =1,level = 1, drop_level=False)
    unimodal_tbl.columns = pd.MultiIndex.from_tuples([(i, 'kronecker') for i, j in unimodal_tbl.columns])
    kronecker_all_exp_tbl= all_exp_tbl.T.drop(unimodal,axis = 1).dropna(how = 'all').T
    kronecker_all_exp_tbl = kronecker_all_exp_tbl.iloc[:,[ 'kronecker' in j for i, j in kronecker_all_exp_tbl.columns]]
    kronecker = pd.concat([unimodal_tbl,kronecker_all_exp_tbl])

    final = pd.concat([early,late,kronecker],axis = 1)
    final = final.T.reset_index().sort_values(['level_0','level_1']).set_index(['level_0','level_1'])
    return final

def overall_cindex(result_dir, e, which_c , k, original_csv):
    k_result = [] 
    if 'val' in which_c:
        for cv in range(k):
            df = pd.DataFrame(pd.read_pickle(os.path.join(result_dir,e,f'split_train_val_{cv}_results.pkl'))) 
            df['cv'] = cv
            k_result.append(df)

    elif 'test' in which_c:
        #k_result = [pd.DataFrame(pd.read_pickle(os.path.join(result_dir,e,f'split_train_test_{i}_results.pkl'))) for i in range(k)]
        for cv in range(k):
            df = pd.DataFrame(pd.read_pickle(os.path.join(result_dir,e,f'split_train_test_{cv}_results.pkl'))) 
            df['cv'] = cv
            k_result.append(df)

    all_result_orig = pd.concat(k_result,axis = 0)
    all_result = pd.DataFrame(all_result_orig.groupby('subject_id').risk.mean()).reset_index().merge(all_result_orig[['subject_id','censorship','survival']].drop_duplicates(),on = 'subject_id',how = 'left')

    if 'gbmlgg' in result_dir:
        all_result_orig_gbm = all_result_orig[all_result_orig.subject_id.isin(original_csv[original_csv.oncotree_code == 'GBM'].subject_id.values)]
        all_result_orig_lgg = all_result_orig[all_result_orig.subject_id.isin(original_csv[original_csv.oncotree_code != 'GBM'].subject_id.values)]
        all_result_gbm = all_result[all_result.subject_id.isin(original_csv[original_csv.oncotree_code == 'GBM'].subject_id.values)]
        all_result_lgg = all_result[all_result.subject_id.isin(original_csv[original_csv.oncotree_code != 'GBM'].subject_id.values)]
        return all_result,all_result_gbm,all_result_lgg, all_result_orig, all_result_orig_gbm, all_result_orig_lgg
    elif 'lung' in result_dir:
        all_result_orig_luad = all_result_orig[all_result_orig.subject_id.isin(original_csv[original_csv.oncotree_code == 'LUAD'].subject_id.values)]
        all_result_orig_lusc = all_result_orig[all_result_orig.subject_id.isin(original_csv[original_csv.oncotree_code != 'LUAD'].subject_id.values)]
        all_result_luad = all_result[all_result.subject_id.isin(original_csv[original_csv.oncotree_code == 'LUAD'].subject_id.values)]
        all_result_lusc = all_result[all_result.subject_id.isin(original_csv[original_csv.oncotree_code != 'LUAD'].subject_id.values)]
        return all_result,all_result_luad,all_result_lusc, all_result_orig, all_result_orig_luad, all_result_orig_lusc
    

def kmplot(all_result_dict, e, thresh = None):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    fig, ax = plt.subplots(1,len(all_result_dict),figsize =  (len(all_result_dict) * 7 ,5),sharex =True, sharey = True )
    if len(all_result_dict) > 1:
        ax = ax.ravel()
    else:
        ax = [ax]
    
    for i ,(which_m, all_result)in enumerate(all_result_dict.items()):    
        high_surv =all_result[all_result.risk >=all_result.risk.median()].survival/12
        high_event = 1-all_result[all_result.risk >=all_result.risk.median()].censorship#.astype(bool)
        low_surv = all_result[all_result.risk < all_result.risk.median()].survival/12
        low_event =1-all_result[all_result.risk < all_result.risk.median()].censorship#.astype(bool)

        results = logrank_test(high_surv, low_surv, event_observed_A=high_event, event_observed_B=low_event)

        kmf_high = KaplanMeierFitter(label='high')
        kmf_high.fit(high_surv, high_event)
        kmf_high.plot(ax = ax[i], show_censors=True, ci_show = False, color = '#ff0000')

        kmf_low = KaplanMeierFitter(label='low')
        kmf_low.fit(low_surv,low_event)
        kmf_low.plot(ax = ax[i], show_censors=True, ci_show = False, color = '#0000ff')
        print(results)
        
        add_at_risk_counts(kmf_high, kmf_low, ax=ax[i])
        title = f'Log-Rank P-Value {results.p_value:.4f}'
        #title = f'{which_m}: log-rank p-value {results.p_value:.4f}'
        if results.p_value < thresh:
            title += '*'
        ax[i].set_title(title)
        ax[i].set_xlabel('Year')
        #ax[i].set_xlabel('')
        ax[i].set_ylabel('Survival')
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

        ax[i].yaxis.set_ticks_position('left')
        ax[i].xaxis.set_ticks_position('bottom')

    #.suptitle(e)
    plt.show()    

def pivot_summary(summary, which_col):
    brain_cox= pd.concat([summary[summary.loss.isin(['cox'])].pivot(index ='modalities',columns = 'concat',values = which_col )], axis =1,keys = ['cox_loss'])
    brain_ranking= pd.concat([summary[summary.loss.isin(['ranking'])].pivot(index ='modalities',columns = 'concat',values = which_col )], axis =1,keys = ['ranking_loss'])
    brain_nll= pd.concat([summary[summary.loss.isin(['nll'])].pivot(index ='modalities',columns = 'concat',values = which_col )], axis =1,keys = ['nll_loss'])
    brain_ranking_nll= pd.concat([summary[summary.loss.isin(['ranking-nll'])].pivot(index ='modalities',columns = 'concat',values = which_col)], axis =1,keys = ['ranking_nll_loss'])
    all_exp_tbl_overall = pd.concat([brain_cox,brain_ranking,brain_nll,brain_ranking_nll],axis = 1).round(4)
    try:
        all_exp_tbl_overall = all_exp_tbl_overall.loc[rows,:]#.T#
    except Exception as e:
        pass
    return all_exp_tbl_overall

def plot_bar(df,which_col ,sort_by):
    plt.figure(figsize = (12,10))
    df = df[which_col]
    df_new = df.stack().reset_index().rename(columns = {'level_0':'model','level_1':'cohort',0:'c-index'})
    sns.barplot(data = df_new, x='c-index', y = 'model', hue = 'cohort' ,
                order=df.sort_values(sort_by).index)
    fig.suptitle('10-Fold CV Val/Test C-index for All Experiments')
    plt.show()

