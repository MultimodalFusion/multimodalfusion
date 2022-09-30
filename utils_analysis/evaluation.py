# Base / Native
import math
import os
from os.path import join
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# Numerical / Array
import lifelines
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index as ci
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from PIL import Image
import pylab
import scipy
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import average_precision_score, auc, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from scipy import interp
import yaml

import os
from os.path import join
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import pandas as pd
from tqdm import tqdm

from scipy.stats import ttest_ind
#from tqdm.notebook import tqdm

### Aesthetics
def font_prop(size = 25, fname='post_process/assets/fonts/HelveticaNeue.ttf'):
    return fm.FontProperties(size = size, fname=fname)

def configure_matplotlib(plot_args):
    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = plot_args.border_thickness

def getConcatImage(imgs, how='horizontal', gap=0):
    from PIL import Image
    gap_dist = (len(imgs)-1)*gap
    
    if how == 'vertical':
        w, h = np.max([img.width for img in imgs]), np.sum([img.height for img in imgs])
        h += gap_dist
        curr_h = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))
        for img in imgs:
            dst.paste(img, (0, curr_h))
            curr_h += img.height + gap

    elif how == 'horizontal':
        w, h = np.sum([img.width for img in imgs]), np.min([img.height for img in imgs])
        w += gap_dist
        curr_w = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))

        for idx, img in enumerate(imgs):
            dst.paste(img, (curr_w, 0))
            curr_w += img.width + gap

    return dst


### Used for Evaluating Models + Creating KM Curves
def hazard2grade(hazard, p):
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)

def getResultsFromPKL(dataroot: str, exp_dir: str, test_idx: int, zscore: bool=True):
    from scipy import stats
    results_df = pd.DataFrame(pd.read_pickle(join(dataroot, exp_dir, 'split_%d_results.pkl' % test_idx))).T
    results_df = results_df.drop(['slide_id'], axis=1).astype(float)
    if zscore:
        results_df['risk'] = stats.zscore(results_df['risk'])
    return results_df

def getPValue_Binary(results_df: pd.DataFrame=None, risk_percentiles=[50]):
    p = np.percentile(results_df['risk'], risk_percentiles)
    results_df.insert(0, 'strat', [hazard2grade(risk, p) for risk in results_df['risk']])
    T_low, T_high = results_df['survival'][results_df['strat']==0], results_df['survival'][results_df['strat']==1]
    E_low, E_high = 1-results_df['censorship'][results_df['strat']==0], 1-results_df['censorship'][results_df['strat']==1]

    low_vs_high = logrank_test(durations_A=T_low, durations_B=T_high, event_observed_A=E_low, event_observed_B=E_high).p_value
    return np.array([low_vs_high])

def getPValue_25_75(results_df: pd.DataFrame=None, risk_percentiles=[25, 50, 75]):
    p = np.percentile(results_df['risk'], risk_percentiles)
    if p[0] == -4.0:
            p[0] += 1e-7
    results_df.insert(0, 'strat', [hazard2grade(risk, p) for risk in results_df['risk']])
    T_low, T_high = results_df['survival'][results_df['strat']==0], results_df['survival'][results_df['strat']==3]
    E_low, E_high = 1-results_df['censorship'][results_df['strat']==0], 1-results_df['censorship'][results_df['strat']==3]

    low_vs_high = logrank_test(durations_A=T_low, durations_B=T_high, event_observed_A=E_low, event_observed_B=E_high).p_value
    return np.array([low_vs_high])


def makeHazardHistogram(results_df: pd.DataFrame=None, save_path: str=None, show=False, name=None,
                        cutoff: float=0, zscore=True, bins=15, norm=True):
    ### Z-Scoring Risks
    if zscore: results_df['risk'] = scipy.stats.zscore(np.array(results_df['risk']))
    if cutoff == 0:
        cutoff = results_df[results_df['censorship']==0]['survival'].median() / 12
        
    ### Plotting
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    low = results_df[results_df['survival'] <= 12*cutoff]
    high = results_df[results_df['survival'] > 12*cutoff]
    low = low[low['censorship'] == 0]
    high = high[high['censorship'] == 0]
    sns.distplot(low['risk'], bins=bins, kde=False, norm_hist=norm, 
                 hist_kws={'histtype':'stepfilled', "linewidth": 1, "alpha": 0.5, "color": "r"}, ax=ax)
    sns.distplot(high['risk'], bins=bins, kde=False, norm_hist=norm, 
                 hist_kws={'histtype':'stepfilled', "linewidth": 1, "alpha": 0.5, "color": "b"}, ax=ax)
    
    ### Aesthetics
    import matplotlib.patches as mpatches 
    
    font_args = configure_font()
    ax.set_xlabel('Hazard (Z-Scored)', fontproperties=font_args['ax_label_fprop'])
    ax.set_ylabel('Density (Normalized)', fontproperties=font_args['ax_label_fprop'])
    #ax.set_title(title, fontproperties=font_args['title_fprop']) # No title at the moment
    plt.setp(ax.get_xticklabels(), fontproperties =font_args['ax_tick_label_fprop'])
    plt.setp(ax.get_yticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
    
    colors = sns.color_palette(palette = ["mediumslateblue" , "salmon"])
    texts = ["Patient Survival < %0.1f" % cutoff, "Patient Survival > %0.1f" % cutoff]
    patches = [mpatches.Patch(color=colors[i], label="{:s}".format(texts[i])) for i in range(len(texts))]
    legend = ax.legend(handles=patches, loc='upper right', prop=font_args['legend_fprop'])
    
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.savefig(join(save_path, '%s_hist.png' % name))
    
    if not show:
        plt.close()

def CI_pm(data, confidence=0.95):
    import scipy
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h
    #return str("{0:.4f} ± ".format(m) + "{0:.3f}".format(h))

def get_results_per_experiment(dataroot:str='./results/mixed_testing/', which_summary:str=''):
    results_df = []
    #print(dataroot)
    for study_name in sorted(os.listdir(dataroot)):
        #print(study_name)
        if os.path.isdir(join(dataroot, study_name)):
            tcga_proj = study_name.split('_')[1]
            summary = pd.read_csv(join(dataroot, study_name, 'summary%s.csv' % which_summary), 
                                  index_col=0, usecols=[1,2])
            val_cindex_mean = summary['val_cindex'].mean()
            val_cindex_std = np.std(summary['val_cindex'])
            val_cindex_ci = CI_pm(np.array(summary['val_cindex']))
            val_cindex_max = summary['val_cindex'].max()
            val_argmax = summary['val_cindex'].argmax()
            results_df.append([tcga_proj, val_cindex_mean, val_cindex_std, val_cindex_ci, val_cindex_max, val_argmax])

    results_df = pd.DataFrame(results_df, columns=['Project', 'Val C-Index (Mean)', 'Val C-Index (STD)',
                                                   'Val C-Index (CI)', 'Val C-Index (Max)', 'val_idx'])
    results_df.index = results_df['Project']
    return results_df

def getResultsFromPKL(dataroot: str, study_dir: str, split: str, idx: int, which_summary:str=''):
    results_df = pd.DataFrame(pd.read_pickle(join(dataroot, study_dir, 'split_train_%s_%d_results.pkl' % 
                                                  ( split, idx)))).T
    #print(results_df)
    #results_df = results_df.drop(['slide_id'], axis=1).astype(float)
    return results_df


"""
def makeKaplanMeierPlot_Strat(results_df: pd.DataFrame=None, risk_percentiles=[50],
                              save_path: str=None, name: str=None, pval: float=None, 
                              x_label: bool=True, y_label: bool=False, title=True, label_size=40, multiplier=1):
    ### Calculating Risk Percentiles
    p = np.percentile(results_df['risk'], risk_percentiles)
    
    if p[0] == -4.0:
        p[0] += 1e-7
    
    results_df.insert(0, 'strat', [hazard2grade(risk, p) for risk in results_df['risk']])
    kmf_pred = lifelines.KaplanMeierFitter()

    ### Plotting
    fig, ax = plt.subplots(figsize=(8,8), dpi=50*multiplier)
    censor_style = {'ms': 20, 'marker': '+'}
    temp = results_df[results_df['strat']==0]
    if temp.shape[0] == 0:
        temp = results_df
    kmf_pred.fit(temp['survival']/12, 1-temp['censorship'], label="Low Risk")
    kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=3, ls='-', 
                  markerfacecolor='black', censor_styles=censor_style)
    
    temp = results_df[results_df['strat']==1]
    if temp.shape[0] == 0:
        temp = results_df
    kmf_pred.fit(temp['survival']/12, 1-temp['censorship'], label="High Risk")
    kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=3, ls='-', 
                  markerfacecolor='black', censor_styles=censor_style)
    
    ax.set_xlim(0, max(results_df['survival'])/12+0.1)
    ax.set_xticks(np.arange(0.0, max(results_df['survival'])/12+0.001, max(results_df['survival'])/12/4))
    
    
    ax.set_ylim(0.0, 1+0.1)
    ax.set_yticks(np.arange(0.2, 1.001, 0.2))
    
    ### Aesthetics
    font_args = configure_font(label_size=label_size)
    
    if x_label:
        ax.xaxis.label.set_visible(True)
        ax.set_xlabel('Time (Years)', fontproperties=font_args['ax_label_fprop'])
    else:
        ax.xaxis.label.set_visible(False)
        
    if y_label:
        ax.yaxis.label.set_visible(True)
        ax.set_ylabel('Proportion Surviving', fontproperties=font_args['ax_label_fprop'])
        #plt.tight_layout()
    else:
        ax.yaxis.label.set_visible(False)
    
    #ax.set_xlabel('Time (Years)', fontproperties=font_args['ax_label_fprop'])
    #ax.set_ylabel('Proportion Surviving', fontproperties=font_args['ax_label_fprop'])
    #ax.set_title(title, fontproperties=font_args['title_fprop']) # No title at the moment
    study = name.split('_')[1]
    if title:
        ax.set_title(study.upper(), fontproperties=font_args['title_fprop'])
    else:
        ax.set_title('')
        x_ticks = np.arange(0.0, max(results_df['survival'])/12+0.0001, max(results_df['survival'])/12/4)
        diff = np.round(np.diff(x_ticks).mean())
        x_ticks = [diff*i for i in range(len(x_ticks))]
        ax.set_xticks(np.array([int(t) for t in x_ticks]))
        
    plt.setp(ax.get_xticklabels(), fontproperties =font_args['ax_tick_label_fprop'])
    plt.setp(ax.get_yticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
    legend = ax.legend(loc="upper right", prop=font_args['legend_fprop'])
    ax.get_legend().remove()
    
    pval = 1e-7 if np.isnan(pval) else pval
    
    sig = '(*)' if pval < 0.05 else ''
    ax.annotate(s='P-Value: %01.2e %s' % (pval, sig), xy=(135*multiplier,330*multiplier), xycoords='figure pixels', fontproperties=font_args['pval_fprop'])
    
    strat_code = '_'.join([str(r) for r in risk_percentiles])
    plt.savefig(join(save_path, name))#, bbox_inches='tight')
    
    plt.close()
"""
def makeKaplanMeierPlot_Strat(results_df: pd.DataFrame=None, risk_percentiles=[50],
                              save_path: str=None, name: str=None, pval: float=None, 
                              x_label: bool=True, y_label: bool=False, title=True, label_size=40, 
                              multiplier=1,exp_name=''):
    ### Calculating Risk Percentiles
    p = np.percentile(results_df['risk'], risk_percentiles)
    
    if p[0] == -4.0:
        p[0] += 1e-7
    
    results_df.insert(0, 'strat', [hazard2grade(risk, p) for risk in results_df['risk']])
    kmf_pred = lifelines.KaplanMeierFitter()

    ### Plotting
    fig, ax = plt.subplots(figsize=(8,8), dpi=50*multiplier)
    censor_style = {'ms': 20, 'marker': '+'}
    temp = results_df[results_df['strat']==0]
    if temp.shape[0] == 0:
        temp = results_df
    kmf_pred.fit(temp['survival']/12, 1-temp['censorship'], label="Low Risk")
    kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='b', linewidth=3, ls='-', 
                  markerfacecolor='black', censor_styles=censor_style)
    
    temp = results_df[results_df['strat']==1]
    if temp.shape[0] == 0:
        temp = results_df
    kmf_pred.fit(temp['survival']/12, 1-temp['censorship'], label="High Risk")
    kmf_pred.plot(ax=ax, show_censors=True, ci_show=False, c='r', linewidth=3, ls='-', 
                  markerfacecolor='black', censor_styles=censor_style)
    
    ax.set_xlim(0, max(results_df['survival'])/12+0.1)
    ax.set_xticks(np.arange(0.0, max(results_df['survival'])/12+0.001, max(results_df['survival'])/12/4))
    
    
    ax.set_ylim(0.0, 1+0.1)
    ax.set_yticks(np.arange(0.2, 1.001, 0.2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ### Aesthetics
    font_args = configure_font(label_size=label_size)
    
    if x_label:
        ax.xaxis.label.set_visible(True)
        ax.set_xlabel('Time (Years)', fontproperties=font_args['ax_label_fprop'])
    else:
        ax.xaxis.label.set_visible(False)
        
    if y_label:
        ax.yaxis.label.set_visible(True)
        ax.set_ylabel('Proportion Surviving', fontproperties=font_args['ax_label_fprop'])
        #plt.tight_layout()
    else:
        ax.yaxis.label.set_visible(False)
    
    ax.set_xlabel('Time (Years)', fontproperties=font_args['ax_label_fprop'])
    ax.set_ylabel('Proportion Surviving', fontproperties=font_args['ax_label_fprop'])
    #ax.set_title(exp_name, fontproperties=font_args['title_fprop']) # No title at the moment
    study = name.split('_')[1]
    if title:
        ax.set_title(exp_name, fontproperties=font_args['title_fprop'],fontsize = 25)
    else:
        ax.set_title('')
        
    x_ticks = np.arange(0.0, max(results_df['survival'])/12+0.0001, max(results_df['survival'])/12/4)
    diff = np.round(np.diff(x_ticks).mean())
    x_ticks = [diff*i for i in range(len(x_ticks))]
    ax.set_xticks(np.array([int(t) for t in x_ticks]))
        
    plt.setp(ax.get_xticklabels(), fontproperties =font_args['ax_tick_label_fprop'])
    plt.setp(ax.get_yticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
    ax.legend(loc="upper right", prop=font_args['legend_fprop'])
    #legend = ax.legend(loc="upper right", prop=font_args['legend_fprop'])
    #ax.get_legend().remove()
    
    pval = 1e-7 if np.isnan(pval) else pval
    
    sig = '(*)' if pval < 0.05 else ''
    ax.annotate(s='P-Value: %01.2e %s' % (pval, sig), xy=(135*multiplier,330*multiplier), 
                xycoords='figure pixels', fontproperties=font_args['pval_fprop'])
    
    strat_code = '_'.join([str(r) for r in risk_percentiles])
    plt.savefig(join(save_path, name))#, bbox_inches='tight')
    
    plt.close()

def configure_font(label_size = 40, font='post_process/assets/fonts/HelveticaNeue.ttf'):
    font_args = dict()
    font_args['ax_label_fprop']= font_prop(label_size, font)
    font_args['ax_tick_label_fprop'] = font_prop(int(label_size * 0.75), font)
    font_args['title_fprop'] = font_prop(label_size, font)
    font_args['zoom_ax_tick_label_fprop'] = font_prop(int(label_size * 0.6), font)
    font_args['legend_fprop'] = font_prop(int(label_size * 0.65), font)
    font_args['pval_fprop'] = font_prop(int(label_size * 0.6), font)
    font_args['exptitle_fprop'] = font_prop(int(label_size * 1.5), font)
    font_args['shap_fprop'] = font_prop(int(label_size * 0.5), font)
    return font_args

"""
def summarize_test(dataroot: str, exp_dir: str, which_summary: str, 
                   save_path:str, risk_percentiles: list=[50], verbose=False, 
                   exp_name=None, create_legend=True, multiplier=1, skip_km=True):
    
    dataroot = join(dataroot, exp_dir)
    results_cindex_df = get_results_per_experiment(dataroot=dataroot, which_summary=which_summary)
    cindex_df, pval_df = {}, {}
    ci_df = {}
    cindex_boot_df = []
    
    cindex_std_df = []
    
    figures = []

    for idx, study_dir in tqdm(enumerate(sorted(os.listdir(dataroot)))):
        if os.path.isdir(join(dataroot, study_dir)):
            proj = study_dir.split('_')[1]
            results_df = [getResultsFromPKL(dataroot=dataroot, study_dir=study_dir,
                                            split='val', idx=i,which_summary=which_summary) 
                          for i in range(5)]
            for split in results_df:
                cin = ci(event_times=split['survival'], 
                         predicted_scores=-1*split['risk'],
                         event_observed=1-split['censorship'])
                cindex_std_df.append([cin, proj.upper(), exp_dir[:3]])
                
            results_df = pd.concat(results_df)
            results_df['case_id'] = results_df.index
            results_df = results_df.groupby('case_id').mean()
            cindex_df[proj] = np.array([ci(event_times=results_df['survival'], 
                                           predicted_scores=-1*results_df['risk'],
                                           event_observed=1-results_df['censorship'])])
            study = "_".join(study_dir.split('_')[:2])
            cindex_boot = boot_cindex(results_df)
            cindex_boot_df.append(pd.DataFrame([cindex_boot,
                                                [proj.upper()]*1000,
                                                [exp_dir[:3]]*1000]).T)
            ci_df[proj] = [CI_pm(cindex_boot)]
            
            
            #if proj.upper() in ['COADREAD', 'HNSC', 'SKCM']:
            #    risk_percentiles = [25, 50, 75]
            #else:
            #    risk_percentiles = [25, 50, 75]
                
            #risk_percentiles = 
            #print(risk_percentiles)
                            
            if risk_percentiles == [50]:
                pval_df[proj] = getPValue_Binary(results_df=results_df.copy(), risk_percentiles=risk_percentiles)
            else:
                pval_df[proj] = getPValue_25_75(results_df=results_df.copy(), risk_percentiles=risk_percentiles)
            
            save_path_study = save_path
            os.makedirs(save_path_study, exist_ok=True)
            #print(pval_df[proj])
            strat_code = '_'.join([str(r) for r in risk_percentiles])
            name = 'tcga_%s_%s_val_km_%s.png' % (proj, exp_dir[:3], strat_code)
            if 'MMF' in exp_dir and idx > 6:
                set_x_label = True
            else:
                set_x_label = False
            name = 'tcga_%s_%s_val_km_%s_subplot.png' % (proj, exp_dir[:3], strat_code)
            makeKaplanMeierPlot_Strat(results_df=results_df.copy(), save_path=save_path_study, 
                                      name=name, risk_percentiles=risk_percentiles, pval=pval_df[proj][0], x_label=set_x_label, y_label=False, multiplier=multiplier)
            figures.append(Image.open(join(save_path_study, name)))
            name = 'tcga_%s_%s_val_km_%s_HQ.png' % (proj, exp_dir[:3], strat_code)
            makeKaplanMeierPlot_Strat(results_df=results_df.copy(), save_path=save_path_study, 
                                      name=name, risk_percentiles=risk_percentiles, pval=pval_df[proj][0], x_label=True, y_label=True, label_size=30, multiplier=multiplier)
            name = 'tcga_%s_%s_val_km_%s_HQE.png' % (proj, exp_dir[:3], strat_code)
            makeKaplanMeierPlot_Strat(results_df=results_df.copy(), save_path=save_path_study, 
                                      name=name, risk_percentiles=risk_percentiles, pval=pval_df[proj][0], x_label=False, y_label=False, title=False, 
                                      label_size=40, multiplier=multiplier*3)
 
    #return results_cindex_df, cindex_df, pval_df
    cindex_df = pd.DataFrame(cindex_df).T.sort_index()
    cindex_df.columns = ['C-Index (All)']
    pval_df = pd.DataFrame(pval_df).T.sort_index()
    pval_df.columns = ['P-Value']
    ci_df = pd.DataFrame(ci_df).T.sort_index()
    ci_df.columns = ['95% CI']
    results_df = results_cindex_df.join(cindex_df).join(pval_df).join(ci_df)
    if verbose:
        display(results_df)
        display(results_df.mean())
        display(pval_df.T < 0.05)
        print("Significant Studies:", (pval_df < 0.05).sum())
        
    cindex_boot_df = pd.concat(cindex_boot_df)
    cindex_boot_df.columns = ['C-Index', 'Study', 'Model']    
    cindex_std_df = pd.DataFrame(cindex_std_df)
    cindex_std_df.columns = ['C-Index', 'Study', 'Model']
    
    if skip_km:
        return results_df, None, cindex_boot_df, cindex_std_df
        
        
    ### Figure Creation
    font_args = configure_font(label_size=20)
    
    # Concatenating Images
    top = getConcatImage(figures[:7])
    bottom = getConcatImage(figures[7:])
    main = getConcatImage([top, bottom], how='vertical')
    
    # Creating Y-Label
    save_path_ylabel = join(save_path, 'ylabel.png')
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100*multiplier)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False) #remove ticks
    plt.box(False) #remove box
    ax.yaxis.label.set_visible(True)
    ax.set_ylabel('Proportion Surviving', fontproperties=font_args['ax_label_fprop'])
    fig.savefig(save_path_ylabel)
    plt.close()
    ylabel = Image.open(save_path_ylabel)
    ylabel = ylabel.crop((0, 0, 50*multiplier, 400*multiplier))
    ylabel = getConcatImage([ylabel, ylabel], how='vertical')

    ### 3. Creating Legend
    import matplotlib.patches as mpatches
    save_path_legend = join(save_path, 'legend.png')
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100*multiplier)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False) #remove ticks
    plt.box(False) #remove box
    red_patch = mpatches.Patch(color='#e64e5b', label='Label1')
    blue_patch = mpatches.Patch(color='#9381ff', label='Label2')
    ax.legend(handles=[red_patch, 
                       blue_patch], 
              labels=['High Risk \n (Top 25% Risk Percentile)', 
                      'Low Risk \n (Bot 25% Risk Percentile)'], 
              prop=font_args['legend_fprop'])
    
    fig.savefig(save_path_legend)
    plt.close()
    legend = Image.open(save_path_legend)
    X = 50*multiplier
    legend = legend.crop((X, 0, 400*multiplier, 400*multiplier))
    empty = Image.new('RGB', (400*multiplier-X, 400*multiplier), color=(255,)*3)
    if create_legend:
        legend = getConcatImage([legend, empty], how='vertical')
    else:
        legend = getConcatImage([empty, empty], how='vertical')
        
    ### 4. Creating EXP_Label
    if len(exp_name) < 20:
        save_path_exp = join(save_path, '%s_explabel.png' % exp_dir[:3].lower())
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100*multiplier)
        plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False) #remove ticks
        plt.box(False) #remove box
        ax.yaxis.label.set_visible(True)
        ax.set_ylabel(exp_name, fontproperties=font_args['exptitle_fprop'])
        #plt.gcf().subplots_adjust(left=0.20)
        fig.savefig(save_path_exp)
        plt.close()
        explabel = Image.open(save_path_exp)
        X = 50*multiplier
        explabel = explabel.crop((0, 0, X, 400*multiplier))
        empty = Image.new('RGB', (X, 200*multiplier), color=(255,)*3)
        explabel = getConcatImage([empty, explabel, empty], how='vertical')
    else:
        save_path_exp = join(save_path, '%s_explabel.png' % exp_dir[:3].lower())
        fig, ax = plt.subplots(figsize=(4, 8), dpi=100)
        plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False) #remove ticks
        plt.box(False) #remove box
        ax.yaxis.label.set_visible(True)
        ax.set_ylabel(exp_name, fontproperties=font_args['exptitle_fprop'])
        #plt.gcf().subplots_adjust(left=0.20)
        fig.savefig(save_path_exp)
        plt.close()
        explabel = Image.open(save_path_exp)
        X = 50*multiplier
        explabel = explabel.crop((0, 0, X, 800*multiplier))

    km_png = getConcatImage([explabel, ylabel, main])
    if create_legend:
        km_png = getConcatImage([km_png, legend])
    
    
    return results_df, km_png, cindex_boot_df, cindex_std_df
"""
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw, concordance_index_censored

def survival_AUC(df1, df2, times=None, method='hc_sksurv', tied_tol=1e-5):
    df1['censorship'] = 1-df1['censorship']
    df2['censorship'] = 1-df2['censorship']
    
    c_tie = concordance_index_censored(np.array(df2['censorship'], bool), 
                                       np.array(df2['survival']), 
                                       np.array(df2['risk']), tied_tol=tied_tol)
    
    
    df2 = df2[~(df2['survival'] > df1['survival_months'].max())]
    #times = np.percentile(df2['survival'], np.linspace(5, 81, 15))
    
    surv1 = np.array(df1, dtype=int)
    risk2 = np.array(df2['risk'])
    surv2 = np.array(df2.drop(['subject_id','risk'], axis=1), dtype=int)

    surv1 = np.core.records.fromarrays(surv1[:, [1,0]].transpose(), names='obs, survival_months', formats = '?, i8')
    surv2 = np.core.records.fromarrays(surv2[:, [1,0]].transpose(), names='obs, survival_months', formats = '?, i8')
    _, iauc = cumulative_dynamic_auc(surv1, surv2, risk2, times)
    ipwc = concordance_index_ipcw(surv1, surv2, risk2, times[-1])
    return iauc, ipwc[0], c_tie[0]

def summarize_test(dataroot: str, exp_dir: str, which_summary: str, 
                   save_path:str, risk_percentiles: list=[50], verbose=False, 
                   exp_name=None, create_legend=True, multiplier=1, skip_km=True, grade_df=None):
    
    dataroot = join(dataroot, exp_dir)
    results_cindex_df = get_results_per_experiment(dataroot=dataroot, which_summary=which_summary)
    cindex_df, pval_df = {}, {}
    ci_df = {}
    cindex_boot_df = []
    cindex_std_df = []
    figures = []

    for idx1, study_dir in tqdm(enumerate(sorted(os.listdir(dataroot)))):
        
        if os.path.isdir(join(dataroot, study_dir)):
            proj = study_dir.split('_')[1]
            clin_df = pd.read_csv('./dataset_csv/tcga_gbmlgg_survival_169_allmodalities.csv')
            clin_df.index = clin_df['subject_id']
            censor_rate = clin_df['censorship'].sum() / clin_df['censorship'].shape[0]
            tau = math.ceil(censor_rate*10)*10+1
            #print('Tau', tau)
            times = np.percentile(clin_df['survival_months'], np.linspace(15, tau, 30))
                        
            results_df = [getResultsFromPKL(dataroot=dataroot, study_dir=study_dir,
                                            split='val', idx=i, which_summary=which_summary) 
                          for i in range(5)]
                        
            for idx, val_pred_df in enumerate(results_df):
                split_i = pd.read_csv(os.path.join('./splits/10foldcv_169/', 'splits_%d.csv' % idx))

                train_pred_df = clin_df.loc[split_i['train']][['survival_months', 'censorship']]
                
                cin = ci(event_times=val_pred_df['survival'], 
                         predicted_scores=-1*val_pred_df['risk'],
                         event_observed=1-val_pred_df['censorship'])
                iauc, ipwc, cin_tie = survival_AUC(train_pred_df, val_pred_df.drop('disc_label', axis=1), times=times)
                #print(iauc)
                cindex_std_df.append([cin, iauc, ipwc, cin_tie, proj.lower(), exp_dir[:3]])
                
            results_df_all = pd.concat(results_df)
            
            results_df_all['subject_id'] = results_df_all.index
            cols=[i for i in results_df_all.columns if i not in ["subject_id"]]
            for col in cols:
                results_df_all[col]=results_df_all[col].astype(float)
            results_df_all = results_df_all.groupby('subject_id').mean()
            cindex_df[proj] = np.array([ci(event_times=results_df_all['survival'], 
                                           predicted_scores=-1*results_df_all['risk'],
                                           event_observed=1-results_df_all['censorship'])])
            study = "_".join(study_dir.split('_')[:2])
            cindex_boot = boot_cindex(results_df_all)
            cindex_boot_df.append(pd.DataFrame([cindex_boot,
                                                [proj.upper()]*1000,
                                                [exp_dir[:3]]*1000]).T)
            ci_df[proj] = [CI_pm(cindex_boot)]
                            
            if risk_percentiles == [50]:
                pval_df[proj] = getPValue_Binary(results_df=results_df_all.copy(), risk_percentiles=risk_percentiles)
            else:
                pval_df[proj] = getPValue_25_75(results_df=results_df_all.copy(), risk_percentiles=risk_percentiles)
            
            save_path_study = save_path
            os.makedirs(save_path_study, exist_ok=True)

            #print(pval_df[proj])
            strat_code = '_'.join([str(r) for r in risk_percentiles])
            name = 'tcga_%s_%s_val_km_%s.png' % (proj, exp_dir[:3], strat_code)
            if 'MMF' in exp_dir and idx1 > 6:
                set_x_label = True
            else:
                set_x_label = False
                
            if skip_km:
                continue

            #name = 'tcga_%s_%s_val_km_%s_subplot.png' % (proj, exp_dir[:3], strat_code)
            #makeKaplanMeierPlot_Strat(results_df=results_df_all.copy(), save_path=save_path_study, 
            #                          name=name, risk_percentiles=risk_percentiles, pval=pval_df[proj][0], 
            #                          x_label=set_x_label, 
            #                          y_label=False, multiplier=multiplier)
            #figures.append(Image.open(join(save_path_study, name)))
            name = 'tcga_%s_%s_val_km_%s_HQ.png' % (proj, exp_dir[:3], strat_code)
            makeKaplanMeierPlot_Strat(results_df=results_df_all.copy(), save_path=save_path_study, 
                                      name=name, risk_percentiles=risk_percentiles, pval=pval_df[proj][0], 
                                      x_label=True, y_label=True, label_size=30, multiplier=multiplier,exp_name=exp_name)
            name = 'tcga_%s_%s_val_km_%s_HQE.png' % (proj, exp_dir[:3], strat_code)
            #makeKaplanMeierPlot_Strat(results_df=results_df_all.copy(), save_path=save_path_study, 
            #                          name=name, risk_percentiles=risk_percentiles, pval=pval_df[proj][0], 
            #                          x_label=False, y_label=False, title=False, 
            #                          label_size=40, multiplier=multiplier*3)
            
    cindex_df = pd.DataFrame(cindex_df).T.sort_index()
    cindex_df.columns = ['C-Index (All)']
    pval_df = pd.DataFrame(pval_df).T.sort_index()
    pval_df.columns = ['P-Value']
    ci_df = pd.DataFrame(ci_df).T.sort_index()
    ci_df.columns = ['95% CI']
    results_df = results_cindex_df.join(cindex_df).join(pval_df).join(ci_df)
    
    cindex_boot_df = pd.concat(cindex_boot_df)
    cindex_boot_df.columns = ['C-Index', 'Study', 'Model']    
    cindex_std_df = pd.DataFrame(cindex_std_df)
    cindex_std_df.columns = ['C-Index', 'I-AUC', 'IPWC', 'C-Tie', 'Project', 'Model']
    
    results_df['Val I-AUC (Mean)'] = cindex_std_df.groupby('Project').mean()['I-AUC']
    results_df['Val I-AUC (STD)'] = cindex_std_df.groupby('Project').std()['I-AUC']
    results_df['Val IPWC (Mean)'] = cindex_std_df.groupby('Project').mean()['IPWC']
    results_df['Val IPWC (STD)'] = cindex_std_df.groupby('Project').std()['IPWC']
    results_df['Val C-Tie (Mean)'] = cindex_std_df.groupby('Project').mean()['C-Tie']
    results_df['Val I-Tie (STD)'] = cindex_std_df.groupby('Project').std()['C-Tie']

    if verbose:
        display(results_df)
        display(results_df.mean())
        display(pval_df.T < 0.05)
        print("Significant Studies:", (pval_df < 0.05).sum())
    
    if skip_km:
        return results_df, None, cindex_boot_df, cindex_std_df

    km_png = None
    return results_df, km_png, cindex_boot_df, cindex_std_df

def boot_cindex(results_df):
    bootci = []
    for seed in range(1000):
        bootstrap = results_df.sample(n=results_df.shape[0], replace=True, random_state=seed)
        bootci.append(ci(event_times=bootstrap['survival'], 
                         predicted_scores=-1*bootstrap['risk'],
                         event_observed=1-bootstrap['censorship']))
    return np.array(bootci)

def CI_pm(data, confidence=0.95):
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(data, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(data, p))
    return '(%0.3f-%0.3f)' % (lower, upper)


###
def get_all_ids(dataroot:str, study=str):
    csv_path = os.path.join(dataroot, '%s_all_clean.csv' % study)
    slide_data = pd.read_csv(csv_path, index_col=0, low_memory=False)
    slide_data.index = slide_data.index.str[:12]

    if 'case_id' not in slide_data:
        slide_data['case_id'] = slide_data.index
        slide_data = slide_data.reset_index(drop=True)

    return slide_data

def generate_heatmap_yamls(results_df, exp_dir, model_type, generate_best=False):
    data_dir = '/media/hdd1/awesomePhD/Datasets/pan-cancer/TCGA-WSIs/tcga_%s'
    save_config_dir = './heatmaps_revision/5foldcv/%s/configs' % exp_dir
    raw_save_dir = '/media/hdd2/heatmaps_revision/5foldcv/%s/heatmap_raw_results' % exp_dir
    production_save_dir = '/media/hdd2/heatmaps_revision/5foldcv/%s/heatmap_production_results' % exp_dir
    os.makedirs(save_config_dir, exist_ok=True)
    
    for vis in ['20x', '40x']:
        for study in results_df.index:

            if study in ['kirc', 'kirp']:
                case_folder = 'kidney'
            elif study in ['luad', 'lusc']:
                case_folder = 'lung'
            else:
                case_folder = study

            if generate_best:
                idx = results_df.loc[study]['val_idx']
                config_dict = yaml.safe_load(open('./heatmaps_revision/template_%s.yaml' % vis, 'r'))
                config_dict['exp_arguments']['exp'] = '5foldcv/%s' % exp_dir
                config_dict['exp_arguments']['save_exp_code'] = 'tcga_%s_%s' % (study, vis)
                config_dict['exp_arguments']['raw_save_dir'] = raw_save_dir
                config_dict['exp_arguments']['production_save_dir'] = production_save_dir
                config_dict['data_arguments']['process_list'] = 'tcga_%s_val_%d_%s.csv' % (study, idx, vis)
                config_dict['data_arguments']['data_dir'] = data_dir % case_folder
                config_dict['model_arguments']['ckpt_path'] = 'results_revision/5foldcv/%s/tcga_%s_%s_s1/s_%d_latest_checkpoint.pt' % (exp_dir, study, exp_dir, idx)
                config_dict['model_arguments']['model_type'] = model_type
                with open('%s/heatmap_config_%s_val_%d_%s.yaml' % (save_config_dir, study, idx, vis), 'w') as outfile:
                    yaml.dump(config_dict, outfile, default_flow_style=False, sort_keys=False)
            else:
                for idx in range(5):
                    if study == 'kirp':
                        if idx == 2 and vis == '20x': continue
                    elif study == 'lihc':
                        if idx == 2 and vis == '20x': continue
                    elif study == 'paad':
                        if idx in [1, 2, 4] and vis == '20x': continue

                    config_dict = yaml.safe_load(open('./heatmaps_revision/template_%s.yaml' % vis, 'r'))
                    config_dict['exp_arguments']['exp'] = '5foldcv/%s' % exp_dir
                    config_dict['exp_arguments']['save_exp_code'] = 'tcga_%s_%s' % (study, vis)
                    config_dict['exp_arguments']['raw_save_dir'] = raw_save_dir
                    config_dict['exp_arguments']['production_save_dir'] = production_save_dir
                    config_dict['data_arguments']['process_list'] = 'tcga_%s_val_%d_%s.csv' % (study, idx, vis)
                    config_dict['data_arguments']['data_dir'] = data_dir % case_folder
                    config_dict['model_arguments']['ckpt_path'] = 'results_revision/5foldcv/%s/tcga_%s_%s_s1/s_%d_latest_checkpoint.pt' % (exp_dir, study, exp_dir, idx)
                    config_dict['model_arguments']['model_type'] = model_type
                    with open('%s/heatmap_config_%s_val_%d_%s.yaml' % (save_config_dir, study, idx, vis), 'w') as outfile:
                        yaml.dump(config_dict, outfile, default_flow_style=False, sort_keys=False)


def series_intersection(s1, s2):
    return pd.Series(list(set(s1) & set(s2)))
    
def fetch_meta(wsi_object):
    try:
        mpp = wsi_object.wsi.properties['openslide.mpp-x']
    except:
        mpp = -1
    try:
        mag = wsi_object.wsi.properties['openslide.objective-power']
    except:
        mag = -1
    misc = {'mag': int(mag), 'mpp': float(mpp)}
    return misc

def __create_splits__():
	splits_dir = './splits/5foldcv/'
	dataset_dir = './dataset_csv_mad/'
	svs_dir = '/media/hdd1/awesomePhD/Datasets/pan-cancer/TCGA-WSIs/%s'
	save_path = './heatmaps/5foldcv/process_list/'

	for study in tqdm(sorted(os.listdir(splits_dir))):
	    
	    slide_data = get_all_ids(dataroot=dataset_dir, study=study[:-4])
	    slide_data = slide_data[slide_data.columns[:7]]
	    slide_data.index = slide_data['slide_id'].str[:12]
	    slide_data.index.name = 'case_id'

	    for i in range(5):
	        split_df = pd.read_csv(join(splits_dir, study, 'splits_%d.csv' % i))
	        pats = split_df['val'].dropna()

	        assert len(pats) == len(set(slide_data['slide_id'].str[:12]).intersection(pats))

	        if 'kirp' in study or 'kirc' in study:
	            case_folder = 'tcga_kidney'
	        elif 'lusc' in study or 'luad' in study:
	            case_folder = 'tcga_lung'
	        else:
	            case_folder = study[:-4]
	            
	        if 'lung' not in case_folder:
	            continue
	        
	        svs_path = svs_dir % case_folder
	        svs_fnames = os.listdir(svs_path)
	        assert len(slide_data.loc[pats]['slide_id']) == len(set(pd.Series(svs_fnames)).intersection(slide_data.loc[pats]['slide_id']))
	        
	        df = slide_data.loc[pats]
	        df = df.reset_index()
	        
	        for idx, row in tqdm(df.iterrows()):
	            slide_id = row['slide_id']
	            slide_path = join(svs_path, slide_id)
	            wsi = WholeSlideImage(path=slide_path)
	            misc = fetch_meta(wsi)
	            if misc['mag'] == 40 and misc['mpp'] < 0.3:
	                df.loc[idx, 'mag'] = 40
	            else:
	                df.loc[idx, 'mag'] = 20
	                
	        if df[df.mag==20].shape[0] != 0:
	            df[df.mag==20].to_csv(join(save_path, '%s_val_%d_20x.csv' % (study[:-4], i)))
	        if df[df.mag==40].shape[0] != 0:
	            df[df.mag==40].to_csv(join(save_path, '%s_val_%d_40x.csv' % (study[:-4], i)))


def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
    
    
### SHAP PLOTTING
import copy
import os
from os.path import join
import pandas as pd
from tqdm import tqdm
import xml
from xml.dom.minidom import Document
import h5py
import numpy as np

import shap
import matplotlib.pyplot as plt
from shap.plots import labels
from shap.plots import *
import pandas as pd
import numpy as np
import pickle
import math
import os
from PIL import Image

import pickle
import pandas as pd
import os
from os.path import join
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import to_hex
import seaborn as sns
import numpy as np
import seaborn as sns

# Stratify Risk Group
def strat(hazard, p):
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)
"""
def getGlobalShap(attr, data, save_path, name, 
                  xtick_stride=None, xlim_range=None, limit=20):
    if xtick_stride is None and xlim_range is None:
        max_attr_val = math.ceil(np.abs(attr.max()))
        while max_attr_val / 2. > np.abs(attr.max()):
            max_attr_val /= 2.
        xtick_stride = max_attr_val / 2.
        xlim_range = [-max_attr_val, max_attr_val]
    
    fig, ax = plt.subplots(figsize=(20,20), dpi=300)
    shap.summary_plot(attr, data, plot_type='dot', plot_size=(10,10), 
                      show=False, color_bar_label='', color_bar=False, max_display=limit)
    ax.tick_params(axis='y', which='major', labelsize=30)
    ax.tick_params(axis='x', which='major', labelsize=36)
    
    
    if x_label:
        ax.xaxis.label.set_visible(True)
        ax.set_xlabel('Time (Years)', fontproperties=font_args['ax_label_fprop'])
    else:
        ax.xaxis.label.set_visible(False)
        
    if y_label:
        ax.yaxis.label.set_visible(True)
        ax.set_ylabel('Proportion Surviving', fontproperties=font_args['ax_label_fprop'])
        #plt.tight_layout()
    else:
        ax.yaxis.label.set_visible(False)
    plt.grid(axis='x')
    ax.set_xlim(xlim_range[0], xlim_range[1])
    ax.set_xticks(np.arange(xlim_range[0], xlim_range[1]+0.01, xtick_stride))

    fig.savefig(os.path.join(save_path, '%s_shap.png' % name), bbox_inches='tight')
    plt.close()
    return Image.open(os.path.join(save_path, '%s_shap.png' % name))#.resize((int(688*1.48), int(581*1.48)))
def getGlobalShap(attr, data, save_path, xtick_stride=None, xlim_range=None, 
                  limit=20, x_label=False, y_label=False, title='BLCA'):#, color_bar=False):
    if xtick_stride is None and xlim_range is None:
        max_attr_val = math.ceil(np.abs(attr.max()))
        while max_attr_val / 2. > np.abs(attr.max()):
            max_attr_val /= 2.
        xtick_stride = max_attr_val / 2.
        xlim_range = [-max_attr_val, max_attr_val]
    
    fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
    shap.summary_plot(attr, data, plot_type='dot', plot_size=(20, 20), 
                      show=False, color_bar_label='relative feature value', 
                      color_bar=True, max_display=limit)
    #ax.tick_params(axis='y', which='major', labelsize=30)
    #ax.tick_params(axis='x', which='major', labelsize=36)
    font_args = configure_font(label_size=60)
    plt.setp(ax.get_xticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
    plt.setp(ax.get_yticklabels(), fontproperties = font_args['shap_fprop'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.gcf().subplots_adjust(left=0.30)
    #plt.grid(axis='x')
    #plt.tight_layout()
    ax.set_xlim(xlim_range[0], xlim_range[1])
    ax.set_xticks(np.arange(xlim_range[0], xlim_range[1]+0.01, xtick_stride))
    #ax.set_yticks([], [])
    #plt.yticks(rotation=30)
    #plt.axis((0,120))
    
    if x_label:
        ax.xaxis.label.set_visible(True)
        ax.set_xlabel('Integrated Gradient Attribution', fontproperties=font_args['ax_label_fprop'])
    else:
        ax.xaxis.label.set_visible(False)
        
    if y_label:
        ax.yaxis.label.set_visible(True)
        ax.set_ylabel('Genes', fontproperties=font_args['ax_label_fprop'])
        #plt.tight_layout()
    else:
        ax.yaxis.label.set_visible(False)
        
    if title:
        ax.set_title(title, fontproperties=font_args['title_fprop'])
    else:
        ax.set_title('')
        

    fig.savefig(save_path)#, bbox_inches='tight')
    plt.close()
    fig = Image.open(save_path)#.resize((1000,1000))
    return fig.resize((400, 400))
"""



def getGlobalShap(train_data_df,test_data_df,train_shap_attr,test_shap_attr, save_path, max_display = 20, row_height = 0.4, axis_color="#333333", alpha=1):

    #features = pd.concat([train_data_df,test_data_df]).reset_index(drop = True)
    #shap_values = np.concatenate([train_shap_attr,test_shap_attr])
    #train_mask = range(len(train_data_df))
    #print(features.shape, shap_values.shape)

    max_attr_val = math.ceil(np.abs(test_shap_attr).max())
    while max_attr_val / 2. > np.abs(test_shap_attr).max():
        max_attr_val /= 2.
    xtick_stride = max_attr_val / 2.
    xlim_range = [-max_attr_val, max_attr_val]

    import matplotlib.pyplot as pl
    # convert from a DataFrame or other types
    feature_names = test_data_df.columns
    # feature index to category flag
    idx2cat = test_data_df.dtypes.astype(str).isin(["object", "category"]).tolist()
    features = test_data_df.values
    #import pdb;pdb.set_trace()

    feature_order = np.argsort(np.sum(np.abs(test_shap_attr), axis=0))[-max_display:]
    pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    for pos, i in enumerate(feature_order):
        pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = test_shap_attr[:, i]
        values = None if features is None else test_data_df.values[:, i]
        values_ref = None if features is None else train_data_df.values[:, i]
        inds = np.arange(len(shaps))
        if values is not None:
            values = values[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]: # check categorical feature
                colored_feature = False
            else:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
        except:
            colored_feature = False
        N = len(shaps)

        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(values_ref, 5)
            vmax = np.nanpercentile(values_ref, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values_ref, 1)
                vmax = np.nanpercentile(values_ref, 99)
                if vmin == vmax:
                    vmin = np.min(values_ref)
                    vmax = np.max(values_ref)
            if vmin > vmax: # fixes rare numerical precision issues
                vmin = vmax
            #print(features.shape[0], len(shaps))

            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

            # plot the nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            #pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin, vmax=vmax, 
            #           s=16, alpha=alpha, linewidth=0,
            #           zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin

            #values[train_mask] = np.nan
            #nan_mask = np.isnan(values)

            pl.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                       cmap='coolwarm',  s=30, vmin=vmin, vmax=vmax,
                       c=cvals[np.invert(nan_mask)], alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)
        else:

            pl.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                       color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    
    #pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    pl.tick_params(axis='y', which='major', labelsize=20)

    pl.gca().tick_params('x', labelsize=20)
    pl.ylim(-1, len(feature_order))
    pl.xlabel('Attributions', fontsize=30,fontweight = 'bold')
    pl.ylabel('Genomic Features', fontsize=30,fontweight = 'bold')
    plt.xlim(xlim_range[0], xlim_range[1])
    plt.xticks(np.arange(xlim_range[0], xlim_range[1]+0.01, xtick_stride))


    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap='coolwarm')
    m.set_array([0, 1])
    cb = pl.colorbar(m, ticks=[0, 1], aspect=50)
    cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
    cb.set_label('Relative Feature Value', size=20, labelpad=0)
    cb.ax.tick_params(labelsize=20, length=0)
    cb.set_alpha(1)
    #cb.draw_all()
    cb.outline.set_visible(False)
    

    pl.tight_layout()

    plt.savefig(save_path,bbox_inches = 'tight')
    plt.close()
    plt.cla()
    plt.clf()



def getSHAPLocalExplanationPlot(shap_values, features=None, reference=None, feature_names=None, max_display=None, plot_type=None,
                                color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                                color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                                class_inds=None,
                                color_bar_label=labels["FEATURE_VALUE"],
                                # depreciated
                                auto_size_plot=None,
                                use_log_scale=False):
    """Create a SHAP summary plot, colored by feature values when they are provided.
    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.
    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand
    feature_names : list
        Names of the features (length # features)
    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)
    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.
    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that 
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.
    """
    import matplotlib.pyplot as pl

    # deprecation warnings
    if auto_size_plot is not None:
        warnings.warn("auto_size_plot=False is deprecated and is now ignored! Use plot_size=None instead.")

    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar" # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot" # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == 'layered_violin':
            color = "coolwarm"
        elif multi_class:
            color = lambda i: colors.red_blue_circle(i/len(shap_values))
        else:
            color = colors.blue_rgb

    # convert from a DataFrame or other types
    if str(type(reference)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = reference.columns
        reference = reference.values
        features = features.values
    elif isinstance(reference, list):
        if feature_names is None:
            feature_names = reference
        reference = None
    elif (reference is not None) and len(reference.shape) == 1 and feature_names is None:
        feature_names = reference
        reference = None

    num_reference = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

    if reference is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_reference - 1 == reference.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_reference == reference.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_reference)])

    if use_log_scale:
        pl.xscale('symlog')

    if max_display is None:
        max_display = 20

    if sort:
        # order reference by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_reference)), 0)

    row_height = 0.4
    if plot_size == "auto":
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    #if not multi_class and plot_type == "bar":
    #    feature_inds = feature_order[:max_display]
    #    y_pos = np.arange(len(feature_inds))
    #    global_shap_values = shap_values.mean(0)
    #    pl.barh(y_pos, global_shap_values[feature_inds], 0.7, align='center', color=color)
    #    pl.yticks(y_pos, fontsize=13)
    #    pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])
        
    if plot_type == "local_bar":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            feat = features[:,i]
            values = None if reference is None else reference[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            #if values is not None:
            #    values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False

            # trim the color range, but prevent the color range from collapsing
            values = np.concatenate((feat, values), axis=0) ### new
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
            if vmin > vmax: # fixes rare numerical precision issues
                vmin = vmax
                            
            # plot the non-nan values colored by the trimmed feature value
            cvals = values
            #cvals = np.concatenate((shaps, cvals), axis=0) ### new
            cvals_imp = cvals.copy()
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            heatmap_color = plt.get_cmap("coolwarm")#colors.red_blue_no_bounds(cvals)[0]
            heatmap_norm  = mpl.colors.Normalize(vmin= vmin, vmax=vmax)

            flag = 0
            if shaps[0] > 2:
                shaps[0] = 2
                flag = 1
            elif shaps[0] < -2:
                shaps[0] = -2
                flag = 1

            pl.barh(pos, shaps, 0.7, align='center',
                   ##vmin=vmin, vmax=vmax, #s=200,
                   color= heatmap_color((feat - vmin)/(vmax - vmin)), alpha=alpha, linewidth=0,
                   zorder=3, rasterized=len(shaps) > 500)

            #if flag:
            #    pl.plot(shaps, pos, marker="x", linestyle="", alpha=0.8, color="k")



    # draw the color bar
    
    if color_bar and reference is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in pl.cm.datad):
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap='coolwarm')
        m.set_array([0, 1])
        cb = pl.colorbar(m, ticks=[0, 1], aspect=50)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, size=25, labelpad=-25)
        cb.ax.tick_params(labelsize=25, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        #bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        #cb.ax.set_aspect((bbox.height - 0.9) * 20)
        #cb.draw_all()
    

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    pl.gca().tick_params(axis='y', which='major', labelsize=20)
    pl.gca().tick_params('x', labelsize=20)
    pl.ylim(-1, len(feature_order))
    pl.xlabel('Attributions', fontsize=30,fontweight = 'bold')
    pl.ylabel('Genomic Features', fontsize=30,fontweight = 'bold')
    pl.title(f'Total Attributions: {shap_values.sum():.2f}', fontsize=30,fontweight = 'bold')


    #if plot_type != "bar":
    #    pl.gca().tick_params('y', length=20, width=0.5, which='major')
    #pl.gca().tick_params('x', labelsize=11)
    #pl.ylim(-1, len(feature_order))
    #if plot_type == "bar":
    #    pl.xlabel(labels['GLOBAL_VALUE'], fontsize=13)
    #else:
    #    pl.xlabel(labels['VALUE'], fontsize=13)
    if show:
        pl.show()
        

import shap
import matplotlib.pyplot as plt
from shap.plots import labels
from shap.plots import *
import pandas as pd
import numpy as np
import pickle
import math
import os
from PIL import Image


def getIndividualSHAP(train_data_df,test_data_df,test_shap_attr, case_id,save_path, max_display = 20, viz=False):
    #xlim_range=[-2,2], xtick_stride=1, max_display=20):

    ### Gets max attr value
    max_attr_val = math.ceil(np.abs(test_shap_attr).max())
    while max_attr_val / 2. > np.abs(test_shap_attr).max():
        max_attr_val /= 2.
    xtick_stride = max_attr_val / 2.
    xlim_range = [-max_attr_val, max_attr_val]
    
    #fig, ax = plt.subplots(figsize=(20,20), dpi=100)
    
    if len(test_data_df.loc[case_id].shape) > 1:
        feats = test_data_df.loc[case_id].iloc[0]
        feats = pd.DataFrame(feats).T
    else:
        feats = test_data_df.loc[case_id]
        feats = pd.DataFrame(feats).T 
    
    getSHAPLocalExplanationPlot(test_shap_attr, 
                            features=feats, 
                            reference=train_data_df,
                            plot_type='local_bar', plot_size='auto', show=False, color_bar_label='Relative Feature Value', 
                            color_bar=True, max_display=max_display)
    #plt.grid(axis='x')
    plt.xlim(xlim_range[0], xlim_range[1])
    plt.xticks(np.arange(xlim_range[0], xlim_range[1]+0.01, xtick_stride))
    plt.savefig(save_path + '.eps', bbox_inches='tight')
    plt.close()
    plt.cla()
    plt.clf()
    #return fig

def myround(x, prec=2, base=0.1):
    return round(base * round(float(x)/base),prec)

def load_genomic_df(process_stack, dataset_path: str='./dataset_csv/', 
    splits_path = None, split_mode = False, modalities = None):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    #genomic_data = pd.read_csv(os.path.join(dataset_path, 'tcga_gbmlgg_survival_169_allmodalities.csv'))
    genomic_data = pd.read_csv(os.path.join(dataset_path)).drop_duplicates('subject_id')
    genomic_data = genomic_data.set_index('subject_id')
    #genomic_data.index = genomic_data.index.str[:12]
    genomic_data = genomic_data.drop(['slide_id', 'survival_months', 'censorship',
                                      'oncotree_code', 'age', 'is_female','train'] + modalities, axis=1)
    genomic_columns = genomic_data.columns
    #whole_set = genomic_data.index.intersection(process_stack.index.unique()).unique()
    splits = pd.read_csv(splits_path)
    train_set = splits.train.values
    test_set = process_stack.index.unique()
    #if split_mode == 'train_val_test':
    #    test_set = splits.test.dropna().values
    #elif split_mode == 'train_val':
    #    test_set = splits.val.dropna().values
    #else:
    #    raise NotImplementedError

    #import pdb; pdb.set_trace()

    train_data = genomic_data.loc[train_set,:]
    test_data = genomic_data.loc[test_set,:]
    test_data_orig = test_data.copy()
    scaler = StandardScaler().fit(train_data)
    train_data_df = pd.DataFrame(scaler.transform(train_data))
    test_data_df = pd.DataFrame(scaler.transform(test_data))
    train_data_df.index = train_data.index
    test_data_df.index = test_data.index
    train_data_df.columns = train_data.columns
    test_data_df.columns = test_data.columns
    
    return train_data_df, test_data_df, test_data_orig
    #return test_data

"""
def load_risk_df(pkl_path):
    with open(pkl_path, 'rb') as f:
        x = pickle.load(f)
    results_df = pd.DataFrame(x).T
    p = np.percentile(results_df.risk, [50])
    results_df.insert(0, 'strat', [hazard2grade(risk, p) for risk in results_df['risk']])
    return results_df
"""
def load_risk_df(pkl_path):
    all_result_pkl =[i for i in os.listdir(pkl_path) if 'pkl' in i and 'val' in i]
    results_df = []
    for pkl_f in all_result_pkl:
        pkl_file = os.path.join(pkl_path,pkl_f)
        with open(pkl_file, 'rb') as f:
            x = pickle.load(f)
            results_df.append(pd.DataFrame(x))
            
    results_df = pd.concat(results_df)
    p = np.percentile(results_df.risk, [50])
    results_df.insert(0, 'strat', [hazard2grade(risk, p) for risk in results_df['risk']])

    results_df.set_index('subject_id',inplace = True)
    return results_df

def load_signature_df(proj, dataset_path: str='./dataset_csv/'):
    signatures = pd.read_csv('%s/signatures.csv' % dataset_path)
    genomic_features = load_genomic_df(proj, dataset_path)

    omic_names = []
    for col in signatures.columns:
        omic = signatures[col].dropna().unique()
        omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
        omic = sorted(series_intersection(omic, genomic_features.columns))
        omic_names.append(omic)
    omic_sizes = [len(omic) for omic in omic_names]

    return omic_names, omic_sizes, genomic_features