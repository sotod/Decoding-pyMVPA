"""
Comparison of classification performance between conscious and non-conscious in all ROIs.
"""

import os
from glob import glob
import pickle

from scipy.stats import ttest_1samp
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 2)


def main():
    res_path = os.path.join('/home/dsoto/public/Fantasmas_MRI_analysis/images_forAnalysis/Functional/results-univar-FS')
    formatted = format_results(res_path)
    
    # paired t-test for comparison of read and reenact
    # classification performance in all ROIs
    p_values = ttest_1samp(formatted, 0.5)[1]
    corr_p_mask, corr_p_values = fdrcorrection0(p_values)
    print pd.DataFrame({"mean": formatted.mean(), 'corrected': corr_p_values, 'uncorrected': p_values}, index = formatted.columns).T

def format_results(res_path):
    """
    Prints formatted results based on:

    Inputs:
    -------
    res_path: '../_results_/results-*'
    """

    results = {}
    for subj in [d for d in glob(os.path.join(res_path, '*')) if os.path.isdir(d)]:
        with open(os.path.join(subj,  'results_w_ph.pkl'), 'rb') as f:
            subj_dat = pickle.load(f)
        
        subj_con_dat = {}
        for roi in subj_dat.keys():
            subj_con_dat[roi] = subj_dat[roi]

        results[os.path.basename(subj)] = subj_con_dat

    return pd.DataFrame(results).T
 

if __name__ == '__main__':
    main()