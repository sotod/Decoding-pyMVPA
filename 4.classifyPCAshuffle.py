"""
This module:
    - classification of semantic category in 'is' as well as 'im' trials
    - reads preprocessed data stored @ `preprocessed*.pkl`, and selects 'is' and 'im' examples out of it, 
    - creates partitions based on 'chunks' with n - 1 used for training and 1 for testing, 
    - uses univariate feature selection and SVM for classification, and
    - finally stores the corresponding accuracies @ `results-univariate*`. 
    - for every subject, it stores one dictionaries i.e. `results.pkl`
    - each of these dictionaries look as follows: 

    ```
    {'roi_1': {'is': accuracy_1, 'im': accuracy_2}, 'roi_2': {'is': accuracy_1, ...}, 'roi_3': ...}
    ```
"""

import os
from glob import glob
import pickle
import argparse as ap

import numpy as np

from mvpa2.suite import *

from mvpa2.generators.partition import NFoldPartitioner
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score

def main():    
    parser = ap.ArgumentParser(description = "Classification -- chunks as folds")
    parser.add_argument('-s', '--subject_number',
                        help = 'subject number',
                        required = True)
    args = vars(parser.parse_args())
    s_id = args['subject_number']
#    s_id = "002"
    source_path = os.path.join('/export/home/dsoto/public/Fantasmas_MRI_analysis/images_forAnalysis/Functional/preproAROMA', s_id)
    tgt_path = os.path.join('/export/home/dsoto/public/Fantasmas_MRI_analysis/images_forAnalysis/Functional/results-univar-FS-AROMA', s_id)
    pkls = os.path.join(source_path, 'z_vstack*.pkl')
    if not os.path.exists(tgt_path): os.makedirs(tgt_path)
    results = {}
    for pkl in glob(pkls):
        roi = os.path.basename(pkl)[:-4]
#        print roi
        """
        call the function to load data and then classify
        """
        ds = load_preprocessed(pkl)#CALLS FUNCTION
        ds = ds[ds.sa.targets != 'nw']  
#        print ds.summary()
        results[roi] = classify(ds)
#    

    print results, '\n'

    with open(os.path.join(tgt_path, 'results_w_ph_AROMA_shuffle.pkl'), 'wb') as f:
        pickle.dump(results, f)

def load_preprocessed(pkl):
    """
    Load all preprocessed data for a specific roi (stored @ pkl).

    Inputs:
    -----------
    pkl: str

    Returns:
    --------
    feat_ds: pymvpa2 Dataset
    """
    
    with open(pkl, 'rb') as f:
        feat_ds = pickle.load(f) 
    return feat_ds

def classify(ds):
    """ Classification of category. """
    np.random.seed(123)#!!!!!!!!!

    ct_results = {}
    ds.sa['trialcount']=range(0, ds.shape[0])
    print ds.sa.trialcount
    uniques = np.unique(ds.sa.targets)

    pt = ChainNode([NFoldPartitioner(cvtype=0.2, attr='trialcount', count=600, selection_strategy='random'),
                   Balancer(attr='targets', count=1, limit='partitions', apply_selection=True),
                   Sifter([('partitions', 2), ('targets', {'uvalues': uniques})])],
                   space='partitions') #real count based on 'if' statement later
    cv_results, counter = (), 0
    for dat in pt.generate(ds): #pyMVPA generates train and test set
        tr = dat[dat.sa.partitions == 1]#pyMVPA
        te = dat[dat.sa.partitions == 2]#pyMVPA
        if int(0.1*tr.shape[0]) < te.shape[0] <= int(0.2*tr.shape[0]):
            fsel = PCA()
            clfr = LinearSVC()
            pipe = make_pipeline(fsel, clfr)
            pipe.fit(tr.samples, tr.sa.targets)
            preds = pipe.predict(te.samples)
            cv_results += (accuracy_score(te.sa.targets, preds),)
        
            counter += 1
        if counter>300:
            break
            
    ct_results = np.mean(cv_results)
    return ct_results

if __name__ == '__main__':
    main()