
# coding: utf-8

# In[2]:


# RUN with 

# python 3stacking_and_prepro -s 636


"""
This module reads labeled data that we have labeled using label.py 
and stacks all sessions corresponding to an ROI into one big matrix of feature vectors.
It will pick each of the 8 matrices (from each of the 8 feat directories) 
and stack them based on intersecting voxel indices.

Once the sessions are stacked, it will preprocess the data using in-place 
linear detrending and z-score normalization.
"""

import os
from glob import glob
import pickle
import argparse as ap

import numpy as np
from mvpa2.base.dataset import vstack
from mvpa2.datasets.miscfx import *
from mvpa2.mappers.zscore import zscore
from mvpa2.mappers.detrend import poly_detrend

def main():  
    parser = ap.ArgumentParser(description = "Stacking and Preprocessing.")
    parser.add_argument('-s', '--subject_number',
                        help = 'subject number',
                        required = True)
    args = vars(parser.parse_args())
    s_id = args['subject_number']
    #s_id = "002" #TO TEST IN CAJAL02 
    #labelled_dirs_path is the despitination path in 2createexamples
    labelled_dirs_path = os.path.join('/export/home/dsoto/public/Fantasmas_MRI_analysis/images_forAnalysis/Functional/rois4MVPAlabeledAROMA', s_id)
    labeled_dirs = sorted(glob(os.path.join(labelled_dirs_path, 'run*')))
    #print labeled_dirs
    
    dest_path = os.path.join('/export/home/dsoto/public/Fantasmas_MRI_analysis/images_forAnalysis/Functional/preproAROMA', s_id)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    pkllist = ['BOLDlh_caudat_fsl.pkl', 'BOLDlh_suptemp_fsl.pkl', 'BOLDlh_parsoper_fsl.pkl',  'BOLD_lIFG.pkl', 'BOLDlh_parsorbi_fsl.pkl', 'BOLDlh_parstri_fsl.pkl', 'BOLDlh_putam_fsl.pkl'] 
    #pkllist = ['BOLDlh_caudat_fsl.pkl', 'BOLDlh_suptemp_fsl.pkl', 'BOLDrh_parstri_fsl.pkl', 'BOLDlh_parsoper_fsl.pkl',  'BOLD_lIFG.pkl',  'BOLDrh_putam_fsl.pkl', 'BOLDlh_parsorbi_fsl.pkl',  'BOLDrh_caudat_fsl.pkl', 'BOLDrh_suptemp_fsl.pkl', 'BOLDlh_parstri_fsl.pkl', 'BOLDrh_parsoper_fsl.pkl', 'BOLD_rIFG.pkl', 'BOLDlh_putam_fsl.pkl', 'BOLDrh_parsorbi_fsl.pkl'] 
    for pklfile in pkllist:
        runs = ()
        print pklfile
        for labeled_dir in labeled_dirs:
            print labeled_dir
            with open(os.path.join(labeled_dir, pklfile), 'rb') as f:
                runs = runs + (pickle.load(f),)#to add a tuple to a tuple you need to have a comma
        stacked = vstack(runs, a='all') #PYMVPA
        print 'stacked:', stacked.shape
        
        #CHOOSING THE 1ST HEADER TO AVOID PROBLEMS WITH map2nifti later in searchlight    
#        stacked.a['imgtype'] = stacked.a.imgtype[0]
#        stacked.a['imghdr'] = stacked.a.imghdr[0]
#        stacked.a['mapper'] = stacked.a.mapper[0]
        #print 'Type of mapper:', type(stacked.a.mapper) # should not be a tuple now
        
        # linear detrending and z-score normalization and add more sample attributes
        preprocessed = preprocess(stacked) #CALLS def preprocess FUNCTION
        #print 'preprocessed:', preprocessed.shape
    
        # store that one big feature set of this roi of this subject @ ../preprocessed/subj/roi.pkl
        with open(os.path.join(dest_path, 'z_vstack' + os.path.basename(pklfile)), 'wb') as f:
            pickle.dump(preprocessed, f)

def preprocess(ds):
    """
    In-place z-score normalization and linear detrending.

    Inputs:
    -------
    ds: pymvpa Dataset

    Returns:
    ds: pymvpa Dataset
    """
    ds = remove_invariant_features(ds) # it returns a dataset with invariant features/voxels removed
    poly_detrend(ds, polyord = 1, chunks_attr = 'chunks') # in-place operations
    zscore(ds, chunks_attr = 'chunks')
    return ds


if __name__ == '__main__':
    main()