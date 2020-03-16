# python 2createexamples.py -s 636 -u True
"""
This module reads filtered data for each of the runs of a specific subject 
and creates examples out of them. While doing so, it does not use the whole
brain as a volume of interest, while masking the relevant non-zero voxels using 
fmri_dataset (mask) either whole brain from FSL feat or a list of different binarized masks 
e.g. right-frontal-pole and left-temporal pole. 

It uses onset times from behavioral files and converts volumes into examples
(or trials). The time between onset + 4s and onset + 7s is assumed to hold 
the maximum information about the peak BOLD response and thus it's these 
three or four TRs (or a mean of them) that are (is) used to create a feature
vector (of length same as the number of voxels in an ROI).

A number of sample and feature attributes are added to the resulting feature
vectors including voxel index, and experimental condition.

Finally, the corresponding labeled feature vectors are pickled and stored in
relevant filtered/*/*.feat/masks directory. 
"""

import os
from glob import glob
import pickle
import argparse as ap
import numpy as np
from mvpa2.datasets.mri import *
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.datasets.miscfx import *

def main():
    # THE PARSER CODE IS PARSING THE TERMINALS ARGS 
    # Just type: python 2creating examplesXXX.py -s subjnum
    parser = ap.ArgumentParser(description = "Labeling of volumes.")
    parser.add_argument('-s', '--subject_number',
                        help = 'subject number',
                        required = True)
    args = vars(parser.parse_args())
    s_id = args['subject_number'] 
#    s_id = "002" #TO TEST IN CAJAL    
    fmri_path = '/export/home/dsoto/public/Fantasmas_MRI_analysis/images_forAnalysis/Functional'
    dest_path = os.path.join(fmri_path +'/rois4MVPAlabeledAROMA', s_id)
    
    # important that each fMRI run gets time-locked to it's own word-onsets.    
    source_path = os.path.join(fmri_path, s_id) 
    behav_path = os.path.join(fmri_path + '/psychopy_files', s_id)
    feat_dirs = sorted(glob(os.path.join(source_path, 'run*/run*mvpaAroma*'))) # CREATES LISTS of SUBFOLDERs
    ##print (feat_dirs)
    behav_pkls = sorted([f for f in glob(os.path.join(behav_path, 'run*', '*.pkl'))])#SORTS BY DATE
    print zip(feat_dirs, behav_pkls) # for checking mri data aligns with behav data
    
       #below searches for the files
    for feat_indexdir, feat_dir in enumerate(feat_dirs):
        print (feat_dir)
        path2ROIpkls = os.path.join(dest_path, os.path.basename(feat_dir)[:-5])
        if not os.path.exists(path2ROIpkls):
            os.makedirs(path2ROIpkls)
        # load all behavioral data for this run of this subject
        beha_file = behav_pkls[feat_indexdir] # corresponding
       # print '\n' + beha_file
    
        with open(beha_file, 'rb') as f: # sorting helps
            beha_pkldat = pickle.load(f)
    
        mri_datafile = os.path.join(feat_dir, 'filtered_func_data.nii.gz') # temp
        
        for mask in glob(source_path+'/label/BOLD*.nii.gz'):
            mri_data = fmri_dataset(mri_datafile, mask=mask)
            """
            NOW CALLS THE FUNCION label_examples -
            """
            mri_data = label_examples(mri_data, beha_pkldat)
            # label chunks
            mri_data.sa['chunks'] = np.array([feat_indexdir] * mri_data.shape[0])
            pkl_path = os.path.join(path2ROIpkls, os.path.basename(mask)[:-7]+'.pkl')#rm .nii.gz
            with open(pkl_path, 'wb') as f:
                pickle.dump(mri_data, f)
        


def label_examples(mri_data, beha_pkldat):

    # extract volume time-stamps from fMRI dataset (pymvpa2 Dataset)
    vol_times = mri_data.sa.time_coords
    
    # extract stimulus information from psychopy files (pandas DataFrame)
    onsets = beha_pkldat['TrialOnset'].values
    if 'trials_1.thisTrialN' in beha_pkldat: 
        trials = beha_pkldat['trials_1.thisTrialN'].values 
    else:
        trials = beha_pkldat['trials_2.thisTrialN'].values 
    
    memory_status = beha_pkldat['condition'].values
   
  
    """
    NOW CALLS THE FUNCION label_trials - which labes relevant TRs with trialnumber and memory_status name
    """
    mri_data.sa['trials'] = label_trials(onsets,trials,vol_times)
    mri_data.sa['targets'] = label_trials(onsets,memory_status,vol_times)
    
    # remove volumes that are of no interest to us
    mri_data = mri_data[mri_data.sa.targets != '_no-use_']
    #if take_mean is True then use mean of volumes as examples
    #print [t for t in mri_data.samples[4:8]]
    #print mri_data.sa.targets
    #print mri_data.shape
    
    mri_data=mri_data.get_mapped(mean_group_sample(['targets', 'trials'], order = 'occurrence'))
    #print [t for t in mri_data.samples[2]]
    print mri_data.sa.targets
    print mri_data.sa.trials
    print vol_times# = mri_data.sa.time_coords
    print mri_data.shape
    """
    IMPORTANT CHECK
    """
    print mri_data.summary()
    return mri_data

def label_trials(onsets, using, vol_coords):
    """
    Not all volumes are volumes of interest. 
    
    This will use onsets, using and vol_coord to create a list of labels. 
    Volume of interest would be labeled as either an activity or a rest
    while those of no use would be labeled as 'no-use'.

    Inputs:
    -------
    onsets: numpy ndarray
    using: numpy ndarray
    vol_coords: numpy ndarray

    Returns:
    --------
    labels: numpy ndarray
    """
    labels = np.array(['_no-use_'] * vol_coords.shape[0])#FOR ARRAY INITIALISATION
    for vol_ind in xrange(labels.shape[0]):
        for onset_ind in xrange(onsets.shape[0]):
            # the test words started at 0 an lasted for 3.28 seconds
            if onsets[onset_ind] + 4 <= vol_coords[vol_ind] < onsets[onset_ind] + 9.28:
                labels[vol_ind] = using[onset_ind]
                print labels
    return labels

if __name__ == '__main__':
    main()