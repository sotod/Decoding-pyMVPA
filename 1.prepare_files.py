import os
from glob import glob
import pickle
import pandas as pd
"""
behav_path contains subject folders and in it the psychopy files for each run
"""
behav_path = os.path.join('/export/home/dsoto/public/Fantasmas_MRI_analysis/images_forAnalysis/Functional/psychopy_files')

"""
loops through subjects and then through csv onset files of the different runs
"""
for subj_dir in [dir for dir in glob(os.path.join(behav_path, '*')) if os.path.isdir(dir)]:
#    print subj_dir
    for run_index, run_file in enumerate(glob(os.path.join(subj_dir, '*', '*.csv'))):
#        print run_file
        #REMOVE EMPTY COLUMNS IN CSV
        run_df = pd.read_csv(run_file).dropna(axis = 1, how = 'all') 
        
        print run_df.TrialOnset
        #REPLACES NO KEY RESPONSES WITH -1 # NO NEEDED AS THERE WERE NO KEYS 
        #run_df = pd.read_csv(run_file).dropna(axis = 1, how = 'all').replace(['None', np.nan], -1)

        # because every 9th row is empty in Psychopy due to block changes
        #run_df = run_df.drop(run_df.index[range(8, 36, 9)])
        #print run_df.shape

        for key in run_df:
            # To account for removed fMRI volumes; 11 * TR of 0.85s = 9.35
            if 'TrialOnset' in key:
                run_df[key] -= 9.35
        print run_df.TrialOnset
                #print run_df[key] 
            # To make sure that all response keys are integers
            # NO NEEDED AS THERE WERE NO KEYS 
            #if 'keys' in key: 
            #    run_df[key] = run_df[key].apply(int)

        #REMOVES EMPTY CHARACTERS BEFORE AND AFTER LABELS PROBLEM IN CSV FILE IN PSYCHOPY PROBLEM!!!
        run_df['condition'] = [i.split(".")[0].split("\\")[1].split("_")[0] for i in run_df['sounds'].tolist()]
        print run_df['condition'] 
        run_df['run'] = [run_index] * run_df.shape[0]
        #print run_df.head()
            #splittext split extension file (csv) fron tha path
            # so [0] below just takes the path, while it adds the .pkl extension
        with open(os.path.splitext(run_file)[0] + '.pkl', 'wb') as f:
                pickle.dump(run_df, f)

