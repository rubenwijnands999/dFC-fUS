import os
import numpy as np

from sICA import sICA
from pre_processing import PreProcessing
from main_processing import MainProcessing
from main_processing_separate_groups import MainProcessingSepGroup

"""
Author: Ruben Wijnands

Script for reproducing results of the paper. 
Note: you do need access to the full data set.
"""
#%% Perform spatial independent component analysis (sICA)

sICA_ID = 11  # Give sICA ID name
TableEntry = 1  # Specific recording of a mouse. TE == Row of CSV table.
NumComp = 25  # Model order of sICA
seed_nr = 0  # Seed number for reproducible sICA results

# Apply sICA routine
s = sICA(sICA_ID, TableEntry, NumComp, seed_nr)  # Create sICA object
s.import_data()
s.pre_processing()
s.run_sICA()
s.post_processing()
s.plot_components(plot=True, save=True)  # Plots are stored under the p['fig_sICA_path'] in util.py

#%% Import brain warp and select regions of interest (ROIs) in the fUS GUI (run run_app.py)
# Brain warps should be of .jpg or .tif format and stored in the data/Warped_brains folder.
# Then, depending on the num_regions parameter in the configurations.ini file, you can select ROIs.
os.system("python run_app.py")

#%% Perform subsequent pre-processing steps
# Give PreProcessingName name
PreProcessingID = 32

# sICA info for pre-processing
sICA_RUN = 7
seed_num = 0

overwrite = False  # Overwrite PRE_x.json
specific_rec = []  # Enter entry for testing purpose (only analyses the specific entry)

PreProcessingName = "PRE_TEST" if len(specific_rec) != 0 else "PRE_" + str(PreProcessingID)

# Execute pre-processing steps
PreProcessing(PreProcessingName, sICA_RUN=sICA_RUN, seed_num=seed_num, overwrite=overwrite, specific_rec=specific_rec)
f = PreProcessing(PreProcessingName)

# Plot most important results
recs = range(f.num_recs) if len(specific_rec) == 0 else specific_rec
for rec in recs:
    f.plot_filled_warp(rec)
    f.plot_sICA_masks_combined(rec)
    f.plot_original_vs_detrended_signal(rec)
    f.plot_time_courses(rec)
    pass

#%% Perform main-processing, i.e. deconvolution and possible HMM inference on all data concatenated. If you want to do
# HMM inference on the concatenated data of two groups of mice separately, please run the next section.
PreProcessingName = "PRE_32"
MainProcessingID = 12

# mus_target (K x M) is here to determine the order of inferred functional networks.
mus_target = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,0]])

overwrite = False  # Overwrite MAIN_x.json
specific_rec = []  # Enter entry for testing purpose (only analyses the specific entry)

# Determine MainProcessingName
MainProcessingName = "MAIN_TEST" if len(specific_rec)!=0 else "MAIN_" + str(MainProcessingID)

# Execute main-processing steps
MainProcessing(PreProcessingName,MainProcessingName,overwrite=overwrite, specific_rec=specific_rec, mus_target=mus_target)
main = MainProcessing(PreProcessingName,MainProcessingName)

# Plot most important results
recs = range(main.num_recs) if len(specific_rec) == 0 else specific_rec
for rec in recs:
    if str(rec) in main.F_final:
        main.plot_neural_time_courses(rec)
        # main.plot_lambda_mse(rec)
        pass

#%% Perform main-processing on separate groups of mice
PreProcessingName = "PRE_32"
MainProcessingName = "MAIN_12"

main_sep_group = MainProcessingSepGroup(PreProcessingName,MainProcessingName)

# Random grouping of mice and evaluating effect. This can take a while -> reduce num_MonteCarlo_runs for faster results,
# but reduced statistical support. MSE_range selects part of the state transition probability matrices over which the
# measured and Monte Carlo effects are calculated.
true_mse_conf, mse_list = main_sep_group.convergence_conf_int(num_states=4, num_MonteCarlo_runs=500, min_changes=1,
                                                              shuffle_method='rec', mse_range=[1, 3])
main_sep_group.plot_REF_mus(type='wt')
main_sep_group.plot_REF_mus(type='hom')
main_sep_group.plot_MEAN_state_trans_prob_matrix_comb()
main_sep_group.plot_MSE_distribution()

