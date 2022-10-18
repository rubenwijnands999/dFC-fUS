import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import matplotlib.colors as colors
import matplotlib
import pandas as pd
import ssm as ssm
import copy
import scipy.linalg as sl
import os
import json
import seaborn as sns
import matplotlib.patches as mpatches
import random
import util

from textwrap import wrap
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.special import gamma
from pre_processing import PreProcessing
from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.linalg import norm
from matplotlib.legend_handler import HandlerTuple
from itertools import groupby
from operator import itemgetter


class MainProcessing(PreProcessing):
    def __init__(self,PreProcessingName,MainProcessingName,overwrite=None,specific_rec=None,mus_target=None):
        super().__init__(PreProcessingName)  # Do not perform new data extraction steps here
        self.MainProcessingName = MainProcessingName

        # Create paths
        self.paths = util.create_paths_dict(MainProcessingName=MainProcessingName)

        # Groups of mice
        self.phenotype_list_reduced = ["wt","hom"]

        # Check if extracted Main-Processing JSON file already exists in simulation folder
        if os.path.exists(self.paths['json_main_path']) and (not overwrite):
            new_job = False
            print("Using MAIN processing data from file " + self.MainProcessingName + " located at " + self.paths['json_main_path'])
        else:
            new_job = True
            print("Performing main-processing")

        if new_job:
            # For re-ordering the inferred functional networks
            self.mus_target = mus_target

            # Parameters used
            self.rel_MSE_threshold = float(self.config['Analysis']['rel_MSE_threshold'])

            # Create HRF
            self.HRF_param = {}
            self.HRF_param['p1'] = float(self.config['Analysis']['p1'])
            self.HRF_param['p2'] = float(self.config['Analysis']['p2'])
            self.HRF_param['T_HRF'] = int(self.config['Analysis']['T_HRF'])
            self.HRF_param['T_ind_HRF'] = int(self.HRF_param['T_HRF'] / self.param['T_res']) + 1
            self.HRF = self.gen_HRF()

            self.Y_list = {}
            self.lambda_thres_list = {}
            self.reconstruction_MSE_table = {}

            self.time = np.linspace(0, int(self.param['Nf'] * self.param['T_res']) - self.param['T_res'], self.param['Nf'])
            self.time_reduced = self.time[:len(self.time)-(self.HRF_param['T_ind_HRF']-1)]

            for rec in range(self.num_recs):

                # Only perform pre-processing on entries filled in csv table
                flag = False
                for region in self.ROI_list[:-2]:
                    if self.MiceDict[region][rec]: # At least one filled table entry
                        flag = True

                if (specific_rec is not None) and (len(specific_rec)!=0):
                    if rec not in specific_rec:
                        flag = False

                if flag:
                    print("Using rec",str(rec))
                    # Find optimal lambdas
                    self.compute_lambda(rec)

                    # Reconstruction using the found lambdas
                    Y_hat, reconstruction_MSE = self.reconstruct(self.HRF,rec)
                    self.Y_list[str(rec)] = Y_hat[:, self.HRF_param['T_ind_HRF'] - 1:]

            # Export results
            self.export_main_json()
        else:
            self.import_main_json()


    def group_analysis(self,num_states):
        self.num_states=num_states
        # Concatenate reconstructed neural data
        self.Y_concatenated = self.concatenate_y_ROIs()
        self.reduce_Y_list_indices()

        # Apply HMM
        self.hmm = ssm.HMM(self.num_states, self.num_regions, observations="gaussian")
        self.hmm_lls = self.hmm.fit(self.Y_concatenated.T, method="em", num_iters=50, init_method="kmeans")
        self.inf_states = self.hmm.most_likely_states(self.Y_concatenated.T) # Viterbi

        if np.size(self.mus_target, axis=0) != self.num_states:
            random_state = np.random.RandomState(seed=0)
            self.mus_target = random_state.randint(2, size=(self.num_states-1, self.num_regions)).astype(int)
            self.mus_target = np.insert(self.mus_target,0,np.zeros(self.num_regions),axis=0)

        # Permute to mus_target
        permutation = self.find_permutation_vec(self.mus_target, self.hmm).astype(int)
        print(permutation)
        self.hmm.permute(permutation)
        self.inf_states = self.hmm.most_likely_states(self.Y_concatenated.T)

        # Compute fractional occupancy of states
        self.frac_occ = self.compute_frac_occupancy(self.num_states,self.inf_states)

    def find_num_states(self):
        # Concatenate reconstructed neural data (includes rescaling of means)
        self.Y_concatenated = self.concatenate_y_ROIs()
        self.reduce_Y_list_indices()

        self.num_states_range = np.arange(2,9)
        self.frac_occ_means = {}
        self.mu_z_unique_networks = {}
        self.hmm_mus = {}

        for num_states in self.num_states_range:
            # Apply HMM
            hmm = ssm.HMM(num_states, self.num_regions, observations="gaussian")
            hmm_lls = hmm.fit(self.Y_concatenated.T, method="em", num_iters=50, init_method="kmeans")
            inf_states = hmm.most_likely_states(self.Y_concatenated.T)  # Viterbi
            self.hmm_mus[str(num_states)] = hmm.observations.mus # Store fitted means

            # Compute fractional occupancy of states
            frac_occ = self.compute_frac_occupancy(num_states,inf_states)
            self.frac_occ_means[str(num_states)] = np.zeros((2, num_states))
            for phenotype,i in zip(self.phenotype_list_reduced,range(len(self.phenotype_list_reduced))):
                indices = np.where(np.asarray(self.phenotype_list) == phenotype)[0]
                # Delete entries with zeros only
                frac_occ_red = []
                for ind in indices:
                    if frac_occ[ind, :].any():
                        frac_occ_red.append(frac_occ[ind, :])
                #frac_occ_red = frac_occ[indices, :]
                frac_occ_red = np.array(frac_occ_red)
                self.frac_occ_means[str(num_states)][i,:] = np.mean(frac_occ_red,axis=0)

            # Count functional network doubles
            activity_detection_matrix = normalize(hmm.observations.mus>1e-4, norm="l2")
            self.mu_z_unique_networks[str(num_states)] = np.size(np.unique(activity_detection_matrix,axis=0),axis=0)


    def gen_HRF(self):
        """
        Generate hemodynamic response function (HRF)
        :return: HRF.
        """
        p1 = self.HRF_param['p1']
        p2 = self.HRF_param['p2']
        p3 = 1  # Temporary

        t = np.linspace(0, self.HRF_param['T_HRF'], self.HRF_param['T_ind_HRF'])[:,None]
        HRF = (p3 * (((t) ** (p1 - 1) * p2 ** (p1) * np.exp(-p2 * (t))) / gamma(p1)))

        # Get index of maximum
        p3 = 1 / HRF[np.argmax(HRF)]
        HRF = p3 * HRF.T

        return HRF[0,:]

    def compute_lambda(self,rec, begin=-6, end=0, num=100):
        """
        Compute regularization parameter lambda.
        :param rec: recording or table entry identifier.
        :param begin: start of parameter sweep lamdba=10^(begin).
        :param end: end of parameter sweep lamdba=10^(end).
        :param num: Number of sweep points between begin and end (on logarithmic scale).
        :return: list of lambdas per recording, stored in self.lambda_thres_list
        """

        length = self.param['Nf']
        HRF_length = self.HRF_param['T_ind_HRF']

        # Create convolution matrix
        H = sl.convolution_matrix(self.HRF, length + HRF_length - 1) #HRF specified by HRF_param
        H = H[HRF_length - 1:length + HRF_length - 1, :]  #In reality we only have a part of H
        self.H_ = H

        self.lambda_range = np.logspace(begin, end, num, endpoint=True)
        self.reconstruction_MSE_table[str(rec)] = np.empty((self.num_regions, num))
        self.lambda_thres_list[str(rec)] = np.zeros(self.num_regions)

        text="Finding lambda"
        for i in range(self.num_regions):
            f = self.F_final[str(rec)][i, :].T
            l = Lasso(fit_intercept=False, positive=True, tol=1e-3, max_iter=10000, selection='cyclic', warm_start=True)  # Fit_intercept is false to avoid mean-subtraction
            for lambda_, j in tqdm(zip(self.lambda_range, range(num)),total=num,position=0,desc=text):
                l.set_params(alpha=lambda_)
                l.fit(H,f)
                y_solution = l.coef_[:length]
                self.reconstruction_MSE_table[str(rec)][i,j] = sum((H[:,:length] @ y_solution - f) ** 2)/len(y_solution)
        rel_MSE = (self.reconstruction_MSE_table[str(rec)]-self.reconstruction_MSE_table[str(rec)][:,0][:,None])/(self.reconstruction_MSE_table[str(rec)][:,-1]-self.reconstruction_MSE_table[str(rec)][:,0])[:,None]
        for j in range(self.num_regions):
            loc = np.where(rel_MSE[j,:]<self.rel_MSE_threshold)[0][-1]
            self.lambda_thres_list[str(rec)][j] = self.lambda_range[loc]


    def reconstruct(self,HRF,rec,single_lambda=None):
        """
        Reconstruct underlying activity of neural populations by deconvolution.
        :param HRF: Hemodynamic response function (HRF) used for deconvolution
        :param rec: recording or table entry identifier.
        :param single_lambda: if you want to use a specific lambda you can specify here (not used)
        :return: y_reconstructed,reconstruction_MSE. The actual reconstructed activity and the MSE of reconstruction.
        """
        length = self.param['Nf']
        HRF_length = self.HRF_param['T_ind_HRF']
        H = sl.convolution_matrix(HRF, length + HRF_length - 1)  # HRF specified by param
        H = H[HRF_length - 1:length + HRF_length - 1, :]  # In reality we only have a part of H
        self.H_ = H

        y_reconstructed = np.zeros((self.num_regions,length))
        reconstruction_MSE = np.empty(self.num_regions)

        for i in range(self.num_regions):
            f = self.F_final[str(rec)][i, :].T

            if single_lambda is not None:
                lambda_input = single_lambda
            else:
                lambda_input = self.lambda_thres_list[str(rec)][i]

            l = Lasso(alpha=lambda_input, fit_intercept=False, positive=True, tol=1e-4, max_iter=10000, selection='cyclic')  # Fit_intercept is false to avoid mean-subtraction
            l.fit(H, f)
            y = l.coef_[:length]
            reconstruction_MSE[i] = sum((H[:, :length] @ y - f) ** 2) / len(y)
            y_reconstructed[i,:] = y.T

        return y_reconstructed,reconstruction_MSE

    def find_permutation_vec(self,ref_mus,hmm_permutation):
        """
        Find permutation based on the mean signature of networks, using cosine similarity
        :param ref_mus: Reference networks
        :param hmm_permutation: HMM object with items to be permuted
        :return: Permutation vector indicating reshuffled order
        """

        ref_vecs = copy.deepcopy(ref_mus)
        hmm_vecs = copy.deepcopy(hmm_permutation.observations.mus > 1e-5)

        permutation_vector = np.empty(hmm_permutation.K)
        num_states = hmm_permutation.K
        mu_zeros_loc=None

        # Find zeros vectors first
        for mu_vec,i in zip(ref_vecs,range(hmm_permutation.K)):
            if np.all((mu_vec <= 1e-5)):
                mu_zeros_loc = i
                break

        for mu_vec,i in zip(hmm_vecs,range(hmm_permutation.K)):
            if np.all((mu_vec <= 1e-5)):
                permutation_vector[mu_zeros_loc]=i
                num_states = hmm_permutation.K-1
                mu_zero_perm_ind = i
                #Remove zero vectors in ref/hmm_vecs
                ref_vecs=np.delete(ref_vecs,mu_zeros_loc,axis=0)
                hmm_vecs=np.delete(hmm_vecs,i,axis=0)
                break

        if (mu_zeros_loc is not None) and (num_states == hmm_permutation.K - 1):  # Both have zero vectors
            # Matrix with cosine similarity measure
            self.cost_matrix = np.zeros((num_states,num_states))
            for i in range(num_states):
                for j in range(num_states):
                    self.cost_matrix[i,j] = np.dot(ref_vecs[i],hmm_vecs[j])/(norm(ref_vecs[i])*norm(hmm_vecs[j]))

            # Minimum cost permutation
            _,permutation = linear_sum_assignment(-self.cost_matrix)

            # Add one index to indices higher than and equal to mu_zeros_loc
            indices = np.where(permutation >= mu_zero_perm_ind)[0]
            addition = np.zeros(num_states)
            np.put(addition, indices, 1)
            permutation += addition.astype(int)

            # Re-insert removed permutation index
            permutation_vector[:mu_zeros_loc] = permutation[:mu_zeros_loc]
            permutation_vector[mu_zeros_loc+1:] = permutation[mu_zeros_loc:]
        else:
            print("No zero vectors found")

        return permutation_vector


    def concatenate_y_ROIs(self):
        """
        Function to concatenate reconstructed neural activity time courses of recordings.
        :return: Y_concatenated.
        """
        Y_concatenated = None
        self.Y_start_end_indices = np.zeros((self.num_recs, 2)).astype(int)
        self.Y_list_indices_reduced_animal = {}
        prev_ind = 0

        for animal in self.MiceDict['animalID']:
            self.Y_list_indices_reduced_animal[animal] = []

        # Concatenate all Y matrices
        for rec in range(self.num_recs):
            if str(rec) in self.Y_list:
                # Rescale means to same level (unity), due to different SNRs
                means = [np.mean(self.Y_list[str(rec)][i, :][self.Y_list[str(rec)][i, :] > np.max(self.Y_list[str(rec)][i, :])*1e-3]) for i in range(self.num_regions)]
                Y_list_rescaled = self.Y_list[str(rec)] / np.array(means)[:, None]

                if Y_concatenated is None:
                    Y_concatenated = Y_list_rescaled
                else:
                    Y_concatenated = np.append(Y_concatenated,Y_list_rescaled,axis=1)
                self.Y_start_end_indices[rec, :] = np.array([prev_ind, int(prev_ind + np.size(Y_list_rescaled, axis=1))])
                self.Y_list_indices_reduced_animal[self.MiceDict['animalID'][rec]].append(self.Y_start_end_indices[rec, :].tolist())

                prev_ind = int(prev_ind + np.size(Y_list_rescaled, axis=1))

        self.Y_start_end_indices = self.Y_start_end_indices[~np.all(self.Y_start_end_indices == 0, axis=1), :]

        return Y_concatenated

    def reduce_Y_list_indices(self):
        """
        Find all reconstructed neural activity indices in Y_concatenated of a mouse and store all corresponding
        indices in a list.
        :return: No return. Stored in self.Y_list_indices_reduced.
        """
        self.animal_list = list(set(self.MiceDict['animalID']))
        self.Y_list_indices_reduced = {}
        self.phenotype_list = [] # corresponding to self.animal_list
        for animalID in self.animal_list:
            self.Y_list_indices_reduced[animalID] = []
            indices = np.where(np.asarray(self.MiceDict['animalID']) == animalID)[0]
            for idx in indices:
                self.Y_list_indices_reduced[animalID].extend( np.arange(self.Y_start_end_indices[idx,0],self.Y_start_end_indices[idx,1]).tolist() )
            # Create new phenotype table
            self.phenotype_list.append(self.MiceDict['phenotype'][indices[0]])

    def compute_frac_occupancy(self,num_states,inf_states):
        """
        Compute the fractional occupancy of states for each mouse or for each group
        :return: Fractional occupancy array (animals x states)
        """
        frac_occ = np.zeros((len(self.animal_list), num_states))
        for animalID,i in zip(self.animal_list,range(len(self.animal_list))):
            for state in range(num_states):
                try:
                    frac_occ[i, state] = np.count_nonzero(inf_states[self.Y_list_indices_reduced[animalID]] == state)/(len(self.Y_list_indices_reduced[animalID]))
                except:
                    frac_occ[i, state] = 0
        return frac_occ

    def gen_phenotype_colors(self):
        self.wt_colors = ['steelblue','dodgerblue','deepskyblue','lightblue']
        self.wt_colors = [colors.to_rgb(color) for color in self.wt_colors]
        self.hom_colors = ['saddlebrown', 'chocolate', 'sandybrown', 'burlywood']
        self.hom_colors = [colors.to_rgb(color) for color in self.hom_colors]
        return self.wt_colors,self.hom_colors

    # Plotting
    def plot_rss_vs_num_states(self):
        mse_mean_frac_occ = np.zeros(len(self.num_states_range))
        for key,i in zip(self.frac_occ_means,range(len(self.num_states_range))):
            mse_mean_frac_occ[i] = sum(abs(self.frac_occ_means[key][0,1:] - self.frac_occ_means[key][1,1:]))

        plt.title("\n".join(wrap("RSS between mean fractional occupancies of WT and HOM",30)))
        plt.xlabel("Number of states")
        plt.ylabel("RSS")
        plt.plot(mse_mean_frac_occ)
        plt.xticks(np.arange(len(self.num_states_range)),labels=self.num_states_range)
        plt.grid(True)
        plt.plot()
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_rss_vs_num_states'
        plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
        print("Saving image")
        plt.show()

    def plot_unique_networks_vs_num_states(self):
        unique = np.zeros(len(self.num_states_range),dtype=int)
        for key,i in zip(self.mu_z_unique_networks,range(len(self.num_states_range))):
            unique[i] = int(self.mu_z_unique_networks[key])

        plt.title("\n" + "Number of unique functional networks")
        plt.xlabel("Number of states")
        plt.ylabel("Unique networks")
        plt.plot(unique)
        plt.xticks(np.arange(len(self.num_states_range)),labels=self.num_states_range)
        plt.grid(True)
        plt.plot()
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_unique_networks_vs_num_states'
        plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
        print("Saving image")
        plt.show()


    def plot_frac_occupancy(self,mouse_type=None):
        if mouse_type is not None:

            # Bar plot for each group with certainty measure of each bin corresponding to state
            indices = np.where(np.asarray(self.phenotype_list) == mouse_type)[0]

            # Delete zero rows in self.frac_occ
            frac_occ_red = self.frac_occ[indices,:]
            frac_occ_red = frac_occ_red[~np.all(frac_occ_red == 0, axis=1),:]

            mean = np.mean(frac_occ_red,axis=0)
            if np.shape(frac_occ_red)[0]>1:
                std = np.std(frac_occ_red,axis=0,ddof=1)

            fig, ax = plt.subplots()
            ax.set_ylabel('Fractional occupancy')
            ax.set_xlabel('State number')
            ax.set_xticks(np.arange(self.num_states))
            ax.set_xticklabels(np.arange(1,1+self.num_states))
            if np.shape(frac_occ_red)[0] > 1:
                ax.bar(np.arange(self.num_states), mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
                ax.set_title('Fractional occupancy of mouse type ' + mouse_type)
            else:
                ax.bar(np.arange(self.num_states), mean, align='center',alpha=0.5,capsize=10)
                ax.set_title('Fractional occupancy of mouse type ' + mouse_type)
            ax.grid(False)
            plt.tight_layout()
            plt.show()

    def plot_frac_occupancy_combined(self):
        mean = np.zeros((2,self.num_states))
        std = np.zeros((2,self.num_states))

        for phenotype, i in zip(self.phenotype_list_reduced,range(len(self.phenotype_list_reduced))):
            # Bar plot for each group with certainty measure of each bin corresponding to state
            indices = np.where(np.asarray(self.phenotype_list) == phenotype)[0]
            frac_occ_red = []
            for ind in indices:
                if self.frac_occ[ind, :].any():
                    frac_occ_red.append(self.frac_occ[ind, :])
            frac_occ_red = np.array(frac_occ_red)
            #frac_occ_red = self.frac_occ[indices,:]
            mean[i,:] = np.mean(frac_occ_red,axis=0)
            std[i,:] = np.std(frac_occ_red,axis=0,ddof=1)

        plt.ylabel('Fractional occupancy')
        plt.xlabel('State number')
        plt.xticks(np.arange(self.num_states),labels=np.arange(1,1+self.num_states))
        plt.bar(np.arange(self.num_states)-0.2, mean[0,:], yerr=std[0,:], width=0.4, alpha=0.5, ecolor='black', capsize=10,label=self.phenotype_list_reduced[0].upper())
        plt.bar(np.arange(self.num_states)+0.2, mean[1,:], yerr=std[1,:], width=0.4, alpha=0.5, ecolor='black', capsize=10,label=self.phenotype_list_reduced[1].upper())
        plt.title('Fractional occupancy of states')
        plt.legend()
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_fracocc_singlebars_num_states_' + str(self.num_states)
        plt.savefig(os.path.join(self.paths['fig_main_full_path'],fig_name + '.pdf'), format='pdf')
        print("Saving image")
        plt.show()

    def plot_frac_occupancy_multi_bar(self,mouse_type=None):
        if mouse_type is not None:
            self.gen_phenotype_colors()
            if mouse_type =='wt':
                colors = self.wt_colors
            elif mouse_type=='hom':
                colors = self.hom_colors

            # Multi bar plot for each group
            indices = np.where(np.asarray(self.phenotype_list) == mouse_type)[0]
            identifiers = []
            frac_occ_red = []
            for ind in indices:
                if self.frac_occ[ind, :].any():
                    frac_occ_red.append(self.frac_occ[ind, :])
                    identifiers.append(self.animal_list[ind])
            frac_occ_red = np.array(frac_occ_red).T.tolist()
            identifiers.insert(0,"State number")

            #frac_occ_red = self.frac_occ[indices,:].T.tolist()

            for i in range(np.size(frac_occ_red,axis=0)):
                frac_occ_red[i].insert(0,str(i+1))
            self.frac_test = frac_occ_red

            df = pd.DataFrame(frac_occ_red,columns=identifiers)

            df.plot(x="State number", y=identifiers[1:], kind="bar",rot=0,color=colors)
            plt.title("\n".join(wrap('Fractional occupancy per state for phenotype '+ mouse_type.upper(),30)) )
            plt.ylabel('Fractional occupancy')
            plt.xlabel('State number')
            plt.ylim([0,0.82])
            plt.grid(True)
            plt.tight_layout()
            fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_fracocc_multibar_num_states_' + str(self.num_states) + '_' + mouse_type
            plt.savefig(os.path.join(self.paths['fig_main_full_path'],fig_name + '.pdf'), format='pdf')
            print("Saving image")
            plt.show()

    def plot_neural_time_courses(self,rec):
        if str(rec) in self.Y_list:
            ID = self.MiceDict['recID'][rec]
            section = self.MiceDict['sectionID'][rec]

            fig, ax = plt.subplots(self.num_regions,1, figsize=(15, 9), sharex=True)
            fig.suptitle(" Reconstructed underlying neural activity for recID " + str(ID) + ", section " + str(section))
            fig.supylabel("Amplitude")
            fig.supxlabel("Time [s]")
            cmap = matplotlib.cm.get_cmap("nipy_spectral").copy()

            color_num = 250
            movement_mask = np.ma.masked_where(self.Movement_dict[str(rec)] == 0, self.Movement_dict[str(rec)])
            cmap.set_bad(color='white')

            self._movement_reduced = movement_mask[None,2*50:int(len(movement_mask)-((self.HRF_param['T_ind_HRF']-1)*50*self.param['T_res'])-self.param['T_res']*50)]

            for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):
                try:
                    margin = 0.01
                    min=np.min(self.Y_list[str(rec)][i,:])
                    max=np.max(self.Y_list[str(rec)][i,:])+margin

                    if region=='Anterior singulate':
                        region='Anterior cingulate'
                    # self._y_reduced = self.Y_list[str(rec)][i, self.HRF_param['T_ind_HRF'] - 1:]

                    ax[i].plot(self.time_reduced, self.Y_list[str(rec)][i,:], color=cmap(self.ROI_color_list[i]))
                    ax[i].set(adjustable='box', aspect='auto')
                    ax[i].set_title(region + ' area')
                    ax[i].grid(True)
                    ax[i].set_xlim([0,self.time_reduced[-1]])
                    ax_2 = ax[i].twiny()
                    ax_2.imshow(color_num * self._movement_reduced, aspect="auto", cmap=cmap,
                                norm=colors.Normalize(vmin=0, vmax=255), interpolation='None',
                                extent=[0, self.param['Nf']-(self.HRF_param['T_ind_HRF']-1), min, max])
                    ax_2.set_xticks([])
                    ax[i].set_zorder(1)  # default zorder is 0 for ax1 and ax2
                    ax[i].patch.set_visible(False)  # prevents ax1 from hiding ax2
                except:
                    print("Missing region")

            plt.tight_layout()
            self.main_fig_path_entry = os.path.join(*[self.paths['fig_main_path'], 'TE_' + str(rec)])
            if not os.path.exists(self.main_fig_path_entry):
                os.makedirs(self.main_fig_path_entry)
            fig_name = self.MainProcessingName + '_' + self.PreProcessingName + '_TE_' + str(rec) + '_Rel_lambda_threshold_' + str(self.rel_MSE_threshold)
            plt.savefig(os.path.join(self.main_fig_path_entry,fig_name + '.pdf'), format='pdf')
            print("Saving image")
            plt.show()

    def compute_fractional_movement(self):
        self.Concatenated_movement = {}
        self.Fractional_movement = np.zeros(len(self.animal_list))

        for animalID,i in zip(self.animal_list,range(len(self.animal_list))):
            self.Concatenated_movement[animalID] = []
            indices = np.where(np.asarray(self.MiceDict['animalID']) == animalID)[0]
            for idx in indices:
                self.Concatenated_movement[animalID].extend(self.Movement_dict_subsampled[str(idx)].tolist())
            self.Fractional_movement[i] = np.count_nonzero(np.array(self.Concatenated_movement[animalID])==1)/len(self.Concatenated_movement[animalID])

    def plot_group_movement(self,include_brain=True):
        self.compute_fractional_movement()
        mean_mov = np.zeros(2)
        std_mov = np.zeros(2)
        mean_act = np.zeros(2)
        std_act = np.zeros(2)

        for phenotype, i in zip(self.phenotype_list_reduced,range(len(self.phenotype_list_reduced))):
            # Bar plot for each group with certainty measure of each bin corresponding to state
            indices = np.where(np.asarray(self.phenotype_list) == phenotype)[0]
            frac_mov_red = self.Fractional_movement[indices]
            mean_mov[i] = np.mean(frac_mov_red)
            std_mov[i] = np.std(frac_mov_red,ddof=1)
            if include_brain==True:
                frac_act_red = []
                activity = (self.Y_concatenated>0).any(axis=0)
                for idx in indices:
                    frac_act_red.append(np.count_nonzero(activity[self.Y_list_indices_reduced[self.animal_list[idx]]]>0)/len(self.Y_list_indices_reduced[self.animal_list[idx]]))
                mean_act[i] = np.mean(frac_act_red)
                std_act[i] = np.std(frac_act_red, ddof=1)

        if include_brain==True:
            version = "IncludeBrain"
            plt.ylabel('Fraction of time')
            plt.xlabel('Type of activity')
            plt.xticks(np.arange(2),labels=["Brain activity","Movement"])
            plt.bar(np.arange(2)-0.2, [mean_act[0],mean_mov[0]], yerr=[std_act[0],std_mov[0]], width=0.4, alpha=0.5, ecolor='black', capsize=10,label=self.phenotype_list_reduced[0].upper())
            plt.bar(np.arange(2)+0.2, [mean_act[1],mean_mov[1]], yerr=[std_act[1],std_mov[1]], width=0.4, alpha=0.5, ecolor='black', capsize=10,label=self.phenotype_list_reduced[1].upper())

            plt.title('Percentual activity WT and HOM mice')
            plt.legend()
            plt.tight_layout()
            fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_group_movement_' + version
            plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
            print("Saving image")
            plt.show()
        else:
            version = "NoBrain"
            plt.ylabel('Fractional movement')
            plt.xlabel('Phenotype')
            plt.xticks(np.arange(2),labels=[self.phenotype_list_reduced[0].upper(),self.phenotype_list_reduced[1].upper()])
            plt.bar(np.arange(2), mean_mov, yerr=std_mov, alpha=0.5, ecolor='black', capsize=10)
            plt.title('Fractional movement of WT and HOM mice')
            plt.tight_layout()
            fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_group_movement_' + version
            plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
            print("Saving image")
            plt.show()

    def plot_scatter_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        color = [util.create_color_palette()[i] for i in self.inf_states]
        ax.scatter(self.Y_concatenated[0,:], self.Y_concatenated[1,:], self.Y_concatenated[2,:], c=color, marker='x')
        ax.set_xlabel(self.ROI_list[0])
        ax.set_ylabel(self.ROI_list[1])
        ax.set_zlabel(self.ROI_list[2])
        plt.tight_layout()
        plt.show()

    def plot_mus(self,num_states=None,clip=False,paper_fig=False):
        if num_states is None:
            mus_matrix = self.hmm.observations.mus
            num_states = self.num_states
            save=True
        else:
            mus_matrix = self.hmm_mus[str(num_states)]
            save=True

        # from matplotlib.colors import LinearSegmentedColormap
        # cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "green"], N=256)
        if clip:
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.suptitle('Inferred functional networks')
            values = 0.9*(mus_matrix.T>1e-4)
            im = ax.imshow(values, cmap='jet', vmin=0, vmax=1)
            colors = [im.cmap(im.norm(value)) for value in np.unique(values.ravel())]
            labels = ["Not active", "Active"]
            # create a patch (proxy artist) for every color
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            # put those patched as legend-handles into the legend
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            version="clip"
        elif paper_fig:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            values = 0.9 * (mus_matrix.T > 1e-4)
            im = ax.imshow(values, cmap='jet', vmin=0, vmax=1)
            colors = [im.cmap(im.norm(value)) for value in np.unique(values.ravel())]
            labels = ["Not active", "Active"]
            # create a patch (proxy artist) for every color
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            # put those patched as legend-handles into the legend
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            #plt.legend(handles=patches, bbox_to_anchor=(0, 1.02,1,0.2), loc='lower left', borderaxespad=0.,mode="expand",ncol=2)
            version = "paper_fig"
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.suptitle('Inferred functional networks')
            im = ax.imshow(mus_matrix.T, cmap='jet', vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)
            version="no_clip"

        ax.set(adjustable='box', aspect='auto')
        ax.set_ylabel('$\mathbf{\mu}_z$')
        ax.set_xlabel('State number')
        ax.set_xticks(np.arange(num_states),[str(i) for i in range(1,num_states+1)])
        ax.set_yticks(np.arange(self.num_regions),[i for i in self.ROI_list[:-2]])
        plt.tight_layout()

        if save:
            fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_hmm_mus_num_states_' + str(num_states) + '_' + version
            plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
            print("Saving image")

        plt.show()


    def plot_state_trans_prob_matrix(self):
        fig, ax = plt.subplots()
        fig.suptitle('Inferred state transition probability matrix')

        im = ax.imshow(self.hmm.transitions.transition_matrix, cmap='jet', vmin=0, vmax=1)
        ax.set(adjustable='box', aspect='auto')
        ax.set_ylabel('Current state')
        ax.set_xlabel('Next state')
        ax.set_xticks(np.arange(self.hmm.K),[str(i) for i in range(1,self.num_states+1)])
        ax.set_yticks(np.arange(self.hmm.K),[str(i) for i in range(1,self.num_states+1)])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_hmm_trans_mat_num_states_' + str(self.num_states)
        plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
        print("Saving image")
        plt.show()

    def plot_lambda_mse(self,rec):
        if str(rec) in self.Y_list:

            colors_old = plt.rcParams['axes.prop_cycle'].by_key()['color']
            cmap = matplotlib.cm.get_cmap("nipy_spectral").copy()
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            leg_list = []
            names = []
            for lam, i in zip(self.lambda_thres_list[str(rec)], range(len(self.lambda_thres_list[str(rec)]))):
                p1, = ax.plot(self.lambda_range, self.reconstruction_MSE_table[str(rec)][i, :], color=cmap(self.ROI_color_list[i]))
                p2 = ax.axvline(x=lam, ls=':', color=cmap(self.ROI_color_list[i]))
                leg_list.append((p1, p2))
                names.append("$m=" + str(i+1) + "$")
            ax.set_title(
                "RE between $\mathbf{f}_m$ and $\hat{\mathbf{f}}_m = \mathbf{H}\hat{\mathbf{y}}_m$ for different $\lambda$'s")
            ax.set_xlabel("$\lambda$")
            ax.set_xscale("log")
            ax.set_ylabel("$RE(\lambda)$")
            # plt.legend(["$m=" + str(i) + "$" for i in range(np.size(self.reconstruction_MSE_table,axis=0))])
            l = ax.legend(leg_list, names, handler_map={tuple: HandlerTuple(ndivide=None)})
            ax.grid(True)
            ax.set_xlim(self.lambda_range[0], self.lambda_range[-1])
            plt.tight_layout()
            self.main_fig_path_entry = os.path.join(*[self.paths['fig_main_path'], 'TE_' + str(rec)])
            if not os.path.exists(self.main_fig_path_entry):
                os.makedirs(self.main_fig_path_entry)
            fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_TE_' + str(rec) + '_lambdaMSEcomb_rel_MSE_threshold' + str(self.rel_MSE_threshold)
            plt.savefig(os.path.join(self.main_fig_path_entry,fig_name + '.pdf'), format='pdf')
            plt.show()

    def compute_state_life_time(self):
        animals = []
        types = []
        states = []
        life_times = []
        # Compute true life time
        for animalID,i in zip(self.animal_list,range(len(self.animal_list))):
            inf_states = self.inf_states[self.Y_list_indices_reduced[animalID]]+1
            for state in range(1,self.num_states+1):
                life_time_groups = [len(list(g)) for i, g in groupby(inf_states) if i == state]
                life_times.extend(np.divide(life_time_groups, self.Fs))
                states.extend([str(state) for _ in range(len(life_time_groups))])
                types.extend([self.phenotype_list[i].upper() for _ in range(len(life_time_groups))])
                animals.extend([self.animal_list[i] for _ in range(len(life_time_groups))])


        d = {'Type': types, 'State':states, 'Life time':life_times,'Animal':animals}
        df = pd.DataFrame(data=d)
        return df

    def compute_mean_diff_life_time_per_animal(self,mouse_type=None):
        mean_list = []
        animals = []
        df = self.compute_state_life_time()
        type_df = df[df['Type']==mouse_type.upper()]

        for animalID, i in zip(self.animal_list, range(len(self.animal_list))):
            if self.phenotype_list[i]==mouse_type:
                means = []
                red_df = type_df[type_df['Animal']==animalID]
                animals.extend([animalID])
                for state in range(1,self.num_states+1):
                    means.append(red_df[red_df['State']==str(state)]['Life time'].mean())
                mean_list.append(means)

        mean_list = np.array(mean_list).T.tolist()
        animals.insert(0, "State number")

        for i in range(np.size(mean_list, axis=0)):
            mean_list[i].insert(0, str(i + 1))

        df = pd.DataFrame(mean_list, columns=animals)
        return df,animals

    def plot_mean_diff_life_time_multi_bar(self,mouse_type=None):
        if mouse_type is not None:
            self.gen_phenotype_colors()
            if mouse_type =='wt':
                colors = self.wt_colors
            elif mouse_type=='hom':
                colors = self.hom_colors

            df,ids = self.compute_mean_diff_life_time_per_animal(mouse_type)

            df.plot(x="State number", y=ids[1:], kind="bar",rot=0,color=colors)
            plt.title("\n".join(wrap('Mean life time per state for phenotype '+ mouse_type.upper(),30)) )
            plt.ylabel('Mean life time [s]')
            plt.xlabel('State number')
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.ylim([0,2.3])
            plt.tight_layout()
            fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_lifetime_multibar_num_states_' + str(self.num_states) + '_' + mouse_type
            plt.savefig(os.path.join(self.paths['fig_main_full_path'],fig_name + '.pdf'), format='pdf')
            print("Saving image")
            plt.show()

    def plot_state_life_time(self):
        df = self.compute_state_life_time()

        fig,ax = plt.subplots(1,self.num_states,figsize=(15, 7.5))
        for i in range(1,self.num_states+1):
            sns.violinplot(x="State", y="Life time", hue="Type", data=df[df['State']==str(i)], palette="muted",ax=ax[i-1],hue_order=['WT','HOM'])
            ax[i-1].get_legend().remove()
            ax[i-1].set(xlabel=None, ylabel=None)
            ax[i-1].grid(True)
            ax[i-1].tick_params(axis='both', which='major', labelsize=15)

        fig.supxlabel("State")
        fig.supylabel("Life time [s]")
        fig.suptitle("State lifetime of both phenotypes")

        plt.legend(fontsize=15,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_life_times_'
        plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
        plt.show()


    def compute_inter_state_time(self,minimum_life_time=0):
        # Set minimum_life_time to 1 for not accounting for states that endure one index
        animals = []
        types = []
        states = []
        inter_state_times = []

        # Compute inter state time
        for animalID,i in zip(self.animal_list,range(len(self.animal_list))):
            red_indices = {}
            inter_state_time_mouse = {}
            for state in range(1, self.num_states + 1):
                red_indices[str(state)] = []
                inter_state_time_mouse[str(state)] = []

            for list_indices in self.Y_list_indices_reduced_animal[animalID]:
                inf_states = self.inf_states[np.arange(list_indices[0],list_indices[1])]+1
                for state in range(1,self.num_states+1):
                    data = np.where(inf_states==state)[0].tolist()
                    for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
                        group = (map(itemgetter(1), g))
                        group = list(map(int, group))
                        if group[-1] - group[0]>=minimum_life_time:
                            red_indices[str(state)].extend(group)
                    inter_state_time_mouse[str(state)].extend(np.diff(red_indices[str(state)]).tolist())

            for state in range(1,self.num_states+1):
                inter_state_time_group_red = [i/self.Fs for i in inter_state_time_mouse[str(state)] if i>1] #inter_state_time_mouse[str(state)][np.array(inter_state_time_mouse[str(state)])>1] / self.Fs
                inter_state_times.extend( inter_state_time_group_red )
                states.extend([str(state) for _ in range(len(inter_state_time_group_red))])
                types.extend([self.phenotype_list[i].upper() for _ in range(len(inter_state_time_group_red))])
                animals.extend([self.animal_list[i] for _ in range(len(inter_state_time_group_red))])


        d = {'Type': types, 'State':states, 'Inter state time':inter_state_times,'Animal':animals}
        df = pd.DataFrame(data=d)
        return df

    def compute_mean_diff_inter_state_time_per_animal(self,mouse_type=None):
        mean_list = []
        animals = []
        df = self.compute_inter_state_time()
        type_df = df[df['Type']==mouse_type.upper()]

        for animalID, i in zip(self.animal_list, range(len(self.animal_list))):
            if self.phenotype_list[i]==mouse_type:
                means = []
                red_df = type_df[type_df['Animal']==animalID]
                animals.extend([animalID])
                for state in range(1,self.num_states+1):
                    means.append(red_df[red_df['State']==str(state)]['Inter state time'].mean())
                mean_list.append(means)

        mean_list = np.array(mean_list).T.tolist()
        animals.insert(0, "State number")

        for i in range(np.size(mean_list, axis=0)):
            mean_list[i].insert(0, str(i + 1))

        df = pd.DataFrame(mean_list, columns=animals)
        return df,animals

    def plot_mean_diff_inter_state_time_multi_bar(self,mouse_type=None):
        if mouse_type is not None:
            self.gen_phenotype_colors()
            if mouse_type =='wt':
                colors = self.wt_colors
            elif mouse_type=='hom':
                colors = self.hom_colors

            df,ids = self.compute_mean_diff_inter_state_time_per_animal(mouse_type)

            df.plot(x="State number", y=ids[1:], kind="bar",rot=0, color=colors)
            plt.title("\n".join(wrap('Mean inter state time per state for phenotype '+ mouse_type.upper(),30)) )
            plt.ylabel('Mean inter state time [s]')
            plt.xlabel('State number')
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.ylim([0,7.7])
            plt.tight_layout()
            fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_interstate_multibar_num_states_' + str(self.num_states) + '_' + mouse_type
            plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
            print("Saving image")
            plt.show()


    def plot_inter_state_time(self,minimum_life_time=0):
        df = self.compute_inter_state_time(minimum_life_time=minimum_life_time)

        fig,ax = plt.subplots(1,self.num_states,figsize=(15, 7.5))
        for i in range(1,self.num_states+1):
            sns.violinplot(x="State", y="Inter state time", hue="Type", data=df[df['State']==str(i)], palette="muted",ax=ax[i-1],hue_order=['WT','HOM'])
            ax[i-1].get_legend().remove()
            ax[i-1].set(xlabel=None, ylabel=None)
            ax[i-1].grid(True)
            ax[i-1].tick_params(axis='both', which='major', labelsize=15)

        fig.supxlabel("State")
        fig.supylabel("Inter state time [s]")
        fig.suptitle("Inter state time of both phenotypes")

        plt.legend(fontsize=15,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_inter_state_times_'
        plt.savefig(os.path.join(self.paths['fig_main_full_path'],fig_name + '.pdf'), format='pdf')
        plt.show()


    def find_ref_mat(self,num_states,num_MonteCarlo_runs,blocksize=0,type=None):

        hmm_dict = {}
        permutation_score_matrix = np.zeros((num_MonteCarlo_runs,num_MonteCarlo_runs))

        for run in range(num_MonteCarlo_runs):
            data = self.subsampled_data(blocksize=blocksize,type=type)

            # HMM inference
            hmm = ssm.HMM(num_states, self.num_regions, observations="gaussian")
            hmm.fit(data.T, method="em", num_iters=50, init_method="kmeans")

            # Save HMM
            hmm_dict[str(run)] = hmm

        for run in range(1,num_MonteCarlo_runs):
            for idx in range(run):
                hmm_dict[str(run)].observations.mus = hmm_dict[str(run)].observations.mus > 5e-2
                hmm_dict[str(idx)].observations.mus = hmm_dict[str(idx)].observations.mus > 5e-2
                m1 = hmm_dict[str(run)].observations.mus
                perm = self.find_permutation_vec(m1, hmm_dict[str(idx)]).astype(int)
                m1 = m1.T
                m2 = hmm_dict[str(idx)].observations.mus.T
                m2 = m2[:,perm]
                permutation_score_matrix[run,idx] = len(np.where((m1[:, None] == m2[..., None]).all(0))[0])
                permutation_score_matrix[idx,run] = permutation_score_matrix[run, idx]


        ref_idx = np.argmax(np.sum(permutation_score_matrix,axis=0))
        ref_mat = hmm_dict[str(ref_idx)].observations.mus > 5e-2

        # Find mean A
        res = np.zeros((num_states,num_states))
        num = 0
        for key in hmm_dict:
            permutation = self.find_permutation_vec(ref_mat, hmm_dict[key]).astype(int)
            hmm_dict[key].permute(permutation)
            m1 = ref_mat.T
            m2 = hmm_dict[key].observations.mus.T
            if len(np.where((m1[:, None] == m2[..., None]).all(0))[0])==num_states:
                res = res + hmm_dict[key].transitions.transition_matrix
                num=num+1
        ref_trans = res/num

        return ref_mat,ref_trans


    def subsampled_data(self,blocksize=None,type=None):
        print("Blocksize = ",blocksize)

        if type=='random':
            data_length = np.size(self.Y_concatenated, axis=1)
            random_loc = np.random.randint(0,data_length-blocksize)
            data = np.delete(self.Y_concatenated, np.arange(random_loc, random_loc + blocksize), axis=1)

        elif type=='rec':
            random_locs=random.sample(range(len(self.Y_start_end_indices[:,0])), blocksize)

            list = [np.arange(self.Y_start_end_indices[random_loc, 0], self.Y_start_end_indices[random_loc, 1]) for
                    random_loc in random_locs]
            data = np.delete(self.Y_concatenated, list, axis=1)

        else:
            print("Specify type")

        print("Data length = ",len(data[0,:]))

        return data

    def compute_confidence_param_estimates(self,num_states,ref_mat,ref_trans,num_MonteCarlo_runs=None,blocksize=None,type=None):
        hmm_dict = {}

        for run in range(num_MonteCarlo_runs):
            data = self.subsampled_data(blocksize=blocksize,type=type)

            # HMM inference
            hmm = ssm.HMM(num_states, self.num_regions, observations="gaussian")
            hmm.fit(data.T, method="em", num_iters=50, init_method="kmeans")

            # Save HMM
            hmm_dict[str(run)] = hmm
            # plt.imshow(hmm.observations.mus,cmap="jet")
            # plt.show()

        score = []
        mse_list = []
        # Permute all to reference matrix
        for run in range(num_MonteCarlo_runs):
            hmm_dict[str(run)].observations.mus = hmm_dict[str(run)].observations.mus > 5e-2
            permutation = self.find_permutation_vec(ref_mat, hmm_dict[str(run)]).astype(int)
            hmm_dict[str(run)].permute(permutation)

            m1 = ref_mat.T
            m2 = hmm_dict[str(run)].observations.mus.T
            sc = len(np.where((m1[:, None] == m2[..., None]).all(0))[0])
            score.append(sc)
            if sc==num_states:
                mse_list.append(np.sum( (ref_trans-hmm_dict[str(run)].transitions.transition_matrix)**2) /(num_states**2))

        # Extract information on the consistency
        mu_percent = (np.sum(np.array(score)==num_states)) / (num_MonteCarlo_runs)
        #mse_list = [np.sum( (ref_trans-hmm_dict[key].transitions.transition_matrix)**2) /(num_states**2) for key in hmm_dict]

        return mu_percent,mse_list

    def convergence_conf_int(self,num_states,num_MonteCarlo_runs,fraction=0.5,steps=11,type=None):

        self.Y_concatenated = self.concatenate_y_ROIs()

        if type is None:
            raise ValueError("Specify type")

        if type=='random':
            self.data_fraction = np.linspace(fraction,1,steps)
            #self.data_fraction = np.array([0.9])
            Nf = np.size(self.Y_concatenated, axis=1)
            blocksizes = ((1-self.data_fraction) * Nf).astype(int)
        elif type=='rec':
            Nf = np.size(self.Y_concatenated, axis=1)
            Nf_rec = self.Y_start_end_indices[0,1]
            N = int(len(self.Y_start_end_indices[:,0]) * (1-fraction))
            blocksizes = np.arange(N+1)[::-1]
            self.data_fraction = (Nf - (blocksizes * Nf_rec)) / Nf

        self.mu_percent_list = []
        self.mean_mse_list = []
        self.std_mse_list = []
        self.num_items_mse_list = []


        ref_mat,ref_trans = self.find_ref_mat(num_states,num_MonteCarlo_runs,blocksizes[-1],type=type)

        for blocksize in blocksizes:
            mu_percent,mse_list = self.compute_confidence_param_estimates(num_states,ref_mat,ref_trans,num_MonteCarlo_runs=num_MonteCarlo_runs,blocksize=blocksize,type=type)
            self.mu_percent_list.append(mu_percent)
            self.mean_mse_list.append(np.mean(mse_list))
            self.std_mse_list.append(np.std(mse_list,ddof=1))
            self.num_items_mse_list.append(len(mse_list))
        self.num_MonteCarlo_runs = num_MonteCarlo_runs

        return self.mu_percent_list,self.mean_mse_list


    def plot_conf(self,num_states):
        plt.subplots(figsize=(7, 5))
        plt.xlabel("Fraction of total data")
        plt.ylabel("Relative frequency of occurrence")
        plt.title("Consistency of functional network inference")
        plt.plot(self.data_fraction,self.mu_percent_list)
        plt.grid(True)
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_mu_z_consistency_NumStates' + str(num_states) + '_MonteCarlo_' + str(self.num_MonteCarlo_runs)
        plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
        plt.show()

        plt.subplots(figsize=(7, 5))
        plt.xlabel("Fraction of total data")
        plt.ylabel("MSE")
        plt.title("Consistency of inferring $\mathbf{A}$")
        plt.plot(self.data_fraction,self.mean_mse_list)
        conf_low = np.array(self.mean_mse_list) - 1.96 * np.array(self.std_mse_list) / np.sqrt(self.num_items_mse_list)
        conf_high = np.array(self.mean_mse_list) + 1.96 * np.array(self.std_mse_list) / np.sqrt(self.num_items_mse_list)
        plt.fill_between(self.data_fraction, conf_high, conf_low, alpha=.3)
        plt.grid(True)
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_A_mean_consistency_NumStates' + str(num_states) + '_MonteCarlo_' + str(self.num_MonteCarlo_runs)
        plt.savefig(os.path.join(self.paths['fig_main_full_path'], fig_name + '.pdf'), format='pdf')
        plt.show()


    def main_dict_nparray_to_list(self):
        self.HRF = self.HRF.tolist()
        self.time = self.time.tolist()
        self.time_reduced = self.time_reduced.tolist()

        for rec in range(self.num_recs):
            try:
                self.Y_list[str(rec)] = self.Y_list[str(rec)].tolist()
                self.lambda_thres_list[str(rec)] = self.lambda_thres_list[str(rec)].tolist()
                self.reconstruction_MSE_table[str(rec)] = self.reconstruction_MSE_table[str(rec)].tolist()
            except:
                pass

    def main_list_to_dict_nparray(self):
        self.HRF = np.array(self.HRF)
        self.time = np.array(self.time)
        self.time_reduced = np.array(self.time_reduced)

        for rec in range(self.num_recs):
            try:
                self.Y_list[str(rec)] = np.array(self.Y_list[str(rec)])
                self.lambda_thres_list[str(rec)] = np.array(self.lambda_thres_list[str(rec)])
                self.reconstruction_MSE_table[str(rec)] = np.array(self.reconstruction_MSE_table[str(rec)])

            except:
                pass

    def import_main_json(self):
        # Import only parameters necessary to initialize script
        with open(self.paths['json_main_path']) as json_file:
            data = json.load(json_file)

        self.PreProcessingName = data['PreProcessingName']
        self.MainProcessingName = data['MainProcessingName']
        self.rel_MSE_threshold = data['rel_MSE_threshold']
        self.HRF_param = data['HRF_param']
        self.HRF = data['HRF']
        self.Y_list = data['Y_list']
        self.lambda_thres_list = data['lambda_thres_list']
        self.reconstruction_MSE_table = data['reconstruction_MSE_table']
        self.time = data['time']
        self.time_reduced = data['time_reduced']
        self.mus_target = np.array(data['mus_target'])
        self.lambda_range = np.array(data['lambda_range'])

        self.main_list_to_dict_nparray()

    def export_main_json(self):
        self.main_dict_nparray_to_list()

        export_dict = {}
        export_dict['PreProcessingName'] = self.PreProcessingName
        export_dict['MainProcessingName'] = self.MainProcessingName
        export_dict['rel_MSE_threshold'] = self.rel_MSE_threshold
        export_dict['HRF_param'] = self.HRF_param
        export_dict['HRF'] = self.HRF
        export_dict['Y_list'] = self.Y_list
        export_dict['lambda_thres_list'] = self.lambda_thres_list
        export_dict['reconstruction_MSE_table'] = self.reconstruction_MSE_table
        export_dict['time'] = self.time
        export_dict['time_reduced'] = self.time_reduced
        export_dict['mus_target'] = self.mus_target.tolist()
        export_dict['lambda_range'] = self.lambda_range.tolist()

        with open(self.paths['json_main_path'], 'w') as json_path:
            json.dump(export_dict, json_path, indent=4)

if __name__ == "__main__":
    PreProcessingName = "PRE_32"
    MainProcessingID = 12

    # mus_target is here to determine the order of inferred functional networks.
    mus_target = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,0]])
    # mus_target = np.array([[0,0,0,0],[0,1,1,1],[0,0,1,1],[0,1,0,0],[0,0,1,0]])

    overwrite = False  # Overwrite MAIN_x.json
    specific_rec = []  # Enter entry for testing purpose (only analyses the specific entry)

    # Determine MainProcessingName
    MainProcessingName = "MAIN_TEST" if len(specific_rec)!=0 else "MAIN_" + str(MainProcessingID)

    # Execute main-processing steps
    MainProcessing(PreProcessingName,MainProcessingName,overwrite=overwrite, specific_rec=specific_rec, mus_target=mus_target)
    main = MainProcessing(PreProcessingName,MainProcessingName)

    recs = range(main.num_recs) if len(specific_rec) == 0 else specific_rec

    # Plotting
    for rec in recs:
        if str(rec) in main.F_final:
            # main.plot_time_courses(rec)
            #main.plot_neural_time_courses(rec)
            #main.plot_lambda_mse(rec)
            pass

    if len(specific_rec) == 0:

    #Find number of states that result in largest difference
        # main.find_num_states()
        # main.plot_rss_vs_num_states()
        # main.plot_unique_networks_vs_num_states()
        # for i in range(2,8):
        #     main.plot_mus(num_states=i,clip=False)

    # Consistency of parameter estimates
    #     mu_percent_list,mean_mse_list = main.convergence_conf_int(5,num_MonteCarlo_runs=10,type='rec',fraction=0.2)
    #     main.plot_conf(5)

        # mu_percent_list,mean_mse_list = main.convergence_conf_int(4, num_MonteCarlo_runs=100,type='rec',fraction=0.2)
        # main.plot_conf(4)

    # Group Analysis
    #     main.group_analysis(num_states=4)
    #
    #     main.plot_group_movement()
    #     main.plot_frac_occupancy_combined()
    #     main.plot_frac_occupancy_multi_bar(mouse_type='wt')
    #     main.plot_frac_occupancy_multi_bar(mouse_type='hom')
    #     main.plot_mus()
    #     main.plot_state_trans_prob_matrix()
    #     main.plot_scatter_3D()
    #
    #     main.plot_state_life_time()
    #     main.plot_mean_diff_life_time_multi_bar(mouse_type='wt')
    #     main.plot_mean_diff_life_time_multi_bar(mouse_type='hom')
    #
    #     main.plot_inter_state_time()
    #     main.plot_mean_diff_inter_state_time_multi_bar(mouse_type='wt')
    #     main.plot_mean_diff_inter_state_time_multi_bar(mouse_type='hom')



    # Different number of states
        main.group_analysis(num_states=4)
        # main.plot_frac_occupancy_combined()
        # main.group_analysis(num_states=5)
        # main.plot_frac_occupancy_combined()
        main.plot_mus(paper_fig=True)
        pass
