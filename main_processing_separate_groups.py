import random
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import ssm as ssm
import copy
import os
import json
import matplotlib.patches as mpatches
import util

from textwrap import wrap
from scipy.optimize import linear_sum_assignment
from pre_processing import PreProcessing
from sklearn.preprocessing import normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.linalg import norm


class MainProcessingSepGroup(PreProcessing):
    def __init__(self,PreProcessingName,MainProcessingName):
        super().__init__(PreProcessingName)  # Do not perform new data extraction steps here
        self.MainProcessingName = MainProcessingName

        # Create paths
        self.paths = util.create_paths_dict(MainProcessingName=MainProcessingName)

        # Color palette
        self.colors = util.create_color_palette()

        self.phenotype_list_reduced = ["wt","hom"]

        # Check if extracted MainProcessingName already exists in simulation folder
        if os.path.exists(self.paths['json_main_path']):
            print("Using MAIN processing data from file " + self.MainProcessingName + " located at " + self.paths['json_main_path'])
            self.import_main_json()


    def separate_group_analysis(self,num_states):
        """
        HMM inference on separate groups of mice.

        :param num_states: Number of states K
        :return: No return
        """
        self.num_states=num_states

        # Concatenate reconstructed neural data
        self.Y_concatenated = self.concatenate_y_ROIs()

        # Check if mus_target is usable, otherwise, create a consistent mus_target over different runs
        if np.size(self.mus_target, axis=0) != self.num_states:
            random_state = np.random.RandomState(seed=0)
            self.mus_target = random_state.randint(2, size=(self.num_states-1, self.num_regions)).astype(int)
            self.mus_target = np.insert(self.mus_target,0,np.zeros(self.num_regions),axis=0)


        # Apply HMM for WT
        self.hmm_wt = ssm.HMM(self.num_states, self.num_regions, observations="gaussian")
        self.hmm_lls_wt = self.hmm_wt.fit(self.Y_concatenated['wt'].T, method="em", num_iters=50, init_method="kmeans")
        self.inf_states_wt = self.hmm_wt.most_likely_states(self.Y_concatenated['wt'].T) # Viterbi

        # Permute
        permutation_wt = self.find_permutation_vec(self.mus_target,self.hmm_wt).astype(int)
        self.hmm_wt.permute(permutation_wt)
        self.inf_states_wt = self.hmm_wt.most_likely_states(self.Y_concatenated['wt'].T)

        # Apply HMM for HOM
        self.hmm_hom = ssm.HMM(self.num_states, self.num_regions, observations="gaussian")
        self.hmm_lls_hom = self.hmm_hom.fit(self.Y_concatenated['hom'].T, method="em", num_iters=50, init_method="kmeans")
        self.inf_states_hom = self.hmm_hom.most_likely_states(self.Y_concatenated['hom'].T)  # Viterbi

        # Permute
        permutation = self.find_permutation_vec(self.mus_target,self.hmm_hom).astype(int)
        self.hmm_hom.permute(permutation)
        self.inf_states_hom = self.hmm_hom.most_likely_states(self.Y_concatenated['hom'].T)


    def find_num_states(self):
        """
        Find number of states K, based on uniqueness of functional networks (captured by mu_z)
        :return: Plots
        """

        # Concatenate reconstructed neural data (includes rescaling of means)
        self.Y_concatenated = self.concatenate_y_ROIs()

        self.num_states_range = np.arange(2,9)
        self.mu_z_unique_networks = {}
        self.hmm_mus = {}

        for type in self.phenotype_list_reduced:
            self.mu_z_unique_networks[type] = {}
            self.hmm_mus[type] = {}

            for num_states in self.num_states_range:
                # Apply HMM
                hmm = ssm.HMM(num_states, self.num_regions, observations="gaussian")
                hmm_lls = hmm.fit(self.Y_concatenated[type].T, method="em", num_iters=50, init_method="kmeans")
                inf_states = hmm.most_likely_states(self.Y_concatenated[type].T)  # Viterbi
                self.hmm_mus[type][str(num_states)] = hmm.observations.mus # Store fitted means

                # Count functional network doubles
                activity_detection_matrix = normalize(hmm.observations.mus>1e-4, norm="l2")
                self.mu_z_unique_networks[type][str(num_states)] = np.size(np.unique(activity_detection_matrix,axis=0),axis=0)

            self.plot_unique_networks_vs_num_states(type)

    def concatenate_y_ROIs(self):
        """
        Concatenate all neural activity time courses of a group of mice. Stored in Y_concatenated[mouse type]
        :return: Y_concatenated
        """
        Y_concatenated = {}

        for type in self.phenotype_list_reduced:
            Y_concatenated[type]=None
            # Concatenate all F matrices
            for rec in self.Y_list:
                if self.MiceDict['phenotype'][int(rec)]==type:
                    # Rescale means to same level (unity), due to different orignal signal levels/powers
                    means = [np.mean(self.Y_list[str(rec)][i, :][self.Y_list[str(rec)][i, :] > np.max(self.Y_list[str(rec)][i, :])*1e-3]) for i in range(self.num_regions)]
                    Y_list_rescaled = self.Y_list[str(rec)] / np.array(means)[:, None]
                    if Y_concatenated[type] is None:
                        Y_concatenated[type] = Y_list_rescaled
                    else:
                        Y_concatenated[type] = np.append(Y_concatenated[type],Y_list_rescaled,axis=1)

        self.Nf_rec = len(self.Y_list[str(rec)][0, :])

        return Y_concatenated

    def concatenate_y_ROIs_random_rec(self,min_changes = 1):
        """

        :param min_changes: Minimum recordings that are interchanged. Default==1.
        :return:
        """
        Y_concatenated_random = {}

        # Change order of self.MiceDict['phenotype'], and ensure the re-ordering is different
        phenotypes = copy.deepcopy(self.MiceDict['phenotype'])
        flag=True
        while flag:
            random.shuffle(phenotypes)
            elems = np.sum([x == y for x, y in zip(phenotypes, self.MiceDict['phenotype'])])
            if elems < len(phenotypes)-min_changes and elems > min_changes:
                flag=False

        for type in self.phenotype_list_reduced:
            Y_concatenated_random[type]=None
            # Concatenate all F matrices
            for rec in self.Y_list:#range(self.num_recs):
                if phenotypes[int(rec)]==type:
                    # Rescale means to same level (unity), due to different orignal signal levels
                    means = [np.mean(self.Y_list[str(rec)][i, :][self.Y_list[str(rec)][i, :] > np.max(self.Y_list[str(rec)][i, :])*1e-3]) for i in range(self.num_regions)]
                    Y_list_rescaled = self.Y_list[str(rec)] / np.array(means)[:, None]
                    #means_rescaled = [np.mean(Y_list_rescaled[i, :][Y_list_rescaled[i, :] > 0]) for i in range(self.num_regions)]
                    if Y_concatenated_random[type] is None:
                        Y_concatenated_random[type] = Y_list_rescaled
                    else:
                        Y_concatenated_random[type] = np.append(Y_concatenated_random[type],Y_list_rescaled,axis=1)

        return Y_concatenated_random,phenotypes

    def concatenate_y_ROIs_random_animal(self,min_changes = 0):
        Y_concatenated_random = {}

        # Change order of self.MiceDict['phenotype'], not identical
        phenotypes = copy.deepcopy(self.phenotype_list)
        flag=True
        while flag:
            random.shuffle(phenotypes)
            elems = np.sum([x == y for x, y in zip(phenotypes, self.MiceDict['phenotype'])])
            if elems < len(phenotypes)-min_changes and elems > min_changes:
                flag=False

        for type in self.phenotype_list_reduced:
            Y_concatenated_random[type]=None
            # Concatenate all F matrices
            for animalID,i in zip(self.animal_list,range(len(self.animal_list))):
                indices = np.where(np.asarray(self.MiceDict['animalID']) == animalID)[0]

                if phenotypes[i] == type:
                    for rec in indices:
                        # Rescale means to same level (unity), due to different orignal signal levels
                        means = [np.mean(self.Y_list[str(rec)][i, :][self.Y_list[str(rec)][i, :] > np.max(self.Y_list[str(rec)][i, :])*1e-3]) for i in range(self.num_regions)]
                        Y_list_rescaled = self.Y_list[str(rec)] / np.array(means)[:, None]
                        #means_rescaled = [np.mean(Y_list_rescaled[i, :][Y_list_rescaled[i, :] > 0]) for i in range(self.num_regions)]
                        if Y_concatenated_random[type] is None:
                            Y_concatenated_random[type] = Y_list_rescaled
                        else:
                            Y_concatenated_random[type] = np.append(Y_concatenated_random[type],Y_list_rescaled,axis=1)

        return Y_concatenated_random,phenotypes

    def create_lists(self):

        # Create lists
        self.animal_list = list(set(self.MiceDict['animalID']))
        self.phenotype_list = []  # corresponding to self.animal_list
        for animalID in self.animal_list:
            indices = np.where(np.asarray(self.MiceDict['animalID']) == animalID)[0]
            self.phenotype_list.append(self.MiceDict['phenotype'][indices[0]])

    def find_permutation_vec(self,ref_mus,hmm_permutation):
        # Find permutation based on the mean signature of networks, using cosine similarity
        ref_vecs = copy.deepcopy(ref_mus)
        hmm_vecs = copy.deepcopy(hmm_permutation.observations.mus>1e-5)

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

            #print(self.cost_matrix)
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

    def find_ref_mat(self,num_states,num_MonteCarlo_runs,mouse_type=None):

        hmm_dict = {}
        permutation_score_matrix = np.zeros((num_MonteCarlo_runs,num_MonteCarlo_runs))

        for run in range(num_MonteCarlo_runs):
            data = self.Y_concatenated[mouse_type]

            # HMM inference
            hmm = ssm.HMM(num_states, self.num_regions, observations="gaussian")
            hmm.fit(data.T, method="em", num_iters=50, init_method="kmeans")

            # Save HMM
            hmm_dict[str(run)] = hmm
            hmm_dict[str(run)].observations.mus = hmm_dict[str(run)].observations.mus > 5e-2

        for run in range(1,num_MonteCarlo_runs):
            for idx in range(run):
                m1 = hmm_dict[str(run)].observations.mus
                perm = self.find_permutation_vec(m1, hmm_dict[str(idx)]).astype(int)
                m1 = m1.T
                m2 = hmm_dict[str(idx)].observations.mus.T
                m2 = m2[:,perm]
                permutation_score_matrix[run,idx] = len(np.where((m1[:, None] == m2[..., None]).all(0))[0])
                permutation_score_matrix[idx,run] = permutation_score_matrix[run, idx]


        ref_idx = np.argmax(np.sum(permutation_score_matrix,axis=0))
        hmm_dict[str(ref_idx)].permute(self.find_permutation_vec(self.mus_target,hmm_dict[str(ref_idx)]).astype(int))
        ref_mat = hmm_dict[str(ref_idx)].observations.mus > 5e-2

        self.hmm_dict=hmm_dict

        # Find mean A
        self.dict_A[mouse_type]={}
        res = np.zeros((num_states,num_states))
        num = 0
        for key in hmm_dict:
            permutation = self.find_permutation_vec(ref_mat, hmm_dict[key]).astype(int)
            hmm_dict[key].permute(permutation)
            m1 = ref_mat.T
            m2 = hmm_dict[key].observations.mus.T
            if len(np.where((m1[:, None] == m2[..., None]).all(0))[0])==num_states:
                self.dict_A[mouse_type][str(num)] = hmm_dict[key].transitions.transition_matrix
                res = res + hmm_dict[key].transitions.transition_matrix
                num=num+1

        extra_runs = 0
        while num<num_MonteCarlo_runs:
            extra_runs = extra_runs+1
            # HMM inference
            hmm = ssm.HMM(num_states, self.num_regions, observations="gaussian")
            hmm.fit(self.Y_concatenated[mouse_type].T, method="em", num_iters=50, init_method="kmeans")
            hmm.observations.mus = hmm.observations.mus > 5e-2
            hmm.permute(self.find_permutation_vec(ref_mat, hmm).astype(int))
            m1 = ref_mat.T
            m2 = hmm.observations.mus.T
            if len(np.where((m1[:, None] == m2[..., None]).all(0))[0]) == num_states:
                self.dict_A[mouse_type][str(num)] = hmm.transitions.transition_matrix
                res = res + hmm.transitions.transition_matrix
                num=num+1

        ref_trans = res / num

        self.rel_occ[mouse_type] = num_MonteCarlo_runs/(extra_runs+num_MonteCarlo_runs)

        return ref_mat,ref_trans

    def convergence_conf_int(self,num_states,num_MonteCarlo_runs,min_changes,shuffle_method,mse_range=None):
        self.num_states,self.num_MonteCarlo_runs,self.min_changes,self.shuffle_method,self.mse_range = \
            num_states, num_MonteCarlo_runs ,min_changes,shuffle_method, mse_range

        self.create_lists()
        self.Y_concatenated = self.concatenate_y_ROIs()
        self.mse_list = []
        self.dict_A={}
        self.dict_rnd_A={}
        self.dict_rnd_A['wt']={}
        self.dict_rnd_A['hom']={}
        self.rel_occ={}

        self.ref_mat_wt,self.ref_trans_wt = self.find_ref_mat(num_states,num_MonteCarlo_runs,mouse_type='wt')
        self.ref_mat_hom,self.ref_trans_hom = self.find_ref_mat(num_states,num_MonteCarlo_runs,mouse_type='hom')

        # Compute true MSEs
        assert len(self.dict_A['wt']) == len(self.dict_A['hom'])
        self.true_mse_conf = []
        for key in self.dict_A['wt']:
            if self.mse_range is not None:
                if isinstance(self.mse_range[0], str):
                    indices = self.mse_range[0].split(",")
                    current = int(indices[0])
                    next = int(indices[-1])
                    t1 = self.dict_A['wt'][key]#[1:4,1:4]
                    t2 = self.dict_A['hom'][key]#[1:4,1:4]
                    t1 /= t1.sum(axis=1, keepdims=True)
                    t2 /= t2.sum(axis=1, keepdims=True)
                    self.true_mse_conf.append(abs(t2[current, next] - t1[current, next]))
                else:
                    start=self.mse_range[0]
                    end=self.mse_range[-1]
                    t1 = self.dict_A['wt'][key][start:end,start:end]
                    t2 = self.dict_A['hom'][key][start:end,start:end]
                    #print(t2)
                    t1 /= t1.sum(axis=1, keepdims=True)
                    t2 /= t2.sum(axis=1, keepdims=True)
                    self.true_mse_conf.append(np.sum((t1-t2)**2)/((end-start)**2))
            else:
                self.true_mse_conf.append(np.sum((self.dict_A['wt'][key]-self.dict_A['hom'][key])**2)/(num_states**2))

        self.phenotype_groups=[]
        run=0
        while run<num_MonteCarlo_runs:
            if self.shuffle_method == 'animal':
                concatenate_y_ROIs_random,rnd_phenotype_group = self.concatenate_y_ROIs_random_animal(min_changes=self.min_changes)
            elif self.shuffle_method == 'rec':
                concatenate_y_ROIs_random,rnd_phenotype_group = self.concatenate_y_ROIs_random_rec(min_changes=self.min_changes)
            else:
                raise ValueError(" Select shuffle method. Options: 'animal', 'rec' ")

            # Apply HMM for WT (not true WT)
            hmm_wt = ssm.HMM(num_states, self.num_regions, observations="gaussian")
            hmm_wt.fit(concatenate_y_ROIs_random['wt'].T, method="em", num_iters=50,init_method="kmeans")

            # Apply HMM for HOM (not true HOM)
            hmm_hom = ssm.HMM(num_states, self.num_regions, observations="gaussian")
            hmm_hom.fit(concatenate_y_ROIs_random['hom'].T, method="em", num_iters=50,init_method="kmeans")

            # Permute
            permutation_wt = self.find_permutation_vec(self.ref_mat_wt, hmm_wt).astype(int)
            hmm_wt.permute(permutation_wt)
            permutation = self.find_permutation_vec(self.ref_mat_wt, hmm_hom).astype(int)
            hmm_hom.permute(permutation)

            hmm_wt.observations.mus = hmm_wt.observations.mus>5e-2
            hmm_hom.observations.mus = hmm_hom.observations.mus>5e-2
            m1_wt = hmm_wt.observations.mus.T
            m1_hom = hmm_hom.observations.mus.T
            m2 = self.mus_target.T
            sc1 = len(np.where((m1_wt[:, None] == m2[..., None]).all(0))[0])
            sc2 = len(np.where((m1_hom[:, None] == m2[..., None]).all(0))[0])

            # fig,ax = plt.subplots(1,2)
            # ax[0].imshow(m1_wt,cmap="jet")
            # ax[1].imshow(m1_hom,cmap="jet")
            # plt.show()

            if sc1==num_states and sc2==num_states:
                if self.mse_range is not None:
                    if isinstance(self.mse_range[0], str):
                        indices = self.mse_range[0].split(",")
                        current = int(indices[0])
                        next = int(indices[-1])
                        t1 = hmm_wt.transitions.transition_matrix#[1:4,1:4]
                        t2 = hmm_hom.transitions.transition_matrix#[1:4,1:4]
                        t1 /= t1.sum(axis=1, keepdims=True)
                        t2 /= t2.sum(axis=1, keepdims=True)
                        self.mse_list.append(abs(t2[current, next] - t1[current, next]))
                    else:
                        start = self.mse_range[0]
                        end = self.mse_range[-1]
                        t1 = hmm_wt.transitions.transition_matrix[start:end,start:end]
                        t2 = hmm_hom.transitions.transition_matrix[start:end,start:end]
                        t1 /= t1.sum(axis=1, keepdims=True)
                        t2 /= t2.sum(axis=1, keepdims=True)
                        self.mse_list.append(np.sum((t1-t2)**2)/((end-start)**2))
                else:
                    t1 = hmm_wt.transitions.transition_matrix
                    t2 = hmm_hom.transitions.transition_matrix
                    self.mse_list.append(np.sum((t1-t2)**2) /(num_states**2))
                run=run+1
                self.dict_rnd_A['wt'][str(run)]=t1
                self.dict_rnd_A['hom'][str(run)]=t2
                print('Run', run, 'of total Monte Carlo runs', num_MonteCarlo_runs)
                self.phenotype_groups.append(rnd_phenotype_group)
        file_name = self.PreProcessingName + '_' + self.MainProcessingName + '_MSE_distribution_' + str(
            self.num_states) + '_minchanges_' + str(self.min_changes) + '_MC_' + str(
            self.num_MonteCarlo_runs) + '_shuffmethod_' + self.shuffle_method + '_mse_range_' + str(
            self.mse_range[0]) + str(self.mse_range[-1])

        print('Lower effect percentage:',len(np.where(self.mse_list<np.mean(self.true_mse_conf))[0])/len(self.mse_list),'%')
        np.save(os.path.join(self.paths['fig_main_sep_path'],file_name+'.npy'),np.array([self.true_mse_conf,self.mse_list]))
        return self.true_mse_conf,self.mse_list



    def subsampled_data(self,blocksize=None,mouse_type=None):
        print("Blocksize = ",blocksize)

        num_recs = round(len(self.Y_concatenated[mouse_type][0,:])/self.Nf_rec)
        random_locs=random.sample(range(num_recs), blocksize)

        list = [np.arange(self.Nf_rec*random_loc, self.Nf_rec*(random_loc+1)) for
                random_loc in random_locs]
        data = np.delete(self.Y_concatenated[mouse_type], list, axis=1)

        print("Data length = ",len(data[0,:]))

        return data

    def compute_confidence_param_estimates(self,num_states,ref_mat,ref_trans,num_MonteCarlo_runs=None,blocksize=None,mouse_type=None):
        hmm_dict = {}

        for run in range(num_MonteCarlo_runs):
            data = self.subsampled_data(blocksize=blocksize,mouse_type=mouse_type)

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


    def functional_network_consistency(self,num_states,num_MonteCarlo_runs,fraction=0.5,steps=11,mouse_type=None):
        self.dict_A={}
        self.rel_occ={}

        self.Y_concatenated = self.concatenate_y_ROIs()

        Nf = np.size(self.Y_concatenated[mouse_type], axis=1)
        num_recs = round(len(self.Y_concatenated[mouse_type][0,:])/self.Nf_rec)
        N = int(num_recs * (1-fraction))
        blocksizes = np.arange(N+1)[::-1]
        self.data_fraction = (Nf - (blocksizes * self.Nf_rec)) / Nf

        mu_percent_list = []
        mean_mse_list = []
        std_mse_list = []
        num_items_mse_list = []

        ref_mat,ref_trans = self.find_ref_mat(num_states,num_MonteCarlo_runs,mouse_type=mouse_type)

        for blocksize in blocksizes:
            mu_percent,mse_list = self.compute_confidence_param_estimates(num_states,ref_mat,ref_trans,num_MonteCarlo_runs=num_MonteCarlo_runs,blocksize=blocksize,mouse_type=mouse_type)
            mu_percent_list.append(mu_percent)
            mean_mse_list.append(np.mean(mse_list))
            std_mse_list.append(np.std(mse_list,ddof=1))
            num_items_mse_list.append(len(mse_list))
        self.num_MonteCarlo_runs = num_MonteCarlo_runs

        return mu_percent_list,mean_mse_list

    # Plotting
    def plot_conf(self,num_states,mu_percent_list,mouse_type):
        plt.subplots(figsize=(7, 5))
        plt.xlabel("Fraction of total data")
        plt.ylabel("Relative frequency of occurrence")
        plt.title("Consistency of functional network inference")
        plt.plot(self.data_fraction,mu_percent_list)
        plt.grid(True)
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_mu_z_consistency_NumStates' + str(num_states) + '_MonteCarlo_' + str(self.num_MonteCarlo_runs) + "_mouse_" + mouse_type.upper() + "TEST"
        plt.savefig(os.path.join(self.paths['fig_main_sep_path'],fig_name + '.pdf'), format='pdf')
        plt.show()

        # plt.subplots(figsize=(7, 5))
        # plt.xlabel("Fraction of total data")
        # plt.ylabel("MSE")
        # plt.title("Consistency of inferring $\mathbf{A}$")
        # plt.plot(self.data_fraction,self.mean_mse_list)
        # conf_low = np.array(self.mean_mse_list) - 1.96 * np.array(self.std_mse_list) / np.sqrt(self.num_items_mse_list)
        # conf_high = np.array(self.mean_mse_list) + 1.96 * np.array(self.std_mse_list) / np.sqrt(self.num_items_mse_list)
        # plt.fill_between(self.data_fraction, conf_high, conf_low, alpha=.3)
        # plt.grid(True)
        # plt.tight_layout()
        # fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_A_mean_consistency_NumStates' + str(num_states) + '_MonteCarlo_' + str(self.num_MonteCarlo_runs)
        # plt.savefig(os.path.join(self.paths['fig_main_sep_path'],fig_name + '.pdf'), format='pdf')
        # plt.show()


    def plot_unique_networks_vs_num_states(self,type=None):
        if type is None:
            raise ValueError("Choose type")

        unique = np.zeros(len(self.num_states_range),dtype=int)
        for key,i in zip(self.mu_z_unique_networks[type],range(len(self.num_states_range))):
            unique[i] = int(self.mu_z_unique_networks[type][key])

        plt.title("\n".join(wrap("Number of unique functional networks of type "+ type.upper(),30)))
        plt.xlabel("Number of states")
        plt.ylabel("Unique networks")
        plt.plot(unique)
        plt.xticks(np.arange(len(self.num_states_range)),labels=self.num_states_range)
        plt.grid(True)
        plt.plot()
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_unique_networks_vs_num_states_type_' + type
        plt.savefig(os.path.join(self.paths['fig_main_sep_path'],fig_name + '.pdf'), format='pdf')
        print("Saving image")
        plt.show()

    def plot_scatter_3D(self,type=None):
        if type=="wt":
            inf_states = self.inf_states_wt
        elif type=="hom":
            inf_states = self.inf_states_hom
        else:
            raise ValueError("Choose type")

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.set_title("Assigned states of neural activity of type " + type.upper())
        color = [self.colors[i] for i in inf_states]
        ax.scatter(self.Y_concatenated[type][0,:], self.Y_concatenated[type][1,:], self.Y_concatenated[type][2,:], c=color, marker='x')
        ax.set_xlabel(self.ROI_list[0])
        ax.set_ylabel(self.ROI_list[1])
        ax.set_zlabel(self.ROI_list[2])
        plt.tight_layout()
        plt.show()

    def plot_mus(self,type=None,clip=False):
        if type=="wt":
            mus_matrix = self.hmm_wt.observations.mus
        elif type=="hom":
            mus_matrix = self.hmm_hom.observations.mus
        else:
            raise ValueError("Choose type")

        fig, ax = plt.subplots()
        fig.suptitle('Inferred functional networks of type ' + type.upper() )
        if clip:
            values = 0.9*(mus_matrix.T>1e-4)
            im = ax.imshow(values, cmap='jet', vmin=0, vmax=1)
            colors = [im.cmap(im.norm(value)) for value in np.unique(values.ravel())]
            labels = ["Not active", "Active"]
            # create a patch (proxy artist) for every color
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            # put those patched as legend-handles into the legend
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            version="clip"
        else:
            im = ax.imshow(mus_matrix.T, cmap='jet', vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)
            version="no_clip"

        ax.set(adjustable='box', aspect='auto')
        ax.set_ylabel('$\mathbf{\mu}_z$')
        ax.set_xlabel('State number')
        ax.set_xticks(np.arange(self.num_states),[str(i) for i in range(1,self.num_states+1)])
        ax.set_yticks(np.arange(self.num_regions),[i for i in self.ROI_list[:-2]])
        plt.tight_layout()

        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_hmm_mus_num_states_' + str(self.num_states) + '_' + version + '_type_' + type
        plt.savefig(os.path.join(self.paths['fig_main_sep_path'], fig_name + '.pdf'), format='pdf')
        print("Saving image")

        plt.show()

    def plot_REF_mus(self, type=None):
        if type == "wt":
            mus_matrix = self.ref_mat_wt
        elif type == "hom":
            mus_matrix = self.ref_mat_hom
        else:
            raise ValueError("Choose type")

        fig, ax = plt.subplots()
        fig.suptitle('Inferred functional networks of type ' + type.upper())
        im = ax.imshow(mus_matrix.T, cmap='jet', vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        version = "no_clip"
        ax.set(adjustable='box', aspect='auto')
        ax.set_ylabel('$\mathbf{\mu}_z$')
        ax.set_xlabel('State number')
        ax.set_xticks(np.arange(self.num_states), [str(i) for i in range(1, self.num_states + 1)])
        ax.set_yticks(np.arange(self.num_regions), [i for i in self.ROI_list[:-2]])
        plt.tight_layout()

        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_hmm_REF_mus_num_states_' + str(
            self.num_states) + '_' + version + '_type_' + type + '_MC_' + str(self.num_MonteCarlo_runs) + '_shuffmethod_' + self.shuffle_method
        plt.savefig(os.path.join(self.paths['fig_main_sep_path'], fig_name + '.pdf'), format='pdf')
        print("Saving image")

        plt.show()

    def plot_state_trans_prob_matrix(self,type=None):
        if type=="wt":
            trans_mat = self.hmm_wt.transitions.transition_matrix
        elif type=="hom":
            trans_mat = self.hmm_hom.transitions.transition_matrix
        else:
            raise ValueError("Choose type")

        fig, ax = plt.subplots()
        fig.suptitle('Inferred state transition probability matrix ('+ type.upper()+')')
        im = ax.imshow(trans_mat, cmap='jet', vmin=0, vmax=1)
        ax.set(adjustable='box', aspect='auto')
        ax.set_ylabel('Current state')
        ax.set_xlabel('Next state')
        ax.set_xticks(np.arange(self.num_states),[str(i) for i in range(1,self.num_states+1)])
        ax.set_yticks(np.arange(self.num_states),[str(i) for i in range(1,self.num_states+1)])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_hmm_trans_mat_num_states_' + str(self.num_states) + '_type_' + type
        plt.savefig(os.path.join(self.paths['fig_main_sep_path'], fig_name + '.pdf'), format='pdf')
        print("Saving image")
        plt.show()

    def plot_MEAN_state_trans_prob_matrix(self,type=None):
        if type=="wt":
            trans_mat = self.ref_trans_wt
        elif type=="hom":
            trans_mat = self.ref_trans_hom
        else:
            raise ValueError("Choose type")

        N_a = int(np.sum([animal==type for animal in self.phenotype_list]))
        fig, ax = plt.subplots()
        #fig.suptitle("\n".join(wrap('Mean inferred state transition probability matrix ('+ type.upper()+ ', $N_{a}$=' + str(N_a)+')',40)))
        im = ax.imshow(trans_mat, cmap='jet', vmin=0, vmax=1)
        ax.set(adjustable='box', aspect='auto')
        ax.set_ylabel('Current state')
        ax.set_xlabel('Next state')
        ax.set_xticks(np.arange(self.num_states),[str(i) for i in range(1,self.num_states+1)])
        ax.set_yticks(np.arange(self.num_states),[str(i) for i in range(1,self.num_states+1)])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_hmm_MEAN_trans_mat_num_states_' + \
                   str(self.num_states) + '_type_' + type + '_MC_' + str(self.num_MonteCarlo_runs) + '_shuffmethod_' + self.shuffle_method
        plt.savefig(os.path.join(self.paths['fig_main_sep_path'], fig_name + '.pdf'), format='pdf')
        print("Saving image")
        plt.show()

    def plot_MEAN_state_trans_prob_matrix_comb(self):
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        ax[0].imshow(self.ref_trans_wt, cmap='jet', vmin=0, vmax=1)
        for (j, i), label in np.ndenumerate(np.matrix.round(self.ref_trans_wt,decimals=2)):
            ax[0].text(i, j, label, ha='center', va='center',color='white')
        ax[0].set(adjustable='box', aspect='auto')
        ax[0].set_ylabel('Current state')
        ax[0].set_xlabel('Next state')
        ax[0].set_xticks(np.arange(self.num_states),[str(i) for i in range(1,self.num_states+1)])
        ax[0].set_yticks(np.arange(self.num_states),[str(i) for i in range(1,self.num_states+1)])
        divider2 = make_axes_locatable(ax[0])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cax2.axis('off')


        im = ax[1].imshow(self.ref_trans_hom, cmap='jet', vmin=0, vmax=1)
        for (j, i), label in np.ndenumerate(np.matrix.round(self.ref_trans_hom, decimals=2)):
            ax[1].text(i, j, label, ha='center', va='center', color='white')
        ax[1].set(adjustable='box', aspect='auto')
        ax[1].set_ylabel('Current state')
        ax[1].set_xlabel('Next state')
        ax[1].set_xticks(np.arange(self.num_states),[str(i) for i in range(1,self.num_states+1)])
        ax[1].set_yticks(np.arange(self.num_states),[str(i) for i in range(1,self.num_states+1)])
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_hmm_MEAN_trans_mat_COMBI_num_states_' + \
                   str(self.num_states) + '_MC_' + str(self.num_MonteCarlo_runs) + '_shuffmethod_' + self.shuffle_method
        plt.savefig(os.path.join(self.paths['fig_main_sep_path'], fig_name + '.pdf'), format='pdf')
        print("Saving image")
        plt.show()

    def plot_MSE_distribution(self):
        conf_high = np.mean(self.true_mse_conf) + 1.96 * np.std(self.true_mse_conf,ddof=1)/np.sqrt(self.num_MonteCarlo_runs)
        conf_low = np.mean(self.true_mse_conf) - 1.96 * np.std(self.true_mse_conf,ddof=1)/np.sqrt(self.num_MonteCarlo_runs)

        plt.figure(figsize=(5,3))
        plt.hist(self.mse_list, bins=20)
        plt.axvline(np.mean(self.true_mse_conf),color='r')
        plt.axvline(conf_high,color='r',ls=':')
        plt.axvline(conf_low,color='r',ls=':')
        plt.xlabel('MSE')
        plt.ylabel('Frequency')
        #plt.title("Distribution of MSEs")
        plt.tight_layout()
        if self.mse_range is not None:
            fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_MSE_distribution_' + str(
                self.num_states) + '_minchanges_' + str(self.min_changes) + '_MC_' + str(
                self.num_MonteCarlo_runs) + '_shuffmethod_' + self.shuffle_method + '_mse_range_' + str(self.mse_range[0]) + str(self.mse_range[-1])
        else:
            fig_name = self.PreProcessingName + '_' + self.MainProcessingName + '_MSE_distribution_' + str(
            self.num_states) + '_minchanges_' + str(self.min_changes) + '_MC_' + str(self.num_MonteCarlo_runs) + '_shuffmethod_' + self.shuffle_method
        plt.savefig(os.path.join(self.paths['fig_main_sep_path'], fig_name + '.pdf'), format='pdf')
        print("Saving image")
        plt.show()

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
        self.HRF_param = data['HRF_param']
        self.HRF = data['HRF']
        self.Y_list = data['Y_list']
        self.lambda_thres_list = data['lambda_thres_list']
        self.reconstruction_MSE_table = data['reconstruction_MSE_table']
        self.time = data['time']
        self.time_reduced = data['time_reduced']
        self.mus_target = np.array(data['mus_target'])

        self.main_list_to_dict_nparray()


if __name__ == "__main__":
    PreProcessingName = "PRE_32"
    MainProcessingName = "MAIN_12"

    main_sep_group = MainProcessingSepGroup(PreProcessingName,MainProcessingName)

    # Plot time courses and their deconvolution
    # main_sep_group.separate_group_analysis(num_states=4)
    #
    # main_sep_group.plot_mus(type="wt")
    # main_sep_group.plot_mus(type="hom")
    # main_sep_group.plot_state_trans_prob_matrix(type="wt")
    # main_sep_group.plot_state_trans_prob_matrix(type="hom")
    # if main_sep_group.num_regions==3:
    #     main_sep_group.plot_scatter_3D(type="wt")
    #     main_sep_group.plot_scatter_3D(type="hom")

    # Evaluate state trans prob matrices
    # true_mse_conf,mse_list = main_sep_group.convergence_conf_int(num_states=4,num_MonteCarlo_runs=500,min_changes=1,shuffle_method='rec',mse_range=[1,3])
    # main_sep_group.plot_REF_mus(type='wt')
    # main_sep_group.plot_REF_mus(type='hom')
    # main_sep_group.plot_MEAN_state_trans_prob_matrix(type='wt')
    # main_sep_group.plot_MEAN_state_trans_prob_matrix(type='hom')
    # main_sep_group.plot_MEAN_state_trans_prob_matrix_comb()
    # main_sep_group.plot_MSE_distribution()

    # Find number of unique functional networks
    # main_sep_group.find_num_states()

    # Consistency per group
    # mu_percent_list_wt,mean_mse_list_wt = main_sep_group.functional_network_consistency(num_states=4,num_MonteCarlo_runs=100,fraction=0.2,mouse_type='wt')
    # main_sep_group.plot_conf(num_states=4, mu_percent_list = mu_percent_list_wt, mouse_type='wt')
    # mu_percent_list_hom,mean_mse_list_hom = main_sep_group.functional_network_consistency(num_states=4,num_MonteCarlo_runs=100,fraction=0.2,mouse_type='hom')
    # main_sep_group.plot_conf(num_states=4, mu_percent_list = mu_percent_list_hom, mouse_type='hom')
