import numpy as np
import ssm as ssm
import seaborn as sns
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import matplotlib.patches as mpatches
import scipy.linalg as sl
import matplotlib.lines as mlines
import pandas as pd
import util

from mpl_toolkits.axes_grid1 import make_axes_locatable
from ssm.util import find_permutation
from matplotlib import cm
from tqdm import tqdm
from ssm.plots import gradient_cmap
from syn_fus import SynData
from matplotlib.legend_handler import HandlerTuple
from sklearn.linear_model import Lasso
from itertools import groupby


class SynDataSimulation(object):

    def __init__(self,SimulationName=None,plot_info=None):
        # Simulation
        self.ID = SimulationName

        # Paths
        self.paths = util.create_paths_dict(SimulationName=SimulationName)

        self.plot_info = plot_info
        self.save_plots = plot_info['save_plots']
        if plot_info is not None:
            self.plot_info['fig_path'] = self.paths['fig_sim_ID_path']
            self.plot_info['ID'] = SimulationName

        # Seed for transition probability matrix
        random_state = np.random.RandomState(seed=2)

        self.colors = util.create_color_palette()
        self.cmap = gradient_cmap(self.colors)

        # Several simulation scenarios are found below. Results can be found in the MSc Thesis.
        if SimulationName=="Bio-1":
            self.channels = 2
            self.T = 720
            self.T_res = 0.25
            self.fs = 4
            self.N = self.channels + 1  # always includes an empty group, made in syn_fus
            self.groups = [[], [0, 1], [0]]

            mat = np.array([[1000,20,20],
                            [20,200,20],
                            [20,20,200]])

            self.trans_prob_mat = mat / np.sum(mat, axis=1)[:, None]

            # HRF parameters
            self.HRF_param = {}
            self.HRF_param['p1'] = 4.0
            self.HRF_param['p2'] = 1.5
            self.HRF_param['p3'] = 0


        elif SimulationName=="Bio-2":
            self.channels = 2
            self.T = 720
            self.T_res = 0.25
            self.fs = 4
            self.N = self.channels + 1  # always includes an empty group, made in syn_fus
            self.groups = [[], [0, 1], [0]]

            mat = np.array([[1000,20,20],
                            [10,200,250],
                            [20,20,100]])

            self.trans_prob_mat = mat / np.sum(mat, axis=1)[:, None]

            # HRF parameters
            self.HRF_param = {}
            self.HRF_param['p1'] = 4.0
            self.HRF_param['p2'] = 1.5
            self.HRF_param['p3'] = 0


        elif SimulationName=="Bio-3":
            self.channels = 2
            self.T = 720
            self.T_res = 0.25
            self.fs = 4
            self.N = self.channels + 1  # always includes an empty group, made in syn_fus
            self.groups = [[], [0, 1], [0]]

            # # Create transition probability matrix
            mat = np.array([[1000,20,20],
                            [40,70,10],
                            [10,100,250]])
            self.trans_prob_mat = mat / np.sum(mat, axis=1)[:, None]

            # HRF parameters
            self.HRF_param = {}
            self.HRF_param['p1'] = 4.0
            self.HRF_param['p2'] = 1.5
            self.HRF_param['p3'] = 0


        elif SimulationName=="Bio-4":
            self.channels = 2
            self.T = 720
            self.T_res = 0.25
            self.fs = 4
            self.N = self.channels + 1  # always includes an empty group, made in syn_fus
            self.groups = [[], [0, 1], [0]]

            # # Create transition probability matrix
            mat = np.array([[1000,20,20],
                            [50,75,75],
                            [50,50,75]])
            self.trans_prob_mat = mat / np.sum(mat, axis=1)[:, None]

            # HRF parameters
            self.HRF_param = {}
            self.HRF_param['p1'] = 4.0
            self.HRF_param['p2'] = 1.5
            self.HRF_param['p3'] = 0


    def long_sim(self,T_times = 5, algorithm="LASSO",noise_std=0.5):
        """
        Create a simulation with T_times concatenated data, generated from the same state transition probability matrix.
        :param T_times: Number of concatenated data segments
        :param algorithm: "NNLS" or "LASSO"
        :param noise_std: Noise standard deviation.
        :return: No return.
        """
        self.algorithm = algorithm
        self.noise_std = noise_std
        self.T_times = T_times

        long_sim_param = [algorithm,noise_std,T_times]

        self.obs = None
        self.true_states = None

        for i in range(self.T_times):
            obs,true_states = self.deconv_data(algorithm=self.algorithm,noise_std=self.noise_std)
            if i==0:
                self.obs = obs
                self.true_states = true_states
            self.obs = np.concatenate((self.obs,obs),axis=0)
            self.true_states = np.concatenate((self.true_states,true_states))

        self.time_bins = np.size(self.obs,axis=0)
        self.time_array = np.linspace(0,self.time_bins*self.T_res-self.T_res, self.time_bins)
        self.true_transition_matrix = self.trans_prob_mat

        # Make HMM
        self.hmm = ssm.HMM(self.N, self.channels,observations="gaussian")
        self.hmm_lls = self.hmm.fit(self.obs, method="em", num_iters=50, init_method="kmeans")

        # Find permutation to compare true vs inferred state sequence
        self.hmm.permute(find_permutation(self.true_states, self.hmm.most_likely_states(self.obs)))
        self.inferred_states = self.hmm.most_likely_states(self.obs)

        # Plotting HMM results
        self.plot_data_2D_HMM(long_sim_param=long_sim_param)
        self.plot_trans_prob_matrices(long_sim_param=long_sim_param)

    def deconv_data(self,algorithm="LASSO",noise_std=0.3):
        """
        Deconvolve synthetic fUS data.

        :param algorithm: "NNLS" or "LASSO"
        :param noise_std: Noise standard deviation.
        :return: Reconstructed time courses, true states
        """
        self.alg = algorithm
        self.noise_std = noise_std

        # Create synthetic data
        d = SynData(self.channels, self.T, self.T_res, self.noise_std, self.fs, self.trans_prob_mat, HRF_param=self.HRF_param, groups=self.groups, algorithm=self.alg, plot_info = self.plot_info)

        # Determine if we need to find the regularization parameters
        if self.alg == "NNLS":
            d.reconstruct()
        elif self.alg == "LASSO":
            self.lambdas = self.compute_lambda(d, rel_MSE_threshold=0.01)
            d.reconstruct(lambda_=self.lambdas)

        # Parameters
        obs = d.x_reconstructed_full[d.T_ind_HRF - 1:, :]
        true_obs = d.imp_seqs[:, :d.T_indices - (d.T_ind_HRF - 1)].T
        true_states = d.true_states[:d.T_indices-(d.T_ind_HRF - 1)]

        return obs,true_states

    def single_run(self,algorithm="LASSO", noise_std=0.3, seed=False):
        """

        :param algorithm: "NNLS" or "LASSO"
        :param noise_std: Noise standard deviation.
        :param seed: Set seed=True for reproducible results
        :return:
        """
        self.alg = algorithm
        self.noise_std = noise_std

        # Create synthetic data
        d = SynData(self.channels, self.T, self.T_res, self.noise_std, self.fs, self.trans_prob_mat, HRF_param=self.HRF_param, groups=self.groups, algorithm=self.alg, plot_info = self.plot_info, seed=seed)

        # Determine if we need to find the regularization parameters and reconstruct
        if self.alg == "NNLS":
            d.reconstruct()
        elif self.alg == "LASSO":
            self.lambdas = self.compute_lambda(d, rel_MSE_threshold=0.01)
            d.reconstruct(lambda_=self.lambdas)

        # Parameters
        self.obs = d.x_reconstructed_full[d.T_ind_HRF - 1:, :]
        self.true_obs = d.imp_seqs[:, :d.T_indices - (d.T_ind_HRF - 1)].T
        self.true_states = d.true_states[:d.T_indices-(d.T_ind_HRF - 1)]
        self.time_bins = d.T_indices - (d.T_ind_HRF - 1)
        self.time_array = np.linspace(0,d.T-(d.T_ind_HRF-1)/d.fs-1/d.fs, self.time_bins)
        self.true_transition_matrix = d.trans_prob_mat

        # Make HMM
        self.hmm = ssm.HMM(d.N, d.channels, observations="gaussian")
        self.hmm_lls = self.hmm.fit(self.obs, method="em", num_iters=50, init_method="kmeans")

        # Find permutation to compare true vs inferred state sequence
        self.hmm.permute(find_permutation(self.true_states, self.hmm.most_likely_states(self.obs)))
        self.inferred_states = self.hmm.most_likely_states(self.obs)

        # Synthetic data plotting
        d.plot_HRF_channels()
        d.plot_fus_seq()
        d.plot_noisy_fus_seq()
        d.plot_imp_seq()
        d.plot_x_reconstructed()
        d.paper_fig()

        if self.alg == "LASSO":
            self.plot_lambda_mse()

        # Plotting HMM results
        self.plot_z_inf_true()
        self.plot_data_2D_HMM()
        # self.plot_data_3D_HMM()
        self.plot_trans_prob_matrices()
        self.plot_true_A()
        self.plot_inf_A()
        #self.plot_z_inf_obs()
        # self.plot_z_inf_clean()
        self.plot_A_mus()
        self.plot_state_life_time()
        self.plot_inter_state_time()

        return


    def noise_analysis_single(self,algorithm="LASSO",num_runs=10,num_stds=5):
        """
        Analyze the influence of noise on inferred matrix A.
        :param algorithm: "LASSO"
        :param num_runs: Number of runs for statistical analysis
        :param num_stds: Number of different noise standard deviations in the range 0.1 to 0.5.
        :return:
        """
        self.alg = algorithm
        self.num_runs = num_runs
        self.num_stds = num_stds
        self.noise_std_range = np.linspace(0.1, 0.5, self.num_stds)

        self.trans_prob_MSE = np.empty((self.num_stds,self.num_runs))

        # Create synthetic data
        for i in tqdm(range(self.num_stds),position=0,desc="Noise std"):
            for j in range(self.num_runs):

                self.noise_std = self.noise_std_range[i]
                d = SynData(self.channels, self.T, self.T_res, self.noise_std, self.fs, self.trans_prob_mat, HRF_param=self.HRF_param,groups=self.groups,algorithm=self.alg)
                if i==0 and j==0: # First time, find lambdas, are independent of noise
                    self.lambdas = self.compute_lambda(d, rel_MSE_threshold=0.01)
                d.reconstruct(lambda_=self.lambdas)
                #d.reconstruct(lambda_=0.001)

                # Make an HMM
                self.obs = d.x_reconstructed_full[d.T_ind_HRF - 1:, :]
                self.true_obs = d.imp_seqs[:, :d.T_indices - (d.T_ind_HRF - 1)].T
                self.true_states = d.true_states[:d.T_indices - (d.T_ind_HRF - 1)]
                self.time_bins = d.T_indices - (d.T_ind_HRF - 1)
                self.true_transition_matrix = d.trans_prob_mat

                self.hmm = ssm.HMM(d.N, d.channels,observations="gaussian")
                self.hmm_lls = self.hmm.fit(self.obs, method="em", num_iters=50, init_method="kmeans")

                # Find permutation to compare true vs inferred state sequence
                self.hmm.permute(find_permutation(self.true_states, self.hmm.most_likely_states(self.obs)))
                self.inferred_states = self.hmm.most_likely_states(self.obs)

                self.trans_prob_MSE[i,j]=np.sum((self.true_transition_matrix -self.hmm.transitions.transition_matrix)**2 )/(self.hmm.K**2)

                # if (i==3):
                #     self.plot_data_2D_HMM() # debugging purposes
        self.plot_mse()

        return

    def noise_analysis_comb(self,num_runs=10,num_stds=5):

        self.num_runs=num_runs
        self.num_stds=num_stds
        self.noise_std_range = np.linspace(0.1, 0.5, self.num_stds)

        self.trans_prob_MSE = np.empty((self.num_stds,self.num_runs))
        self.trans_prob_MSE_comb = np.empty((self.num_stds,self.num_runs,2))

        # Create synthetic data
        for a in range(2):
            if a==0:
                alg="NNLS"
            else:
                alg="LASSO"

            for i in tqdm(range(self.num_stds),desc="Noise std"):
                for j in range(self.num_runs):
                    d = SynData(self.channels, self.T, self.T_res, self.noise_std_range[i], self.fs, self.trans_prob_mat, HRF_param=self.HRF_param,groups=self.groups, algorithm=alg)

                    if a == 0:
                        d.reconstruct(lambda_=0.001)
                    else:
                        if j == 0 and i == 0 and a == 1:  # First time, find lambdas
                            self.lambdas = self.compute_lambda(d, rel_MSE_threshold=0.01)
                        d.reconstruct(lambda_=self.lambdas)

                    # Make an HMM
                    self.obs = d.x_reconstructed_full[d.T_ind_HRF - 1:, :]
                    self.true_obs = d.imp_seqs[:, :d.T_indices - (d.T_ind_HRF - 1)].T
                    self.true_states = d.true_states[:d.T_indices - (d.T_ind_HRF - 1)]
                    self.time_bins = d.T_indices - (d.T_ind_HRF - 1)
                    self.true_transition_matrix = d.trans_prob_mat

                    self.hmm = ssm.HMM(d.N, d.channels,observations="gaussian")  # , transitions="recurrent_only") #,transition_kwargs=dict(kappa=1000))
                    self.hmm_lls = self.hmm.fit(self.obs, method="em", num_iters=50, init_method="kmeans")

                    # Find permutation to compare true vs inferred state sequence
                    self.hmm.permute(find_permutation(self.true_states, self.hmm.most_likely_states(self.obs)))
                    self.inferred_states = self.hmm.most_likely_states(self.obs)
                    self.trans_prob_MSE[i,j]=np.sum((self.true_transition_matrix -self.hmm.transitions.transition_matrix)**2 )/(self.hmm.K**2)
            self.trans_prob_MSE_comb[:,:,a] = self.trans_prob_MSE

        self.plot_mse_combined()

        return

    def convergence(self,alg='LASSO',length=360,times=16,runs=5):
        """
        Analyse converge to show that dynamics do not depend on the impulse sequence for a certain length
        :param alg: "NNLS" or "LASSO"
        :param length: Duration of each data block
        :param times: Number of data blocks concatenated
        :param runs: Number of runs for computing confidence intervals
        :return:
        """
        self.num_runs=runs
        self.seq_time = np.arange(length,length*(times+1),length) # not the exact times...
        self.stds = np.array([0.1,0.3,0.5])
        self.convergence_result = {}
        for std in tqdm(self.stds):
            self.convergence_result[str(std)] = np.zeros((2,times))
            MSE_res = np.zeros((runs,times))
            for run in range(runs):
                self.obs = None
                self.true_states = None
                for i in range(times):
                    obs, true_states = self.deconv_data(algorithm=alg,noise_std=std)
                    if i == 0:
                        self.obs = obs
                        self.true_states = true_states
                    self.obs = np.concatenate((self.obs, obs), axis=0)
                    self.true_states = np.concatenate((self.true_states, true_states))

                    # Make HMM
                    hmm = ssm.HMM(self.N, self.channels,observations="gaussian")
                    hmm_lls = hmm.fit(self.obs, method="em", num_iters=50, init_method="kmeans")

                    # Find permutation to compare true vs inferred state sequence
                    hmm.permute(find_permutation(self.true_states, hmm.most_likely_states(self.obs)))
                    MSE_res[run,i] = np.sum((hmm.transitions.transition_matrix - self.trans_prob_mat)**2)/(hmm.K**2)
            self.convergence_result[str(std)][0,:] = np.mean(MSE_res,axis=0)
            self.convergence_result[str(std)][1,:] = np.std(MSE_res,axis=0,ddof=1)

    def compute_lambda(self,data,rel_MSE_threshold=0.01,begin=-4,end=0,num=50):

        length = data.T_indices
        HRF_length = data.T_ind_HRF

        # Create convolution matrix
        H = sl.convolution_matrix(data.HRF[0, :], length + HRF_length - 1) #HRF specified by HRF_param
        H = H[HRF_length - 1:length + HRF_length - 1, :]  #In reality we only have a part of H
        self.H_ = H

        dim = data.channels
        self.rel_MSE_threshold = rel_MSE_threshold
        self.lambda_range = np.logspace(begin, end, num, endpoint=True)
        self.reconstruction_MSE_table = np.empty((dim, num))
        lambdas = np.zeros(dim)
        text="Finding lambdas"
        for i in range(data.channels):
            y = data.noisy_fus_seq[i,:].T
            l = Lasso(fit_intercept=False, positive=True, tol=1e-3, max_iter=10000, selection='cyclic', warm_start=True)  # Fit_intercept is false to avoid mean-subtraction
            for lambda_, j in tqdm(zip(self.lambda_range, range(num)),total=num,position=0,desc=text):
                l.set_params(alpha=lambda_)
                l.fit(H,y)
                x_solution = l.coef_[:length]
                self.reconstruction_MSE_table[i,j] = sum((H[:,:length] @ x_solution - y) ** 2)/len(x_solution)
        rel_MSE = (self.reconstruction_MSE_table-self.reconstruction_MSE_table[:,0][:,None])/(self.reconstruction_MSE_table[:,-1]-self.reconstruction_MSE_table[:,0])[:,None]
        for j in range(dim):
            loc = np.where(rel_MSE[j,:]<self.rel_MSE_threshold)[0][-1]
            lambdas[j] = self.lambda_range[loc]

        return lambdas

    # Plotting
    def plot_convergence(self):
        plt.figure(figsize=(12, 7))
        labels = []
        for std, i in zip(self.stds, range(len(self.stds))):
            plt.plot(self.seq_time, self.convergence_result[str(std)][0, :], marker=".",markersize=15,label="$\sigma$=" + str(round(self.stds[i], 1)))
            conf_low = self.convergence_result[str(std)][0, :] - 1.96 * self.convergence_result[str(std)][1, :] / np.sqrt(
                self.num_runs)
            conf_high = self.convergence_result[str(std)][0, :] + 1.96 * self.convergence_result[str(std)][1, :] / np.sqrt(
                self.num_runs)
            plt.fill_between(self.seq_time, conf_high, conf_low, alpha=.3)
            labels.append("$\sigma$=" + str(round(self.stds[i], 1)))

        colors = plt.get_cmap("tab10")
        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=colors(i), label=labels[i]) for i in range(len(labels))]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title("Convergence of inferred state transition probability matrix")
        plt.xlabel("Time course duration [s]")
        plt.ylabel("MSE")
        plt.xlim([self.seq_time[0], self.seq_time[-1]])
        plt.grid(True)
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.ID + '_convergence'
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'], fig_name + '.pdf'), format='pdf')
            np.save(os.path.join(self.paths['fig_sim_ID_path'], fig_name), self.convergence_result)
        plt.show()

    def plot_A_mus(self):
        fig, ax = plt.subplots(figsize=(5,5))
        labels = [str(i) for i in range(1,self.hmm.K+1)]
        im = ax.imshow(self.true_transition_matrix, cmap='jet', vmin=0, vmax=1)
        ax.set(adjustable='box', aspect='auto')
        fig.suptitle('True $\mathbf{A}$')
        ax.set_ylabel('Current state')
        ax.set_xlabel('Next state')
        ax.set_xticks(np.arange(self.hmm.K),labels)
        ax.set_yticks(np.arange(self.hmm.K),labels)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        for (j, i), label in np.ndenumerate(np.matrix.round(self.true_transition_matrix,decimals=2)):
            ax.text(i, j, label, ha='center', va='center',color='white')

        plt.tight_layout()
        if self.save_plots:
            fig_name = self.ID + '_TRUE_A'
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'], fig_name + '.pdf'), format='pdf')
        plt.show()

        fig, ax = plt.subplots(figsize=(5,7))
        #fig.suptitle('True functional networks')
        values = 0.9 * (self.hmm.observations.mus.T > 1e-4)
        im = ax.imshow(values, cmap='jet', vmin=0, vmax=1)
        colors = [im.cmap(im.norm(value)) for value in np.unique(values.ravel())]
        labels = ["Not active", "Active"]
        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(0, 1.02), loc='lower left', borderaxespad=0.)

        #plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set(adjustable='box', aspect='auto')
        ax.set_ylabel('$\mathbf{\mu}_z$')
        ax.set_xlabel('State number')
        ax.set_xticks(np.arange(self.hmm.K), [str(i) for i in range(1, self.hmm.K + 1)])
        ax.set_yticks(np.arange(self.channels), ['$m=$' + str(i+1) for i in range(self.channels)])
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.ID + '_TRUE_MUS'
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'],fig_name + '.pdf'), format='pdf')
        plt.show()



    def plot_lambda_mse(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        leg_list = []
        names = []
        for lam, i in zip(self.lambdas, range(len(self.lambdas))):
            p1, = ax.plot(self.lambda_range, self.reconstruction_MSE_table[i, :], color=colors[i])
            p2 = ax.axvline(x=lam, ls=':', color=colors[i])
            leg_list.append((p1, p2))
            names.append("$m=" + str(i) + "$")
        ax.set_title(
            "RE between $\mathbf{f}_m$ and $\hat{\mathbf{f}}_m$ for different $\lambda$'s")
        ax.set_xlabel("$\lambda$")
        ax.set_xscale("log")
        ax.set_ylabel("$RE(\lambda)$")
        # plt.legend(["$m=" + str(i) + "$" for i in range(np.size(self.reconstruction_MSE_table,axis=0))])
        l = ax.legend(leg_list, names, handler_map={tuple: HandlerTuple(ndivide=None)})
        ax.grid(True)
        ax.set_xlim(self.lambda_range[0], self.lambda_range[-1])
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.ID + '_lambdaMSEcomb_rel_MSE_threshold' + str(self.rel_MSE_threshold) + "_noisestd_" +str(self.noise_std)+ '_'+ self.alg
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'], fig_name + '.pdf'), format='pdf')
        plt.show()

    def plot_mse(self):
        plt.figure(figsize=(8, 6))
        mean = np.mean(self.trans_prob_MSE, axis=1)
        std = np.std(self.trans_prob_MSE, axis=1, ddof=1)  # unbiased estimator
        conf_low = mean - 1.96 * std / np.sqrt(self.num_runs)
        conf_high = mean + 1.96 * std / np.sqrt(self.num_runs)

        plt.plot(self.noise_std_range, mean, marker=".",label=self.alg)
        if self.num_runs>1:
            plt.fill_between(self.noise_std_range, conf_high, conf_low, alpha=.3)
        plt.title("MSE between true and inferred $\mathbf{A}$ matrix under different $\sigma$'s")
        plt.xticks(self.noise_std_range, [str(round(i,1)) for i in self.noise_std_range])
        plt.xlabel("Noise standard deviation $\sigma$")
        plt.ylabel("MSE")
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.xlim(self.noise_std_range[0], self.noise_std_range[-1])
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.ID + '_MSEcomb_rel_MSE_threshold' + str(self.rel_MSE_threshold) + "_" + self.alg
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'], fig_name + '.pdf'), format='pdf')
        plt.show()

    def plot_mse_combined(self):
        legend_labels = ["NNLS","LASSO"]
        plt.figure(figsize=(8, 6))
        for i in range(2):
            mean = np.mean(self.trans_prob_MSE_comb[:,:,i], axis=1)
            std = np.std(self.trans_prob_MSE_comb[:,:,i], axis=1, ddof=1)  # unbiased estimator
            conf_low = mean - 1.96 * std / np.sqrt(self.num_runs)
            conf_high = mean + 1.96 * std / np.sqrt(self.num_runs)

            plt.plot(self.noise_std_range, mean, marker=".",label=legend_labels[i])
            if self.num_runs > 1:
                plt.fill_between(self.noise_std_range, conf_high, conf_low, alpha=.3)
        plt.title("MSE between true and inferred $\mathbf{A}$ matrix under different $\sigma$'s")
        plt.xticks(self.noise_std_range, [str(round(i, 1)) for i in self.noise_std_range])
        plt.xlabel("Noise standard deviation $\sigma$")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.legend()
        plt.xlim(self.noise_std_range[0], self.noise_std_range[-1])
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.ID + '_MSEcomb_rel_MSE_threshold' + str(self.rel_MSE_threshold)
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'],fig_name + '.pdf'), format='pdf')
        plt.show()


    # Plot the true and inferred discrete states
    def plot_z_inf_true(self):
        fig, ax = plt.subplots(2, 1, figsize=(15, 5),dpi=500)
        fig.suptitle("True and inferred state sequence $\mathbf{z}$")
        ax[0].imshow(self.true_states[None,:], aspect="auto", cmap=self.cmap, vmin=0, vmax=len(self.colors)-1,interpolation='none')
        ax[0].set_xlim(0, self.time_bins)
        ax[0].set_ylabel("$\mathbf{z}_{\\mathrm{true}}$")
        ax[0].set_yticks([])
        ax[0].set_xticks([])

        ax[1].imshow(self.inferred_states[None,:], aspect="auto", cmap=self.cmap, vmin=0, vmax=len(self.colors)-1,extent=[self.time_array[0],self.time_array[-1],-1,1],interpolation='none')
        #plt.xlim(0, self.time_bins)
        ax[1].set_ylabel("$\mathbf{z}_{\\mathrm{inferred}}$")
        ax[1].set_yticks([])
        ax[1].set_xlabel("Time [s]")

        patches = [mpatches.Patch(color=self.colors[i], label="State " + str(i+1)) for i in range(self.hmm.K)]
        # put those patched as legend-handles into the legend
        ax[0].legend(handles=patches, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.,prop={'size': 20})

        plt.tight_layout()
        if self.save_plots:
            fig_name = self.ID + '_Zseq_sigma' + str(self.noise_std) + '_' + self.alg
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'],fig_name + '.pdf'), format='pdf')
        plt.show()

    # Plotting
    def plot_EM_conv(self):
        plt.figure()
        plt.plot(self.hmm_lls, label="EM")
        #plt.plot([0, N_iters], true_ll * np.ones(2), ':k', label="True")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log Probability")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    def plot_z_inf_obs(self):
        lim = 1.05 * abs(self.obs).max()
        fig = plt.figure(figsize=(12, 6))
        plt.imshow(self.inferred_states[None,:],
                   aspect="auto",
                   cmap=self.cmap,
                   vmin=0,
                   vmax=len(self.colors)-1,
                   extent=(0, self.time_bins, -lim, (self.hmm.D)*lim),interpolation='none')

        for i in range(self.hmm.D):
            plt.plot(self.obs[:,i] + lim * (self.hmm.D-1-i), '-k')

        plt.xlim(0, self.time_bins)
        plt.xlabel("Time [s]")
        plt.yticks(lim * np.arange(self.hmm.D), ["$x_{}$".format(i) for i in range(self.hmm.D-1,-1,-1)])
        plt.title("Inferred state sequence from reconstructed neural population data")
        plt.tight_layout()
        plt.show()

    def plot_z_inf_clean(self):
        lim = 1.05 * abs(self.true_obs).max()
        plt.figure(figsize=(12, 6))
        plt.imshow(self.inferred_states[None,:],
                   aspect="auto",
                   cmap=self.cmap,
                   vmin=0,
                   vmax=len(self.colors)-1,
                   extent=(0, self.time_array[-1], -lim, (self.hmm.D)*lim))

        for i in range(self.hmm.D):
            plt.plot(self.true_obs[:,i] + lim * (self.hmm.D-1-i), '-k')

        plt.xlim(0, self.time_array[-1])
        plt.xlabel("Time [s]")
        plt.yticks(lim * np.arange(self.hmm.D), ["$x_{}$".format(i) for i in range(self.hmm.D-1,-1,-1)])
        plt.title("Inferred state sequence vs true impulse sequence")
        plt.tight_layout()
        plt.show()

    def plot_data_2D_HMM(self,long_sim_param=None):
        if self.alg=='LASSO':
            alg='NNLASSO'
        else:
            alg=self.alg

        fig, ax = plt.subplots(1,2, figsize=(15, 7.5))
        plt.suptitle("Classification of $\mathbf{y}_n$ under noise $\sigma = $" + str(round(self.noise_std,1)) + ' using '+ alg)
        ax[0].set_title("True classification")
        color = [self.colors[i] for i in self.true_states]
        ax[0].scatter(self.obs[:, 0], self.obs[:,1], c=color, marker='x', s=80)
        ax[0].grid(visible=True)
        ax[0].set_xlabel("$y_{n,1}$")
        ax[0].set_ylabel("$y_{n,2}$")

        ax[1].set_title("Inferred classification")
        color = [self.colors[i] for i in self.inferred_states]
        im=ax[1].scatter(self.obs[:, 0], self.obs[:, 1], c=color, marker='x', s=80)
        ax[1].grid(visible=True)
        ax[1].set_xlabel("$y_{n,1}$")
        ax[1].set_ylabel("$y_{n,2}$")

        leg_list = []
        for i in range(self.hmm.K):
            leg_list.append(mlines.Line2D([], [], color=self.colors[i], marker='x', linestyle='None',markersize=10, label="State " + str(i+1)))
        plt.legend(handles=leg_list, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.tight_layout()
        if self.save_plots:
            if long_sim_param is not None:
                fig_name = self.ID + '_2D_long_sim_' + long_sim_param[0] + '_' + str(long_sim_param[1]) + '_' + str(long_sim_param[2])
            else:
                fig_name = self.ID + '_2D_sigma' + str(self.noise_std) + '_' + alg
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'],fig_name + '.pdf'), format='pdf')
        plt.show()


    def plot_data_3D_HMM(self):
        yAmplitudes = self.obs[:,0] # your data here
        xAmplitudes = self.obs[:,1] # your other data here
        x = np.array(xAmplitudes)  # turn x,y data into numpy arrays
        y = np.array(yAmplitudes)

        fig = plt.figure()  # create a canvas, tell matplotlib it's 3d
        ax = fig.add_subplot(111, projection='3d')
        # make histogram stuff - set bins - I choose 20x20 because I have a lot of data
        hist, xedges, yedges = np.histogram2d(x, y, bins=(20, 20))
        xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:]) - abs(xedges[1]-xedges[0])
        xpos = xpos.flatten() / 2.
        ypos = ypos.flatten() / 2.
        zpos = np.zeros_like(xpos)

        clip=20
        dx = xedges[1] - xedges[0]
        dy = yedges[1] - yedges[0]
        dz = hist.flatten()
        dz[dz > clip] = clip
        cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
        max_height = np.max(dz)  # get range of colorbars so we can normalize
        min_height = np.min(dz)
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k - min_height) / max_height) for k in dz]
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
        ax.set_zlim3d(0,clip)
        plt.title("Reconstructed activity")
        plt.xlabel("$y_0$")
        plt.ylabel("$y_1$")
        plt.show()

    def plot_trans_prob_matrices(self,long_sim_param=None):

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #fig.suptitle('Temporal dynamics captured by state transition probability matrix', fontsize=18)
        labels = [str(i) for i in range(1,self.hmm.K+1)]

        im = ax[0].imshow(self.true_transition_matrix, cmap='jet', vmin=0, vmax=1)
        for (j, i), label in np.ndenumerate(np.matrix.round(self.true_transition_matrix,decimals=2)):
            ax[0].text(i, j, label, ha='center', va='center',color='white')
        ax[0].set(adjustable='box', aspect='auto')
        ax[0].set_title('$\mathbf{A}_{\\mathrm{true}}$')
        ax[0].set_ylabel('Current state')
        ax[0].set_xlabel('Next state')
        ax[0].set_xticks(np.arange(self.hmm.K),labels)
        ax[0].set_yticks(np.arange(self.hmm.K),labels)
        # divider = make_axes_locatable(ax[0])
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # plt.colorbar(im, cax=cax)

        im = ax[1].imshow(self.hmm.transitions.transition_matrix, cmap='jet', vmin=0, vmax=1)
        for (j, i), label in np.ndenumerate(np.matrix.round(self.hmm.transitions.transition_matrix,decimals=2)):
            ax[1].text(i, j, label, ha='center', va='center',color='white')
        ax[1].set(adjustable='box', aspect='auto')
        ax[1].set_title('$\mathbf{A}_{\\mathrm{inferred}}$')
        ax[1].set_ylabel('Current state')
        ax[1].set_xlabel('Next state')
        ax[1].set_xticks(np.arange(self.hmm.K), labels)
        ax[1].set_yticks(np.arange(self.hmm.K), labels)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        if self.save_plots:
            if long_sim_param is not None:
                fig_name = self.ID + '_A_long_sim_' + long_sim_param[0] + '_' + str(long_sim_param[1]) + '_' + str(long_sim_param[2])
            else:
                fig_name = self.ID + '_A_sigma' + str(self.noise_std) + '_'+ self.alg

            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'], fig_name + '.pdf'), format='pdf')
        plt.show()

    def plot_true_A(self):
        fig, ax = plt.subplots(1,1,figsize=(4, 4))
        labels = [str(i) for i in range(1,self.hmm.K+1)]

        im=ax.imshow(self.true_transition_matrix, cmap='jet', vmin=0, vmax=1)
        for (j, i), label in np.ndenumerate(np.matrix.round(self.true_transition_matrix,decimals=2)):
            ax.text(i, j, label, ha='center', va='center',color='white')
        ax.set(adjustable='box', aspect='auto')
        ax.set_ylabel('Current state')
        ax.set_xlabel('Next state')
        ax.set_xticks(np.arange(self.hmm.K),labels)
        ax.set_yticks(np.arange(self.hmm.K),labels)
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)
        #plt.colorbar(im, cax=cax)
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.ID + '_A_true'
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'], fig_name + '.pdf'), format='pdf')
        plt.show()

    def plot_inf_A(self,long_sim_param=None):
        fig, ax = plt.subplots(1,1,figsize=(5, 4))
        labels = [str(i) for i in range(1,self.hmm.K+1)]
        im = ax.imshow(self.hmm.transitions.transition_matrix, cmap='jet', vmin=0, vmax=1)
        for (j, i), label in np.ndenumerate(np.matrix.round(self.hmm.transitions.transition_matrix,decimals=2)):
            ax.text(i, j, label, ha='center', va='center',color='white')
        ax.set(adjustable='box', aspect='auto')
        ax.set_ylabel('Current state')
        ax.set_xlabel('Next state')
        ax.set_xticks(np.arange(self.hmm.K), labels)
        ax.set_yticks(np.arange(self.hmm.K), labels)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        if self.save_plots:
            if long_sim_param is not None:
                fig_name = self.ID + '_A_inf_long_sim_' + long_sim_param[0] + '_' + str(long_sim_param[1]) + '_' + str(
                    long_sim_param[2])
            else:
                fig_name = self.ID + '_A_inf_sigma' + str(self.noise_std) + '_' + self.alg
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'], fig_name + '.pdf'), format='pdf')
        plt.show()

    def plot_state_life_time(self,long_sim_param=None):
        types = []
        states = []
        life_times = []
        # Compute true life time
        for state in range(self.N):
            life_time_groups = [len(list(g)) for i, g in groupby(self.true_states) if i == state]
            life_times.extend(life_time_groups)
            states.extend([str(state) for _ in range(len(life_time_groups))])
            types.extend(['True' for _ in range(len(life_time_groups))])

        # Compute inferred life time
        for state in range(self.N):
            life_time_groups = [len(list(g)) for i, g in groupby(self.inferred_states) if i == state]
            life_times.extend(life_time_groups)
            states.extend([str(state) for _ in range(len(life_time_groups))])
            types.extend(['Inferred' for _ in range(len(life_time_groups))])

        d = {'Type': types, 'State':states, 'Life time':life_times}
        df = pd.DataFrame(data=d)

        fig,ax = plt.subplots(1,self.N,figsize=(15, 7.5))
        for i in range(self.N):
            sns.violinplot(x="State", y="Life time", hue="Type", data=df[df['State']==str(i)], palette="muted",ax=ax[i])
            ax[i].get_legend().remove()
            ax[i].set(xlabel=None, ylabel=None)
            ax[i].grid(True)
            ax[i].tick_params(axis='both', which='major', labelsize=15)

        fig.supxlabel("State")
        fig.supylabel("Life time")
        fig.suptitle("True and inferred state lifetime")

        plt.legend(fontsize=15,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        if self.save_plots:
            if long_sim_param is not None:
                fig_name = self.ID + '_life_times_' + long_sim_param[0] + '_' + str(long_sim_param[1]) + '_' + str(long_sim_param[2])
            else:
                fig_name = self.ID + '_life_times_' + str(self.noise_std) + '_'+ self.alg
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'],fig_name + '.pdf'), format='pdf')
        plt.show()

    def plot_inter_state_time(self,long_sim_param=None):
        types = []
        states = []
        inter_state_times = []
        # Compute true inter state time
        for state in range(self.N):
            inter_state_time_group = np.diff(np.where(self.true_states==state))
            inter_state_time_group_red = inter_state_time_group[inter_state_time_group>1]
            inter_state_times.extend( inter_state_time_group_red )
            states.extend([str(state) for _ in range(len(inter_state_time_group_red))])
            types.extend(['True' for _ in range(len(inter_state_time_group_red))])

        # Compute inferred inter state time
        for state in range(self.N):
            inter_state_time_group = np.diff(np.where(self.inferred_states==state))
            inter_state_time_group_red = inter_state_time_group[inter_state_time_group>1]
            inter_state_times.extend(inter_state_time_group_red)
            states.extend([str(state) for _ in range(len(inter_state_time_group_red))])
            types.extend(['Inferred' for _ in range(len(inter_state_time_group_red))])

        d = {'Type': types, 'State':states, 'Inter state time':inter_state_times}
        df = pd.DataFrame(data=d)

        fig,ax = plt.subplots(1,self.N,figsize=(15, 7.5))
        for i in range(self.N):
            sns.violinplot(x="State", y="Inter state time", hue="Type", data=df[df['State']==str(i)], palette="muted",ax=ax[i])
            ax[i].get_legend().remove()
            ax[i].set(xlabel=None, ylabel=None)
            ax[i].grid(True)
            ax[i].tick_params(axis='both', which='major', labelsize=15)

        fig.supxlabel("State")
        fig.supylabel("Inter state time")
        fig.suptitle("True and inferred inter state time")

        plt.legend(fontsize=15,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        if self.save_plots:
            if long_sim_param is not None:
                fig_name = self.ID + '_inter_state_times_' + long_sim_param[0] + '_' + str(long_sim_param[1]) + '_' + str(long_sim_param[2])
            else:
                fig_name = self.ID + '_inter_state_times_' + str(self.noise_std) + '_'+ self.alg
            plt.savefig(os.path.join(self.paths['fig_sim_ID_path'],fig_name + '.pdf'), format='pdf')
        plt.show()

if __name__=="__main__":
    for i in [1]: #range(4):
        SimulationName = "Bio-" + str(i+1)

        # Plots
        plot_info =  {}
        plot_info['save_plots'] = True
        plot_info['fig_path'] = None
        plot_info['fig_ID'] = SimulationName

        a = SynDataSimulation(SimulationName=SimulationName,plot_info=plot_info)

        # Show convergence of inferred state transition probability matrices
        # a.convergence(alg='LASSO',times=9,runs=5,length=720)
        # a.plot_convergence()

        # Simulations.
        #a.single_run(algorithm="NNLS",noise_std=0.1, seed=True)
        #a.single_run(algorithm="NNLS", noise_std=0.3, seed=True)
        # a.single_run(algorithm="NNLS", noise_std=0.5, seed=True)
        # a.single_run(algorithm="LASSO", noise_std=0.1, seed=True)
        a.single_run(algorithm="LASSO", noise_std=0.3, seed=True)
        # a.single_run(algorithm="LASSO", noise_std=0.5, seed=True)

        # Simulate inference on the concatenation of 8 mouse recordings of 12 minutes
        # a.long_sim(T_times=8,algorithm="LASSO",noise_std=0.1)
        # a.long_sim(T_times=8,algorithm="LASSO",noise_std=0.3)
        # a.long_sim(T_times=8,algorithm="LASSO",noise_std=0.5)


        #a.noise_analysis_single(algorithm="LASSO",num_runs=3,num_stds=5)
        #a.noise_analysis_comb(num_runs=3,num_stds=5)
