import numpy as np
import scipy.linalg as sl
import random
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from scipy.optimize import nnls
from sklearn.linear_model import Lasso
from scipy.special import gamma

class SynData(object):


    def __init__(self, channels, T, T_res, noise_std, fs, trans_prob_mat, HRF_param=None, groups=None, algorithm="NNLS",neural_std=0,plot_info = None,seed=False):
        """
        Function: Generate synthetic fUS data.
        Author: Ruben Wijnands

        :param channels: number of channels providing time-series data.
        :param T: total time over which the data is defined [s]
        :param T_res: Time resolution [s]
        :param noise_std: noise standard deviation
        :param fs: fUS sampling frequency [Hz]
        :param trans_prob_mat: transition probability matrix for producing Markovian data
        :param HRF_param: Specific HRF parameters that create a custom HRF
        :param algorithm: algorithm for reconstruction of underlying activity. Possibilities: "NNLS", l2-norm constrained "NNLS_const", "LASSO" with positive coefficients
        :param neural_std: Add noise on the true underlying activity. Default: binary impulse sequence without noise
        :param seed: Seed for similar time courses but different noise realizations.
        """

        self.channels,self.T,self. T_res,self.noise_std,self.fs,self.trans_prob_mat,self.HRF_param,self.groups,self.algorithm,self.neural_std, self.seed = \
            channels,T,T_res,noise_std,fs,trans_prob_mat,HRF_param,groups,algorithm,neural_std,seed

        # Select a short HRF
        self.T_HRF = 8  # seconds

        self.T_ind_HRF = int(self.T_HRF / self.T_res) + 1
        self.T_indices = int(self.T / self.T_res)

        self.time = np.linspace(0, self.T - self.T_res, self.T_indices)
        self.time_reduced = self.time[:len(self.time) - (self.T_ind_HRF - 1)]
        self.time_HRF = self.time[:self.T_ind_HRF]

        self.N = len(self.groups)  # Number of true groups, i.e. functionally connected networks
        self.HRF = self.gen_HRF_channels()
        self.imp_seqs = self.gen_impulse_series()
        self.fus_seq = self.gen_syn_fus()
        self.noisy_fus_seq = self.noisy_syn_fus()

        if plot_info is None:
            self.save_plots = False
            self.fig_path = None
        else:
            self.save_plots = plot_info['save_plots']
            self.fig_path = plot_info['fig_path']
            self.fig_ID = plot_info['fig_ID']


    def gen_HRF_channels(self):
        """
        Method to generate HRFs.

        If self.HRF_param is specified, one single HRF according to specified parameters is constructed.
        HRFs are always scaled to unit amplitude and in case of multiple HRFs shifted to the same time-to-peak.
        """

        if self.HRF_param is not None: # set HRF parameters manually
            p1 = self.HRF_param['p1']
            p2 = self.HRF_param['p2']
            p3 = 0 # neglect time shifts for computational convenience
            p4 = 1

            t = np.linspace(0, self.T_HRF, self.T_ind_HRF).reshape(self.T_ind_HRF, 1)

            HRF_channel = (np.heaviside(t - p3, 1) * p4 * (((t - p3) ** (p1 - 1) * p2 ** (p1) * np.exp(-p2 * (t - p3))) / gamma(p1)))

            # Get index of maximum
            amplitude=1
            index = np.argmax(HRF_channel)
            p4 = amplitude / HRF_channel[index]
            print("p4=",p4)
            HRF=p4*HRF_channel.T
        else:
            raise ValueError("No HRF_param")

        return HRF

    def gen_impulse_series(self):
        """
        Method to generate multivariate impulse sequences, resembling underlying activity of neural populations.
        Multi-channel, the states follow a transition probability matrix in a Markovian fashion.
        """

        imp_seqs = np.zeros([self.channels, self.T_indices+self.T_ind_HRF-1])
        self.true_states = np.empty([self.T_indices+self.T_ind_HRF-1])
        # Imp sequences of groups follow a transition matrix
        i_prev=0

        if self.seed:
            random.seed(0)

        for t in range(0, self.T_indices+self.T_ind_HRF-1):
            #i = np.random.choice(np.arange(0, self.N), p=self.trans_prob_mat[i_prev,:].tolist()) #Choose group index
            i = random.choices(np.arange(0, self.N), weights=self.trans_prob_mat[i_prev, :].tolist())[0]
            if len(self.groups[i])!=0:
                imp_seqs[self.groups[i], t] = 1
            i_prev = i
            self.true_states[t] = i

        self.true_states = self.true_states.astype(int)

        # Add neural noise, if specified
        # imp_seqs = imp_seqs + np.random.normal(0, self.neural_std, size=(np.size(imp_seqs,axis=0),np.size(imp_seqs,axis=1)))

        return imp_seqs


    def gen_syn_fus(self):
        """
        Method to convolve the impulse sequences with the HRF.
        """
        if self.HRF_param is not None:
            fus_seq = np.zeros([self.channels, self.T_indices])
            for i in range(0, self.channels):
                fus_seq[i] = np.convolve(self.HRF[0,:], self.imp_seqs[i], mode='full')[:self.T_indices]

        return fus_seq

    def noisy_syn_fus(self):
        """
        Method to add I.I.D. Gaussian noise on observations
        """
        noisy_fus_seq = self.fus_seq + np.random.normal(0, self.noise_std, size=(self.channels,self.T_indices))
        return noisy_fus_seq


    def reconstruct(self,lambda_=None):
        """
        Method to reconstruct the underlying activity.
        :param lambda_: Regularization parameter for algorithm== "NNLS_const" and "LASSO"
        :return: No return
        """

        length = self.T_indices
        HRF_length = self.T_ind_HRF
        self.x_reconstructed = np.zeros((self.T_indices,self.channels))
        self.x_reconstructed_full = np.zeros((self.T_indices,self.channels))

        # Create convolution matrix
        H = sl.convolution_matrix(self.HRF[0, :], length + HRF_length - 1) #HRF specified by HRF_param
        H = H[HRF_length - 1:length + HRF_length - 1, :]  #In reality we only have a part of H
        self.H_ = H

        self.reconstruction_MSE = np.empty(self.channels)
        for i in range(self.channels):
            y = self.noisy_fus_seq[i,:].T
            if self.algorithm == "NNLS":
                # NNLS
                x_nnls, self.residual = nnls(H, y)
                x_nnls = x_nnls[:length]
            elif self.algorithm == "NNLS_const":
                A = H.T @ H + lambda_ * np.eye(length+HRF_length-1)
                b = H.T @ y
                x_nnls, self.residual = nnls(A, b)
                x_nnls = x_nnls[:length]
            elif self.algorithm == "LASSO":
                if np.isscalar(lambda_):
                    lambda_input = lambda_
                else:
                    lambda_input = lambda_[i]

                l = Lasso(alpha=lambda_input, fit_intercept=False, positive=True,tol=1e-4,max_iter=10000,selection='random') #Fit_intercept is false to avoid mean-subtraction
                l.fit(H,y)
                x_nnls = l.coef_
                x_nnls = x_nnls[:length]
                self.reconstruction_MSE[i] = sum((H[:,:length] @ x_nnls - y) ** 2)/len(x_nnls)

            self.x_reconstructed_full[:,i] = x_nnls
            self.x_reconstructed[:,i] = x_nnls

            # Set first samples to zero for removing overfitting to noise at beginning (only used for visualization, are discarded usually)
            self.x_reconstructed_full[0:self.T_ind_HRF,i] = 0
            self.x_reconstructed[0:self.T_ind_HRF,i] = 0

    # Plotting

    def plot_x_reconstructed(self):

        if self.algorithm == 'LASSO':
            alg = 'NNLASSO'
        else:
            alg = self.algorithm

        fig, ax = plt.subplots(self.channels, sharex="all",figsize=(int(5+self.T/48), self.channels*3.5))
        fig.suptitle("Reconstructed underlying activity $\hat{\mathbf{y}}_{m}$ using " + alg)
        fig.supylabel('Amplitude')

        for i in range(0, self.channels):
            (_, _, baseline) = ax[i].stem(self.time_reduced, self.x_reconstructed_full[self.T_ind_HRF-1:,i], markerfmt=" ")
            ax[i].set_ylabel("$\hat{{\mathbf{{y}}}}_{i}$".format(i=i+1),rotation=0, labelpad=20,va="center")

        plt.xlabel('Time [s]')
        plt.xlim([0, self.T-self.T_HRF-self.T_res])
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.fig_ID + '_x_reconstructed_' + alg + '_noisestd_' + str(self.noise_std) + '_channels_' + str(self.channels)
            plt.savefig(self.fig_path + '/' + fig_name + '.pdf', format='pdf')
        plt.show()

    def plot_imp_seq(self):

        fig, ax = plt.subplots(self.channels, sharex="all",figsize=(int(5+self.T/48), self.channels*3))
        fig.suptitle('True underlying activity $\mathbf{y}_m$')
        fig.supylabel('Amplitude')

        for i in range(0, self.channels):
            (_, _, baseline) = ax[i].stem(self.time, self.imp_seqs[i, :self.T_indices], markerfmt=" ")
            ax[i].set_ylabel("$\mathbf{{y}}_{i}$".format(i=i+1),rotation=0, labelpad=20, va="center")

        plt.xlabel("Time [s]")
        plt.xlim([0, self.T])
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.fig_ID + '_true_imp_seq' + '_noisestd_' + str(self.noise_std) + '_channels_' + str(self.channels)
            plt.savefig(self.fig_path + '/' + fig_name + '.pdf', format='pdf')
        plt.show()

    def plot_HRF_channels(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.time_HRF, self.HRF.T)
        plt.title('Hemodynamic response function')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.xlim([0, self.T_HRF])
        plt.grid(True)
        plt.tight_layout()
        if self.save_plots:
            fig_name = 'HRF'
            plt.savefig(self.fig_path + '/' + fig_name + '.pdf', format='pdf')
        plt.show()

    def plot_fus_seq(self):
        fig, ax = plt.subplots(self.channels, sharex="all",figsize=(int(5+self.T/48), self.channels*3.5))
        fig.suptitle('fUS time courses')
        fig.supylabel('Amplitude')

        for i in range(0,self.channels):
            ax[i].plot(self.time, self.fus_seq[i])
            ax[i].grid(True)

        plt.xlabel('Time [s]')
        plt.xlim([0, self.T])
        plt.grid(False)
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.fig_ID + '_clean_fus_seq' + '_noisestd_' + str(self.noise_std) + '_channels_' + str(self.channels)
            plt.savefig(self.fig_path + '/' + fig_name + '.pdf', format='pdf')
        plt.show()

    def plot_noisy_fus_seq(self):
        fig, ax = plt.subplots(self.channels, sharex=True,figsize=(int(5+self.T/48), self.channels*3.5))
        fig.suptitle('Noisy fUS time courses $\mathbf{f}_m$')
        fig.supylabel('Amplitude')

        for i in range(0, self.channels):
            ax[i].plot(np.linspace(0, self.T, int(self.T / self.T_res)), self.noisy_fus_seq[i])
            ax[i].set_ylabel("$\mathbf{{f}}_{i}$".format(i=i+1),rotation=0, labelpad=20, va="center")
            ax[i].grid(False)
        plt.xlim([0, self.T])
        plt.xlabel('Time [s]')
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.fig_ID + '_noisy_fus_seq' + '_noisestd_' + str(self.noise_std) + '_channels_' + str(self.channels)
            plt.savefig(self.fig_path + '/' + fig_name + '.pdf', format='pdf',dpi=200)
        plt.show()

    def plot_noise(self):
        fig, ax = plt.subplots(self.channels, sharex="all",figsize=(20, self.channels*3.5))
        fig.suptitle('Noise time courses $\mathbf{\epsilon}_m$')
        fig.supylabel('Amplitude')
        noise = np.random.normal(0, self.noise_std, size=(self.channels,self.T_indices))

        for i in range(0, self.channels):
            ax[i].plot(np.linspace(0, self.T, int(self.T / self.T_res)), noise[i,:])
            ax[i].set_ylabel("$\mathbf{{\epsilon}}_{i}$".format(i=i+1),rotation=0, labelpad=20)
            ax[i].grid(False)
        plt.xlim([0, self.T])
        plt.xlabel('Time [s]')

        plt.tight_layout()
        if self.save_plots:
            fig_name = self.fig_ID + '_noise' + '_noisestd_' + str(self.noise_std) + '_channels_' + str(self.channels)
            plt.savefig(self.fig_path + '/' + fig_name + '.pdf', format='pdf',dpi=200)
        plt.show()

    def paper_fig(self,index=1):
        fig, ax = plt.subplots(2, sharex=True,gridspec_kw={'height_ratios': [1,1.5]},figsize=(int(5+self.T/48), 6))

        ax[0].stem(self.time, self.imp_seqs[index, :self.T_indices], markerfmt=" ")

        ax[1].plot(np.linspace(0, self.T, int(self.T / self.T_res)), self.noisy_fus_seq[index])
        #ax[i].set_ylabel("$\mathbf{{f}}_{i}$".format(i=i+1),rotation=0, labelpad=20, va="center")
        ax[1].grid(False)
        ax[1].set_xlim([0, self.T])


        ax[1].set_xlabel('Time [s]')
        plt.tight_layout()
        if self.save_plots:
            fig_name = self.fig_ID + 'paper_fig' + '_noisestd_' + str(self.noise_std)
            plt.savefig(self.fig_path + '/' + fig_name + '.pdf', format='pdf',dpi=200)
        plt.show()

if __name__=="__main__":
    channels = 2
    T = 720
    T_res = 0.25
    noise_std = 0.1
    fs = 4

    groups = [[0],[0,1],[]]
    # Create transition probability matrix
    N = channels+1
    bias=15
    mat = np.random.rand(N,N)
    np.fill_diagonal(mat,np.concatenate((np.ones(N-1),np.array([bias]))))
    trans_prob_mat = mat/np.sum(mat,axis=1)[:,None]

    # HRF parameters
    HRF_param = {}
    HRF_param['p1'] = 4.0 #4.4 #1.99
    HRF_param['p2'] = 1.5 #1.75 #1.27
    HRF_param['p3'] = 0

    # Plots
    plot_info =  {}
    plot_info['save_plots']=False
    plot_info['fig_path']=None
    plot_info['fig_ID']='Bio_TEST'

    for i in range(1):

        d = SynData(channels, T, T_res, noise_std, fs, trans_prob_mat, HRF_param=HRF_param, groups=groups, algorithm="LASSO",plot_info = plot_info)

        d.plot_HRF_channels()
        # d.plot_fus_seq()
        d.plot_noisy_fus_seq()
        d.plot_imp_seq()
        d.plot_noise()
        #
        # # Reconstruct
        # d.reconstruct(lambda_=0.001) # Testing purpose
        # d.plot_x_reconstructed()





