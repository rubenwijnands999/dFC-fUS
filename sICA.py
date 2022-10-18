import numpy as np
import matplotlib.pyplot as plt
import os
import util

from sklearn.decomposition import FastICA
from data_import import Data

class sICA(object):

    def __init__(self,sICA_ID,TableEntry,NumComp,seed_nr):
        """
        Function: Initialize sICA object.
        Author: Ruben Wijnands.

        :param sICA_ID: ID of specfic run. Can be used for testing variability in sICA runs. [int]
        :param TableEntry: Row in the CSV table with mice. [int]
        :param NumComp: Number of extracted components in the sICA procedure. [int]
        :param seed_nr: Seed for reproducible results. [int]
        """

        self.TableEntry, self.NumComp, self.seed_nr = TableEntry, NumComp, seed_nr

        # Create paths
        self.paths = util.create_paths_dict(sICA_ID=sICA_ID,TableEntry=TableEntry)

        # Read mice table
        self.MiceDict = util.read_csv_file(self.paths['csv_path'])

        # Beginning of name of sICA components
        self.fig_name = "RecID_" + self.MiceDict["recID"][self.TableEntry] + "_sectionID_" + self.MiceDict["sectionID"][self.TableEntry]


    def import_data(self):
        # Import data from server
        ID = self.MiceDict["recID"][self.TableEntry]
        section = self.MiceDict["sectionID"][self.TableEntry]
        npdi = -1  # Type -1 for all PDIs
        data_obj = Data(ID, section, npdi)

        # Extract parameters
        self.Nz = data_obj.scan_parameters['Nz']
        self.Nx = data_obj.scan_parameters['Nx']
        self.Nf = data_obj.Nf

        # Apply log transform
        data_obj.log_data()

        # Vectorize images
        self.X = np.reshape(data_obj.PDI,(self.Nz * self.Nx, self.Nf ), order="F")

    def pre_processing(self):
        # Standardize temporally
        self.X = (self.X - np.mean(self.X, axis=1)[:, None])
        self.X = np.divide(self.X, np.std(self.X, ddof=1, axis=1)[:, None])

    def run_sICA(self):
        # sICA using FastICA. In the FastICA package, first each image is centered and PCA is applied.
        ica = FastICA(n_components=self.NumComp, random_state=self.seed_nr)
        self.S = ica.fit_transform(self.X)
        self.A = ica.mixing_

    def post_processing(self):
        # Reconstruct component images
        self.im_seq = np.zeros((self.NumComp, self.Nz, self.Nx))
        for comp in range(self.NumComp):
            self.im_seq[comp, :, :] = self.S[:, comp].reshape([self.Nx, self.Nz]).T

    # Plotting and saving
    def plot_component_timecourses(self,plot=True,save=True):
        for comp in range(self.NumComp):
            fig = plt.figure(figsize=(5, 7))
            fig.add_subplot(2, 1, 1)
            plt.title("Component " + str(comp) + ", recID " + str(self.MiceDict["recID"][self.TableEntry]) + ", section " + self.MiceDict["sectionID"][self.TableEntry])
            plt.imshow(self.im_seq[comp,:,:])
            plt.colorbar()
            fig.add_subplot(2, 1, 2)
            plt.plot(self.A[:, comp])
            plt.tight_layout()
            if save:
                plt.savefig(os.path.join(self.paths['fig_sICA_entry_time_course_path'], self.fig_name + '_comp_' + str(comp) + '.eps'), format='eps')
                np.save(os.path.join(self.paths['fig_sICA_entry_path'], self.fig_name + '_seed_' + str(self.seed_nr)), self.im_seq)
                if not plot:
                    plt.close()
            if plot:
                plt.show()

    def plot_components(self,plot=True,save=True):
        for comp in range(self.NumComp):
            plt.figure(figsize=(7, 5))
            plt.title("Component " + str(comp) + ", recID " + str(self.MiceDict["recID"][self.TableEntry]) + ", section " + self.MiceDict["sectionID"][self.TableEntry])
            plt.imshow(self.im_seq[comp,:,:])
            plt.colorbar(fraction=0.039, pad=0.04)
            plt.tight_layout()
            if save:
                #plt.savefig(os.path.join(self.paths['fig_sICA_entry_path'], self.fig_name + '_comp_' + str(comp) + '_seed_' + str(self.seed_nr) + '.eps'), format='eps')
                plt.savefig(os.path.join(self.paths['fig_sICA_entry_path'], self.fig_name + '_comp_' + str(comp) + '_seed_' + str(self.seed_nr) + '.png'))
                np.save(os.path.join(self.paths['fig_sICA_entry_path'], self.fig_name + '_seed_' + str(self.seed_nr)), self.im_seq)
                if not plot:
                    plt.close()
            if plot:
                plt.show()


if __name__ == "__main__":

    # Give ID name
    sICA_ID = 11
    TableEntry = 1 # Read from CSV table (row wise)
    NumComp = 25
    seed_nr = 0

    # Apply sICA routine
    s = sICA(sICA_ID,TableEntry,NumComp,seed_nr)
    s.import_data()
    s.pre_processing()
    s.run_sICA()
    s.post_processing()
    s.plot_components(plot=True,save=True)

