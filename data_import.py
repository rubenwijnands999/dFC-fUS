import numpy as np
import os
import h5py
import scipy.signal as ss
import pylab as py
import matplotlib.pyplot as plt
import util

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft, fftfreq

class Data(object):

    def __init__(self,ID,slice,ndpi=-1,mask_threshold=0.1):

        self.ID, self.slice, = ID, slice
        self.mask_threshold = mask_threshold

        # Read configurations.ini for accessing fUS server
        self.config = util.read_config_file()

        # Create paths
        self.paths = util.create_paths_dict(ID,slice)

        # Read CSV file
        self.MiceDict = util.read_csv_file(self.paths['csv_path'])

        # Check whether data is stored locally
        local_data_flag = os.path.isfile(self.paths['loc_pdi_path']) and os.path.isfile(self.paths['loc_param_path'])
        if local_data_flag:
            print("Using local data")
        else:
            print("Importing data from server")
            self.access_srv()

        # Import imaging parameters
        self.scan_parameters = self.import_scan_parameters(local_data_flag)

        # Read power_doppler_images.dat file
        pdi_stack = self.import_pdi(local_data_flag,npdi=ndpi)

        # Find centers from CSV table and cut borders
        self.PDI = self.remove_borders(pdi_stack)

    def access_srv(self):
        mnt_folder = self.config["DataImportSettings"]['mount_folder']
        serv_name = self.config["DataImportSettings"]['server_name']
        share_name = self.config["DataImportSettings"]['share_name']
        user_name = self.config["DataImportSettings"]['username']
        password = self.config["DataImportSettings"]['password']

        mnt_cmd = "mount_smbfs //" + user_name + ":" + password + "@" + serv_name + "/" + share_name + " " + mnt_folder

        if not os.path.exists(mnt_folder):
            os.makedirs(mnt_folder)

        # First check if connection is already present
        if not os.path.ismount(mnt_folder):
            print("Creating mount")
            os.system(mnt_cmd)
            if not os.path.ismount(mnt_folder):
                raise Exception("Mounting procedure not possible, check connection to the network")
            else:
                print("Mounting successful")
        else:
            print("Using existing mount")

    def import_scan_parameters(self,local_data_flag):
        """
        Function to import the ScanParameters.mat file
        :param param_path: path to the folder containing the ScanParameters.mat file
        """
        scan_parameters = {}

        if local_data_flag:
            param_path = self.paths['loc_param_path']
        else:
            param_path = self.paths['srv_param_path']

        if os.path.isfile(param_path):
            current_parameters = h5py.File(param_path, 'r')
            scan_parameters['Nz'] = int(current_parameters.get('RECON/Nz')[()])
            scan_parameters['Nx'] = int(current_parameters.get('RECON/Nx')[()])
            scan_parameters['Fs'] = 1/(int(current_parameters.get('ACQ/pdi_trigger_time')[()])/1000)
        else:
            raise Exception("No scan parameter file found.")

        return scan_parameters

    def import_pdi(self, local_data_flag, npdi):
        """
        Import the PDI
        :param x,y: pixel location
        :param npdi: number of PDIs. Type -1 for all PDIs.
        :return: returns the PDI stack of dim(nz, nx, nt)
        Author: Bas Generowicz
        Contributor: Ruben Wijnands
        """

        if local_data_flag:
            pdi_path = self.paths['loc_pdi_path']
        else:
            pdi_path = self.paths['srv_pdi_path']

        if npdi > 0:
            pdi_samples = npdi * self.scan_parameters['Nz'] * self.scan_parameters['Nx']
        else:
            pdi_samples = -1

        if os.path.isfile(pdi_path):
            pdi_stack = np.fromfile(pdi_path, dtype='single', count=pdi_samples)
        else:
            raise Exception("No PDI file found")

        self.Nf = int(len(pdi_stack) / self.scan_parameters['Nz'] / self.scan_parameters['Nx'])
        pdi_stack = np.reshape(pdi_stack, (self.scan_parameters['Nz'], self.scan_parameters['Nx'], self.Nf),order='F')

        return pdi_stack

    def find_entry(self):
        """
        Find entry in the CSV table named 'MiceDict' for a given recID and sectionID
        :return: entry index
        """
        entry_recID = [i for i, e in enumerate(self.MiceDict['recID']) if e == str(self.ID)]
        entry_sectionID = [i for i, e in enumerate(self.MiceDict['sectionID']) if e == str(self.slice)]

        entry=None
        for rec in entry_recID:
            if rec in entry_sectionID:
                entry = rec
        if entry is None:
            raise Exception("ID and slice mismatch")

        return entry

    def remove_borders(self, pdi_stack):

        entry = self.find_entry()
        x_centre = int(self.MiceDict['x_centre'][entry])
        y_centre = int(self.MiceDict['y_centre'][entry])

        # Common border removal after for all mice after centering
        start = y_centre - int(self.config['BorderFromCentre']['upper_border'])
        end = y_centre + int(self.config['BorderFromCentre']['lower_border'])
        left = x_centre - int(self.config['BorderFromCentre']['left_border'])
        right = x_centre + int(self.config['BorderFromCentre']['right_border'])

        # Adjust scan parameters to new parameters
        self.scan_parameters['Nz'] = end-start
        self.scan_parameters['Nx'] = right-left
        PDI = pdi_stack[start:end, left:right, :]
        return PDI

    # From here, specialized tools operating on data directly
    def load_GUI_data(self):
        list = []
        offset = 1 - np.min(self.PDI) # Add offset for app video, shows log-video so ensure no negative values
        for i in range(0, len(self.PDI[0, 0, :])):
            frame = offset + self.PDI[:, :, i]
            list.append(frame.tolist())
        return list

    # Below are data tools
    def find_max(self):
        max_value = np.max(self.PDI)
        return max_value

    def find_min(self):
        min_value = np.min(self.PDI)
        return min_value

    def log_data(self):
        self.PDI = np.log10(1 - np.min(self.PDI) + self.PDI)

    def mean_fig(self):
        mean_figure = np.mean(self.PDI, axis=2)
        return mean_figure

    def var_fig(self):
        var_figure = np.var(self.PDI, axis=2)
        return var_figure

    def get_mask(self):
        # Variance based mask
        var = np.log10(self.var_fig())
        threshold = np.min(var) + self.mask_threshold*(np.max(var) - np.min(var))
        mask = np.empty((self.scan_parameters['Nz'], self.scan_parameters['Nx']))

        self.thres_val = threshold
        for i in range(0, self.scan_parameters['Nz']):
            for j in range(0, self.scan_parameters['Nx']):
                if var[i,j]>threshold:
                    mask[i,j] = 1
                else:
                    mask[i,j] = 0

        return mask


    def fir_low_pass_data(self, cutoff=0.3, pass_band=0.1, stop_att=40, x=None,y=None):
        """
        :param cutoff: Cut-off frequency of filter [Hz]
        :param pass_band: Pass-band width [Hz]
        :param stop_att: Desired attenuation in the stop band [dB]
        :return No return
        """

        nyq_rate = self.scan_parameters['Fs']/2
        width = pass_band / nyq_rate
        N, beta = ss.kaiserord(stop_att, width)

        if self.Nf < N:
            print("Not enough samples for the given filter order")
        else:
            if (x and y) is None:
                # Use firwin with a Kaiser window to create a lowpass FIR filter.
                self.b = ss.firwin(N, cutoff / nyq_rate, window=('kaiser', beta))
                self.PDI = ss.lfilter(self.b, 1, self.PDI, axis=2)
                self.PDI = self.PDI[:,:,N:]
                self.Nf = self.Nf - N
            else:
                self.b = ss.firwin(N, cutoff / nyq_rate, window=('kaiser', beta))
                temp = ss.filtfilt(self.b, 1, self.PDI[x,y,:])
                filt_time_course = temp #[N:]
                x_axis = [i for i in range(0,self.Nf)]
                return filt_time_course, x_axis

    def normalize_temporal(self):
        max_matrix = np.amax(self.PDI, axis=2)
        min_matrix = np.amin(self.PDI, axis=2)

        for i in range(0, self.scan_parameters['Nz']):
            for j in range(0, self.scan_parameters['Nx']):
                if self.mask[i,j]==1:
                    self.PDI[i, j, :] = (self.PDI[i, j, :] - min_matrix[i, j]) / (
                                max_matrix[i, j] - min_matrix[i, j])
                else:
                    self.PDI[i, j, :] = np.zeros([1, self.Nf])

    def normalize_spatial(self):
        for i in range(0, self.Nf):
            frame = self.PDI[:,:,i]
            max = np.max(frame)
            min = np.min(frame)
            self.PDI[:,:,i] = (self.PDI[:,:,i] - min)/(max-min)

    def subtract_mean_spatial(self,mask=True,mask_threshold=None):

        if mask_threshold is not None:
            self.mask_threshold = mask_threshold

        if mask==True:
            # Only take pixels outside the mask to subtract the mean
            self.mask = self.get_mask()
            locs = np.where(self.mask==0)
            for i in range(0, self.Nf):
                frame = self.PDI[:,:,i]
                mean = np.mean(frame[locs[0],locs[1]])
                self.PDI[:,:,i] = self.PDI[:,:,i] - mean
        else:
            for i in range(0, self.Nf):
                frame = self.PDI[:,:,i]
                mean = np.mean(frame)
                self.PDI[:,:,i] = self.PDI[:,:,i] - mean

    def standardize_temporal(self):
        mean = self.mean_fig()
        var = self.var_fig()

        for i in range(0, self.scan_parameters['Nz']):
            for j in range(0, self.scan_parameters['Nx']):
                if self.mask[i,j]==1:
                    self.PDI[i,j,:] = (self.PDI[i,j,:]-mean[i,j])/np.sqrt(var[i,j])

        min = self.find_min()

        for i in range(0, self.scan_parameters['Nz']):
            for j in range(0, self.scan_parameters['Nx']):
                if self.mask[i,j]==0:
                    self.PDI[i,j,:] = min * np.ones([1, self.Nf])


# Plotting

    def plot_freq_resp_filt(self):
        nyq_rate = self.scan_parameters['Fs']/2
        w, h = ss.freqz(self.b, worN=8000)
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.5)

        fig.add_subplot(2, 1, 1)
        plt.plot((w/np.pi)*nyq_rate, 20 * np.log10(np.absolute(h)), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$|H(f)|^2$ [dB]')
        plt.title('Frequency response low-pass filter')
        plt.ylim(1*(np.min(20 * np.log10(np.absolute(h)))+50),1)
        plt.grid(True)

        fig.add_subplot(2, 1, 2)

        plt.plot((w / np.pi) * nyq_rate, py.unwrap(py.arctan2(py.imag(h),py.real(h))), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase [Degree]')
        plt.title('Phase response low-pass filter')
        #plt.ylim(1 * (np.min(20 * np.log10(np.absolute(h))) + 50), 1)
        plt.grid(True)

        plt.show()

    def plot_noise_hist(self,x,y,res=0.01):
        time_course = self.PDI[x,y,:]
        n_bins = 100 #int((np.max(time_course)-np.min(time_course))/res)

        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        plt.hist(time_course, bins=n_bins)
        plt.xlabel('Signal value')
        plt.ylabel('Frequency')
        plt.title('Histrogram of pixel values over time')
        #plt.xlim(0,0.5)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.show()

    def plot_var_hist(self,res=0.001):
        var = self.var_fig().flatten()
        n_bins = int(np.max(var)/res)

        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        plt.hist(var, bins=n_bins)
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        plt.title('Histrogram of pixel variance over time')
        plt.xlim(0,0.5)
        #plt.xscale('log')
        plt.yscale('log')
        plt.show()


    def plot_power_spectrum(self):
        # Remove DC component first
        mean = self.mean_fig()
        ps = np.zeros(self.Nf)
        count=0

        for i in range(0, self.scan_parameters['Nz']):
            for j in range(0, self.scan_parameters['Nx']):
                if self.mask[i,j]==1:
                    ps = ps + abs(fft(self.PDI[i,j,:])/self.Nf)
                    count=+1

        ps = ps/count

        freq_ax = fftfreq(self.Nf, 1/self.scan_parameters['Fs'])
        index = np.argsort(freq_ax)
        index = index[range(int(self.Nf/2),self.Nf)]

        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        plt.plot(freq_ax[index],ps[index])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('|Y(f)|')
        plt.title('Averaged single-sided spectrum of masked pixels')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

    def plot_mask(self):
        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        plt.imshow(self.mask)
        plt.xlabel('Width')
        plt.ylabel('Depth')
        plt.title('Mask')
        plt.show()

    def plot_var_fig(self):
        var = np.log10(self.var_fig())
        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        plt.imshow(var)
        plt.colorbar()
        plt.xlabel('Width')
        plt.ylabel('Depth')
        plt.title('Log pixel variance')
        plt.show()

    def save_var_mask_figures(self,rec,ID,section,PreProcessingName,fig_path=None):
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
        fig.suptitle("Blood vessel mask for recID " + str(ID) + ", section " + str(section))
        fig.supxlabel("Width")
        im = ax[0].imshow(np.log10(self.var_fig()))
        ax[0].set(adjustable='box', aspect='auto')
        ax[0].set_title('Log pixel variance')
        ax[0].set_ylabel('Depth')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax[1].imshow(self.mask)
        ax[1].set(adjustable='box', aspect='auto')
        ax[1].set_title('Thresholded mask of blood vessels')

        plt.tight_layout()
        if fig_path is not None:
            fig_name = PreProcessingName + '_TE_' + str(rec) + '_blood_vessel_mask_threshold_' + str(self.mask_threshold)
            pre_fig_path_entry = os.path.join(*[fig_path, 'TE_' + str(rec)])
            if not os.path.exists(pre_fig_path_entry):
                os.makedirs(pre_fig_path_entry)
            plt.savefig(pre_fig_path_entry + '/' + fig_name + '.pdf', format='pdf',dpi=200)
            print("Saving image")
        plt.show()

if __name__=="__main__":

    # Settings
    TE=1
    ID = 890
    slice_num = 3
    npdi = 10

    obj = Data(ID, slice_num, npdi)
    obj.subtract_mean_spatial(mask=True,mask_threshold=float(obj.MiceDict['mask_threshold'][TE]))

    obj.plot_var_fig()
    obj.plot_mask()

