import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import matplotlib.colors as colors
import matplotlib
import copy
import os
import json
import util
import ast

from scipy.ndimage import zoom
from GUI_tools import dilation
from data_import import Data
from mpl_toolkits.axes_grid1 import make_axes_locatable
from movement_extraction import MovementExtraction
from scipy.stats import linregress


class PreProcessing(object):

    def __init__(self,PreProcessingName,sICA_RUN=None,seed_num=None,overwrite=False,specific_rec=None):
        self.PreProcessingName = PreProcessingName
        self.sICA_RUN = sICA_RUN
        self.seed_num = seed_num

        # Read configurations.ini for accessing fUS server
        self.config = util.read_config_file()

        # Create paths
        self.paths = util.create_paths_dict(PreProcessingName=PreProcessingName,sICA_ID=sICA_RUN)

        # Read mice CSV table
        self.MiceDict = util.read_csv_file(self.paths['csv_path'])

        # Check if entered PreProcessingName already exists in simulation folder (check JSON)
        if os.path.exists(self.paths['json_pre_path']) and (not overwrite):
            extract_new_data = False
            print("Using data from file " + self.PreProcessingName + " located at " + self.paths['json_pre_path'])
        else:
            extract_new_data = True
            print("Performing data extraction steps")

        if extract_new_data:

            self.num_recs = len(self.MiceDict['recID'])  # Number of recordings in CSV table
            self.num_regions = int(self.config['Analysis']['num_regions'])  # Anatomical regions per mouse
            self.Fs = int(self.config['Analysis']['Fs'])  # Sampling frequency
            self.Nf = int(self.config['Analysis']['Nf'])  # Number of samples in a recording

            # Number of samples in sliding window of moving_average_subtract()
            self.baseline_subtraction_window_size = int(int(self.config['Analysis']['baseline_subtraction_window_size']) * self.Fs)

            # Detrending parameters
            self.detrending_param = {}
            self.detrending_param['variance_window_size'] = int(self.config['Analysis']['sliding_variance_window_size'])
            self.detrending_param['rel_act_threshold'] = ast.literal_eval(self.config['Analysis']['rel_act_threshold'])
            self.detrending_param['movement_extension'] = int(self.config['Analysis']['movement_extension'])

            if len(self.detrending_param['rel_act_threshold']) < int(self.config['Analysis']['num_regions']):
                raise Exception("Enter more entries in 'rel_act_threshold' in configurations.ini file")

            # Threshold for sICA component pixels
            self.sICA_threshold = float(self.config['Analysis']['sICA_threshold'])

            # List with ROIs, read from CSV table and colors specified in configurations.ini
            self.ROI_list, self.ROI_color_list = util.create_region_color_list()

            self.F_list = {}  # Contains the mean fUS ROI time courses, filled by methods mean_fus()
            self.F_final = {}  # Detrended mean fUS signals

            self.Mouse_ROI_pixels = {}  # All pixels inside an ROI from the warp
            self.Thresholded_Mouse_ROI_pixels = {}  # Pixels inside an ROI after thresholding sICA components within the ROI warp.

            self.Movement_dict = {}
            self.Movement_dict_subsampled = {}
            self.Movement_dict_subsampled_extended = {}
            self.sliding_window_vars = {}
            self.Lin_reg_masks = {}
            self.detrended_signals = {}

            self.warps = self.get_warps()  # Get brain warps per ID, create in the GUI!
            self.sICA_image_dict = self.import_sICA_imseq()  # Import sICA images needed, read from CSV table
            self.invert_sICA_imseq()  # works on self.sICA_image_dict directly. Inverts such that an ROI has positive values.

            # Pre-process and subtract mean fUS time courses of ROIs here
            for rec in range(self.num_recs):

                # Only perform pre-processing on entries filled in csv table
                flag = False
                for region in self.ROI_list[:-2]:
                    if self.MiceDict[region][rec]:
                        flag = True
                if (specific_rec is not None) and (len(specific_rec)!=0):
                    if rec not in specific_rec:
                        flag = False

                if flag:
                    ID = self.MiceDict['recID'][rec]
                    section = self.MiceDict['sectionID'][rec]

                    # Import data and subtract scattering effect
                    rec_obj = Data(ID, section)
                    rec_obj.subtract_mean_spatial(mask=True,mask_threshold=float(self.MiceDict['mask_threshold'][rec]))

                    print("ID=",ID,", SEC=",section,", Fs=",rec_obj.scan_parameters['Fs'])

                    # Plot and save mask from rec_obj
                    rec_obj.save_var_mask_figures(rec,ID,section,self.PreProcessingName,fig_path=self.paths['fig_pre_path'])

                    # Extract movement
                    print("Extracting movement")
                    m = MovementExtraction(rec)
                    self.Movement_dict[str(rec)] = m.movement
                    self.Movement_dict_subsampled[str(rec)] = m.movement_sub_sampled

                    # Normalizing of sICA components: Set background to 0 (obtained from mask), highest activity point to 1.
                    self.norm_sICA_components(rec_obj, rec)

                    # If multiple components per region, combine by keeping the maximum pixel value.
                    self.combine_sICA_components(rec)

                    # Threshold within anatomical region, mean and set to zero-baseline
                    self.find_ROI_pixels(rec)
                    self.threshold_ROI_pixels(rec_obj,rec,self.sICA_threshold)

                    # Take mean of fUS pixel signals belonging to a certain ROI.
                    self.F_list[str(rec)] = self.mean_fus(rec_obj,rec)

                    # Detrending of fUS signals. Results stored in self.detrended_signals.
                    self.detrend(rec)

                    # Remove baseline
                    self.moving_average_subtract(rec,self.baseline_subtraction_window_size)

            # Create param dictionary
            self.param = {}
            self.param['num_recs'] = self.num_recs
            self.param['num_regions'] = self.num_regions
            self.param['baseline_subtraction_window_size'] = self.baseline_subtraction_window_size
            self.param['sICA_threshold'] = self.sICA_threshold
            self.param['Fs'] = self.Fs # Assumed the same for all mice
            self.param['T_res'] = 1/self.Fs
            self.param['Nf'] = self.Nf

            # Export results to JSON file
            self.export_json()

        else:
            # Read data and parameters from JSON file
            self.import_json()

    # Methods
    def get_warps(self):
        """
        Import warps for each recording entry in the CSV table. The warps can be created by using the fUS app.
        :return: 3D matrix containing warps
        """
        heigth = int(self.config['BorderFromCentre']['lower_border']) + int(self.config['BorderFromCentre']['upper_border'])
        width = int(self.config['BorderFromCentre']['right_border']) + int(self.config['BorderFromCentre']['left_border'])
        warp_matrix = np.zeros((self.num_recs,heigth,width))
        for rec in range(self.num_recs):
            ID = self.MiceDict['recID'][rec]
            path_npy = os.path.join(self.paths['warp_path'],str(ID)+'_warp.npy')
            try:
                warp = np.load(path_npy)
                warp_rescaled = zoom(warp, heigth/np.shape(warp)[0], order=0)  # Nearest neighbour interpolation
                warp_matrix[rec,:,:] = warp_rescaled
            except:
                print("Missing numpy warp")

        return warp_matrix

    def get_warp_border(self,rec,region_index):
        """
        Find the border of a brain region by performing a dilation once.
        :param rec: recording entry in CSV table
        :param region_index: index of an anatomical brain region in the ROI_color_list
        :return: border of a brain region in the warp
        """
        color=self.ROI_color_list[region_index]
        original = copy.deepcopy(self.warps[rec,:,:])
        border = dilation(original, color) - self.warps[rec,:,:]
        border[abs(border) > 0] = color
        return border

    def import_sICA_imseq(self):
        """
        Import the sICA images that are selected in the CSV table. If multiple sICA images are selected,
        a 3D matrix is created.
        :return: Dictionary with sICA images, assigned to a brain region.
        """
        sICA_image_dict = {}
        for rec in range(self.num_recs):
            imseq = None
            sICA_image_dict[str(rec)] = {}
            ID = self.MiceDict['recID'][rec]
            section = self.MiceDict['sectionID'][rec]
            paths = util.create_paths_dict(PreProcessingName=self.PreProcessingName, sICA_ID=self.sICA_RUN,TableEntry=rec)

            np_name = "RecID_" + str(ID) + "_sectionID_" + str(section) + "_seed_" + str(self.seed_num) + ".npy"

            try:
                imseq = np.load(os.path.join(paths['fig_sICA_entry_path'], np_name))
            except:
                print("No numpy image sequence found, apply sICA procedure first")

            if imseq is not None:
                for region in self.ROI_list[:-2]:
                    # Find all integer numbers in the table
                    list_ints = []
                    list = self.MiceDict[region][rec].split(",")
                    for item in list:
                        try:
                            list_ints.append(int(item))
                        except:
                            pass
                    try:
                        # If only one integer number, only import one image
                        if len(list_ints) == 1:
                            sICA_component_number = int(self.MiceDict[region][rec].split(",")[0])
                            sICA_image_dict[str(rec)][region] = imseq[sICA_component_number, :, :]
                        else:
                            #if multiple integer numbers
                            image_array = []
                            for comp in list_ints:
                                image_array.append(imseq[comp, :, :])
                            sICA_image_dict[str(rec)][region] = np.array(image_array)
                    except:
                        print("No valid number in CSV table")

        return sICA_image_dict

    def invert_sICA_imseq(self):
        """
        Inversion of sICA images due to sign ambiguity.
        :return: No return. Acts on the imported sICA images directly.
        """
        for rec in range(self.num_recs):
            for region in self.ROI_list[:-2]:
                try:
                    if np.ndim(self.sICA_image_dict[str(rec)][region])==2:
                        inversion = self.MiceDict[region][rec].split(",")[1]
                        if inversion == 'yes':
                            self.sICA_image_dict[str(rec)][region] = self.sICA_image_dict[str(rec)][region] * -1
                    else:
                        num_comps = np.shape(self.sICA_image_dict[str(rec)][region])[0]
                        inversion = self.MiceDict[region][rec].split(",")[num_comps:2*num_comps]
                        for comp in range(num_comps):
                            if inversion[comp] == 'yes':
                                self.sICA_image_dict[str(rec)][region][comp,:,:] = self.sICA_image_dict[str(rec)][region][comp,:,:] * -1

                except:
                    print("No valid inversion statement in CSV table")

    def norm_sICA_components(self,rec_obj,rec):
        """
        Normalize the sICA components. The 'background' of a sICA image is set to zero.
        :param rec_obj: Entire imported data object (since the mask is needed)
        :param rec: Recording entry in CSV table
        :return:
        """
        for region in self.ROI_list:
            outside_mask_pixels = np.where(rec_obj.mask == 1)
            try:
                if np.ndim(self.sICA_image_dict[str(rec)][region])==2:
                    mean = np.mean(self.sICA_image_dict[str(rec)][region][outside_mask_pixels[0],outside_mask_pixels[1]])
                    max = np.max(self.sICA_image_dict[str(rec)][region])
                    self.sICA_image_dict[str(rec)][region] = (self.sICA_image_dict[str(rec)][region]-mean)/(max-mean)
                else:
                    num_comps = np.shape(self.sICA_image_dict[str(rec)][region])[0]
                    for comp in range(num_comps):
                        mean = np.mean(self.sICA_image_dict[str(rec)][region][comp, outside_mask_pixels[0], outside_mask_pixels[1]])
                        max = np.max(self.sICA_image_dict[str(rec)][region][comp,:,:])
                        self.sICA_image_dict[str(rec)][region][comp,:,:] = (self.sICA_image_dict[str(rec)][region][comp,:,:] - mean) / ( max - mean)

            except:
                pass

    def combine_sICA_components(self,rec):
        """
        Combine sICA if there are multiple sICA images for a single brain region.
        The pixel-wise maximum, after inversion and normalization of sICA images, is kept.
        :param rec: Recording entry.
        :return: No return. Acts on data directly.
        """
        for region in self.ROI_list:
            try:
                if np.ndim(self.sICA_image_dict[str(rec)][region]) == 3:
                    self.sICA_image_dict[str(rec)][region] = np.max(self.sICA_image_dict[str(rec)][region],axis=0)
            except:
                pass

    def find_ROI_pixels(self,rec):
        """
        Identify pixels that belong to an brain region of interest (ROI).
        :param rec: recording entry in CSV table.
        :return: No return. Created Mouse_ROI_pixels dictionary object.
        """
        self.Mouse_ROI_pixels[str(rec)] = {}
        for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):
            self.Mouse_ROI_pixels[str(rec)][region] = np.where(self.warps[rec,:,:] == self.ROI_color_list[i])

    def threshold_ROI_pixels(self,rec_obj,rec,threshold_percentage):
        """
        Threshold the pixels inside ROIs based on a percentage threshold between
        the maximum pixel value and mean background value of the sICA image.
        :param rec_obj: Import data object.
        :param rec: Recording entry in CSV table.
        :param threshold_percentage: Percentage threshold between the maximum and mean background pixel value of the sICA image.
        :return: No return. Created Thresholded_Mouse_ROI_pixels dictionary object.
        """
        self.Thresholded_Mouse_ROI_pixels[str(rec)] = {}
        for region in self.ROI_list[:-2]:
            x = self.Mouse_ROI_pixels[str(rec)][region][0]
            y = self.Mouse_ROI_pixels[str(rec)][region][1]
            try:
                sICA_image = self.sICA_image_dict[str(rec)][region]
                max = np.max(sICA_image[x,y])
                outside_mask_pixels = np.where(rec_obj.mask==1)
                mean = np.mean(sICA_image[outside_mask_pixels[0],outside_mask_pixels[1]])
                dynamic_threshold = threshold_percentage * (max - mean)
                locs = np.where(sICA_image[x,y]>dynamic_threshold)
                self.Thresholded_Mouse_ROI_pixels[str(rec)][region] = (x[locs],y[locs])
            except:
                print("Thresholding not possible. Complete table or run sICA procedure")

    def mean_fus(self,rec_obj,rec):
        """
        Compute the mean fUS signal in the thresholded ROIs.
        :param rec_obj: Import data object.
        :param rec: Recording entry in CSV table.
        :return: No return. Result stored in F.
        """
        F = np.zeros((self.num_regions, self.Nf))

        for region,i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):
                try:
                    if int(self.MiceDict['recID'][rec]) == 959:
                        signal_5Hz = np.mean(rec_obj.PDI[self.Thresholded_Mouse_ROI_pixels[str(rec)][region][0],
                                             self.Thresholded_Mouse_ROI_pixels[str(rec)][region][1], :], axis=0)
                        F[i,:] = ss.resample(signal_5Hz, self.Nf) # Accidentally sampled at 5 Hz, resample to 4 Hz
                    else:
                        F[i,:] = np.mean(rec_obj.PDI[self.Thresholded_Mouse_ROI_pixels[str(rec)][region][0],self.Thresholded_Mouse_ROI_pixels[str(rec)][region][1],:],axis=0)

                except:
                    print("Mean operation incomplete")
        return F

    def sliding_window_var(self,rec):
        """
        Apply sliding window that computes the variance in each window.
        This result is needed for detection of the baseline.
        :param rec: Recording entry of CSV table.
        :return: Output stored in self.sliding_window_vars.
        """
        window_size = self.detrending_param['variance_window_size'] * self.Fs  # in samples now

        self.sliding_window_vars[str(rec)] = np.zeros((self.num_regions,self.Nf))
        for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):

            for idx in range(self.Nf):
                window = []
                for j in range(int(-window_size / 2), int(window_size / 2) + 1):
                    if (idx + j >= 0 and idx + j < self.Nf):
                        window.append(self.F_list[str(rec)][i,idx + j])

                self.sliding_window_vars[str(rec)][i,idx] = np.var(window, ddof=1)

    def extend_movement_mask(self,rec):
        """
        Extend the active parts of the movement mask (indicating movement of the mouse) by the duration of the HRF.
        :param rec: Recording entry of CSV table.
        :return: Recording entry of CSV table.
        """
        window_size = self.detrending_param['movement_extension'] * self.Fs # In samples
        mov = self.Movement_dict_subsampled[str(rec)]
        self.Movement_dict_subsampled_extended[str(rec)] = np.zeros(self.Nf)

        for idx in range(self.Nf):
            if mov[idx]==True: # Movement
                for i in range(int(window_size)):
                    if idx + i < self.Nf:
                        self.Movement_dict_subsampled_extended[str(rec)][idx + i] = True

    def detrend(self,rec):
        """
        Detrending of the fUS mean time courses by subtracting the line fitted by a linear regression.
        :param rec: Recording entry of CSV table.
        :return: No return. Results stored in self.detrended_signals.
        """
        self.sliding_window_var(rec)
        self.extend_movement_mask(rec)

        self.Lin_reg_masks[str(rec)] = np.zeros((self.num_regions,self.Nf), dtype=bool)
        self.detrended_signals[str(rec)] = np.zeros((self.num_regions,self.Nf))

        if str(rec) in self.F_list:
            for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):
                thres = self.detrending_param['rel_act_threshold'][i] * (np.max(self.sliding_window_vars[str(rec)][i,:]) - np.min(self.sliding_window_vars[str(rec)][i,:])) + np.min(self.sliding_window_vars[str(rec)][i,:])
                self.Lin_reg_masks[str(rec)][i,:] = np.logical_and(self.Movement_dict_subsampled_extended[str(rec)] == 0, self.sliding_window_vars[str(rec)][i,:] < thres)

                time_indices = np.arange(self.Nf)
                baseline_signal = copy.deepcopy(self.F_list[str(rec)][i,:])

                # Apply mask on data
                time_indices = [time_indices[self.Lin_reg_masks[str(rec)][i,:]==True]][0]
                baseline_signal = [baseline_signal[self.Lin_reg_masks[str(rec)][i,:]==True]][0]
                self.Lin_reg_masks[str(rec)][i,:]=~self.Lin_reg_masks[str(rec)][i,:]

                locs = np.where(self.Lin_reg_masks[str(rec)][i, :] == False)[0]
                if len(locs)>1:
                    max_dist = locs[-1] - locs[0]
                    if len(baseline_signal)>30 and max_dist>800:
                        res = linregress(time_indices,baseline_signal)
                        self.detrended_signals[str(rec)][i,:] = self.F_list[str(rec)][i,:] - (res.slope * np.arange(self.Nf) + res.intercept)


    def moving_average_subtract(self,rec,window_size):
        """
        For each ROI, apply sliding window and lower offset by subtracting smallest mean.
        :param rec: Recording entry in CSV table.
        :param window_size: Window in which the average is computed.
        :return: No return. Result stored in self.F_final.
        """
        self.F_final[str(rec)] = np.zeros((self.num_regions,self.Nf))

        for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):
            moving_avgs = np.empty((self.num_regions,self.Nf - window_size + 1))
            if not self.detrended_signals[str(rec)][i,:].any():
                signal = self.F_list[str(rec)][i,:]
            else:
                signal = self.detrended_signals[str(rec)][i,:]

            for k in range(self.Nf - window_size + 1):
                moving_avgs[:,k] = np.mean(signal[k:k+window_size])
                self.F_final[str(rec)][i,:] = signal - np.min(moving_avgs)

    # Plotting
    def plot_warp_sICA_comps(self,rec):
        for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):
            border = self.get_warp_border(rec,i)
            border_mask = np.ma.masked_where(border == 0, border)

            ID = self.MiceDict['recID'][rec]
            section = self.MiceDict['sectionID'][rec]
            sICA_component_number = self.MiceDict[region][rec]

            if sICA_component_number!='':
                try:
                    sICA_component_number = sICA_component_number.split(",")[0]
                    plt.title(region + " border of mask for recID " + str(ID) + ", section " + str(section) + ", sICA_comp " + str(sICA_component_number))
                    plt.imshow(self.sICA_image_dict[str(rec)][region])
                    plt.colorbar(fraction=0.039, pad=0.04)
                    plt.imshow(border_mask, cmap='nipy_spectral', interpolation='none',norm=colors.Normalize(vmin=self.ROI_color_list[0], vmax=self.ROI_color_list[-2]))
                    plt.show()
                except:
                    print("No valid number in CSV table")

    def plot_thresholded_sICA_comps(self,rec):
        for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):

            # threshold_mask = np.zeros((np.shape((self.sICA_image_dict[str(rec)][region]))))
            # threshold_mask[self.Thresholded_Mouse_ROI_pixels] = self.ROI_color_list[region]
            # threshold_mask = np.ma.masked_where(threshold_mask == 0, threshold_mask)

            ID = self.MiceDict['recID'][rec]
            section = self.MiceDict['sectionID'][rec]
            sICA_component_number = self.MiceDict[region][rec]

            if sICA_component_number!='':
                try:
                    sICA_component_number = sICA_component_number.split(",")[0]
                    threshold_mask = np.zeros((np.shape((self.sICA_image_dict[str(rec)][region]))))
                    pixels = np.array(self.Thresholded_Mouse_ROI_pixels[str(rec)][region])
                    threshold_mask[pixels[0],pixels[1]] = self.ROI_color_list[i]
                    threshold_mask = np.ma.masked_where(threshold_mask == 0, threshold_mask)

                    plt.title(region + " thresholded mask for recID " + str(ID) + ", section " + str(section) + ", sICA_comp " + str(sICA_component_number))
                    plt.imshow(self.sICA_image_dict[str(rec)][region])
                    plt.colorbar(fraction=0.039, pad=0.04)
                    plt.imshow(threshold_mask, cmap='nipy_spectral', interpolation='none',norm=colors.Normalize(vmin=self.ROI_color_list[0], vmax=self.ROI_color_list[-2]))
                    plt.show()
                except:
                    print("No valid number in CSV table")

    def plot_sICA_masks_combined(self,rec):
        for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):
            border = self.get_warp_border(rec,i)
            border_mask = np.ma.masked_where(border == 0, border)

            ID = self.MiceDict['recID'][rec]
            section = self.MiceDict['sectionID'][rec]

            # Find all integer numbers in the table
            list_ints = []
            list = self.MiceDict[region][rec].split(",")
            for item in list:
                try:
                    list_ints.append(int(item))
                except:
                    pass

            comp_string = ",".join(map(str,list_ints))

            if len(list_ints)!=0:
                try:
                    threshold_mask = np.zeros((np.shape((self.sICA_image_dict[str(rec)][region]))))
                    pixels = np.array(self.Thresholded_Mouse_ROI_pixels[str(rec)][region])
                    threshold_mask[pixels[0],pixels[1]] = self.ROI_color_list[i]
                    threshold_mask = np.ma.masked_where(threshold_mask == 0, threshold_mask)

                    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
                    fig.suptitle(region + " masks for recID " + str(ID) + ", section " + str(section) + ", sICA component " + comp_string)
                    cmap = matplotlib.cm.nipy_spectral
                    ax[0].imshow(self.sICA_image_dict[str(rec)][region],vmin=-1, vmax=1,cmap='viridis')
                    ax[0].imshow(border_mask, cmap=cmap, interpolation='none',norm=colors.Normalize(vmin=0, vmax=255))
                    ax[0].set(adjustable='box', aspect='auto')
                    ax[0].set_title('Border of anatomical area')

                    im = ax[1].imshow(self.sICA_image_dict[str(rec)][region],vmin=-1, vmax=1,cmap='viridis')
                    ax[1].imshow(threshold_mask, cmap=cmap, interpolation='none',norm=colors.Normalize(vmin=0, vmax=255))
                    ax[1].set(adjustable='box', aspect='auto')
                    ax[1].set_title('Thresholded mask of anatomical area')
                    divider = make_axes_locatable(ax[1])
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    plt.colorbar(im, cax=cax)

                    plt.tight_layout()
                    fig_name = self.PreProcessingName+ '_TE_' + str(rec) + '_' + region + '_threshold_' + str(self.sICA_threshold)
                    pre_fig_path_entry = os.path.join(*[self.paths['fig_pre_path'], 'TE_' + str(rec)])
                    if not os.path.exists(pre_fig_path_entry):
                        os.makedirs(pre_fig_path_entry)
                    plt.savefig(pre_fig_path_entry + '/' + fig_name + '.pdf', format='pdf',dpi=200)
                    print("Saving image")
                    plt.show()

                except:
                    print("No valid number in CSV table")

    def plot_time_courses(self,rec):
        if str(rec) in self.F_final:
            T_res = self.param['T_res']
            samples = self.Nf
            try:

                ID = self.MiceDict['recID'][rec]
                section = self.MiceDict['sectionID'][rec]

                fig, ax = plt.subplots(self.num_regions,1, figsize=(15, 9), sharex=True)
                fig.suptitle(" Time courses for recID " + str(ID) + ", section " + str(section))
                fig.supylabel("Amplitude")
                fig.supxlabel("Time [s]")
                cmap = matplotlib.cm.get_cmap("nipy_spectral").copy()
                time = np.linspace(0, int(samples * T_res) - T_res, samples)

                color_num = 250
                movement_mask = np.ma.masked_where(self.Movement_dict[str(rec)] == 0, self.Movement_dict[str(rec)])
                cmap.set_bad(color='white')

                for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):
                    margin = 0.01
                    min=np.min(self.F_final[str(rec)][i,:])-margin
                    max=np.max(self.F_final[str(rec)][i,:])+margin

                    if region=='Anterior singulate':
                        region='Anterior cingulate'

                    ax[i].plot(time,self.F_final[str(rec)][i,:], color=cmap(self.ROI_color_list[i]))
                    ax[i].set(adjustable='box', aspect='auto')
                    ax[i].set_title( region + ' area')
                    ax[i].grid(True)
                    ax[i].set_xlim([0,time[-1]])
                    ax_2 = ax[i].twiny()
                    ax_2.imshow(color_num * movement_mask[None, 100:], aspect="auto", cmap=cmap,
                                norm=colors.Normalize(vmin=0, vmax=255), interpolation='None',
                                extent=[0, samples, min, max])
                    ax_2.set_xticks([])
                    ax[i].set_zorder(1)  # default zorder is 0 for ax1 and ax2
                    ax[i].patch.set_visible(False)  # prevents ax1 from hiding ax2


                plt.tight_layout()
                fig_name = self.PreProcessingName+ '_TE_' + str(rec) + '_Timecourses_threshold_' + str(self.sICA_threshold)
                pre_fig_path_entry = os.path.join(*[self.paths['fig_pre_path'], 'TE_' + str(rec)])
                if not os.path.exists(pre_fig_path_entry):
                    os.makedirs(pre_fig_path_entry)
                plt.savefig(pre_fig_path_entry + '/' + fig_name + '.pdf', format='pdf')
                print("Saving image")
                plt.show()
            except:
                pass

    def plot_filled_warp(self,rec):

        if str(rec) in self.F_list:
            ID = self.MiceDict['recID'][rec]
            path_npy = os.path.join(self.paths['warp_path'], str(ID) + '_warp.npy')
            warp = np.load(path_npy)
            cmap = matplotlib.cm.get_cmap("nipy_spectral").copy()

            plt.figure(figsize=(10,8))
            plt.imshow(warp,aspect="auto",cmap=cmap,norm=colors.Normalize(vmin=0, vmax=255), interpolation='None')
            plt.title("Filled brain warp of recID "+ str(ID))
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            fig_name = self.PreProcessingName+ '_TE_' + str(rec) + '_FilledWarp_recID_' + str(ID)
            pre_fig_path_entry = os.path.join(*[self.paths['fig_pre_path'], 'TE_' + str(rec)])
            if not os.path.exists(pre_fig_path_entry):
                os.makedirs(pre_fig_path_entry)
            plt.savefig(pre_fig_path_entry + '/' + fig_name + '.pdf', format='pdf')
            plt.show()

    def plot_original_vs_detrended_signal(self,rec):

        if str(rec) in self.F_list:
            for region, i in zip(self.ROI_list[:-2], range(len(self.ROI_list[:-2]))):
                ID = self.MiceDict['recID'][rec]
                section = self.MiceDict['sectionID'][rec]
                T_res = self.param['T_res']
                samples = self.Nf
                time = np.linspace(0, int(samples * T_res) - T_res, samples)
                cmap = matplotlib.cm.get_cmap("nipy_spectral").copy()
                color_num = 250

                fig, ax = plt.subplots(2, 1, figsize=(13, 7),gridspec_kw={'height_ratios': [3,1]},sharex=True)
                fig.suptitle('Detrending of fUS time course from recID '+ str(ID) + ', section ' + section + ', '+ region + ' area')
                fig.supxlabel('Time [s]')
                activity_mask = np.ma.masked_where(self.Lin_reg_masks[str(rec)][i,:] == 0, self.Lin_reg_masks[str(rec)][i,:])
                movement_mask = np.ma.masked_where(self.Movement_dict[str(rec)] == 0, self.Movement_dict[str(rec)])
                cmap.set_bad(color='white')

                ax[0].plot(time,self.F_list[str(rec)][i,:], color=cmap(self.ROI_color_list[i]),label="Original")
                if self.detrended_signals[str(rec)][i,:].any():
                    ax[0].plot(time,self.F_list[str(rec)][i,:]-self.detrended_signals[str(rec)][i,:],label="Linear regression",color=cmap(220),linewidth=2.5)
                    ax[0].plot(time,self.detrended_signals[str(rec)][i,:],label="Detrended",color=cmap(80))
                ax[0].set_xlim([0,time[-1]])
                ax[0].set_ylabel("Amplitude")
                ax[0].legend()
                ax_2 = ax[0].twiny()
                ax_2.imshow(color_num * movement_mask[None, 100:], aspect="auto", cmap=cmap,
                            norm=colors.Normalize(vmin=0, vmax=255), interpolation='None',
                            extent=[0, samples, np.min(self.detrended_signals[str(rec)][i,:])-10e-9, np.max(self.F_list[str(rec)][i,:])+10e-9])
                ax_2.set_xticks([])
                ax[0].set_zorder(1)  # default zorder is 0 for ax1 and ax2
                ax[0].patch.set_visible(False)  # prevents ax1 from hiding ax2
                ax[0].yaxis.set_ticks_position('both')

                ax[1].set_title("Activity mask for linear regression")
                ax[1].imshow(0 * activity_mask[None,:], aspect="auto", cmap=cmap,
                                        norm=colors.Normalize(vmin=0, vmax=255), interpolation='None',
                                        extent=[0, time[-1], 0, 1])

                ax[1].set_yticks([])
                plt.tight_layout()
                fig_name = self.PreProcessingName + '_Detrend' + '_TE_' + str(rec) + '_' + region + '_activitythreshold_' + str(self.detrending_param['rel_act_threshold'][i]) + '_varwindow_' + str(self.detrending_param['variance_window_size'])
                pre_fig_path_entry = os.path.join(*[self.paths['fig_pre_path'], 'TE_' + str(rec)])
                if not os.path.exists(pre_fig_path_entry):
                    os.makedirs(pre_fig_path_entry)
                plt.savefig(pre_fig_path_entry + '/' + fig_name + '.pdf', format='pdf')
                print("Saving image")
                plt.show()


    # Importing/exporting

    def dict_nparray_to_list(self):

        for rec in range(self.num_recs):
            for region in self.ROI_list:
                try:
                    self.Mouse_ROI_pixels[str(rec)][region] = np.array(self.Mouse_ROI_pixels[str(rec)][region]).tolist()
                    self.Thresholded_Mouse_ROI_pixels[str(rec)][region] = np.array(
                        self.Thresholded_Mouse_ROI_pixels[str(rec)][region]).tolist()
                except:
                    pass
                try:
                    self.sICA_image_dict[str(rec)][region] = self.sICA_image_dict[str(rec)][region].tolist()
                except:
                    pass
            try:
                self.F_list[str(rec)] = self.F_list[str(rec)].tolist()
                self.F_final[str(rec)] = self.F_final[str(rec)].tolist()
                self.Lin_reg_masks[str(rec)] = self.Lin_reg_masks[str(rec)].tolist()
                self.detrended_signals[str(rec)] = self.detrended_signals[str(rec)].tolist()
                self.Movement_dict[str(rec)] = self.Movement_dict[str(rec)].tolist()
                self.Movement_dict_subsampled[str(rec)] = self.Movement_dict_subsampled[str(rec)].tolist()
            except:
                pass

    def list_to_dict_nparray(self):
        for rec in range(self.num_recs):
            for region in self.ROI_list:
                try:
                    self.Mouse_ROI_pixels[str(rec)][region] = (np.array(self.Mouse_ROI_pixels[str(rec)][region])[0],np.array(self.Mouse_ROI_pixels[str(rec)][region])[1])
                    self.Thresholded_Mouse_ROI_pixels[str(rec)][region] = (np.array(self.Thresholded_Mouse_ROI_pixels[str(rec)][region])[0],np.array(self.Thresholded_Mouse_ROI_pixels[str(rec)][region])[1])
                except:
                    pass
                try:
                    self.sICA_image_dict[str(rec)][region] = np.array(self.sICA_image_dict[str(rec)][region])
                except:
                    pass
            try:
                self.F_list[str(rec)] = np.array(self.F_list[str(rec)])
                self.F_final[str(rec)] = np.array(self.F_final[str(rec)])
                self.detrended_signals[str(rec)] = np.array(self.detrended_signals[str(rec)])
                self.Lin_reg_masks[str(rec)] = np.array(self.Lin_reg_masks[str(rec)])
                self.Movement_dict[str(rec)] = np.array(self.Movement_dict[str(rec)])
                self.Movement_dict_subsampled[str(rec)] = np.array(self.Movement_dict_subsampled[str(rec)])
            except:
                pass

    def import_json(self):
        # Import only parameters necessary to initialize script
        with open(self.paths['json_pre_path']) as json_file:
            data = json.load(json_file)

        self.PreProcessingName = data['PreProcessingName']
        self.param = data['param']
        self.sICA_RUN = data['sICA']['sICA_RUN']
        self.seed_num = data['sICA']['seed_num']
        self.detrending_param = data['detrending_param']

        self.num_recs = self.param['num_recs']
        self.num_regions = self.param['num_regions']
        self.baseline_subtraction_window_size = self.param['baseline_subtraction_window_size']
        self.sICA_threshold = self.param['sICA_threshold']
        self.Fs = self.param['Fs']
        self.Nf = self.param['Nf']

        self.F_list = data['F_list']
        self.F_final = data['F_final']

        self.Lin_reg_masks = data['Lin_reg_masks']
        self.detrended_signals = data['detrended_signals']

        self.Movement_dict = data['Movement_dict']
        self.Movement_dict_subsampled = data['Movement_dict_subsampled']
        self.MiceDict = data['MiceDict']
        self.ROI_list = data['ROI_list']
        self.ROI_color_list = data['ROI_color_list']
        self.Mouse_ROI_pixels = data['Mouse_ROI_pixels']
        self.Thresholded_Mouse_ROI_pixels = data['Thresholded_Mouse_ROI_pixels']
        self.sICA_image_dict = data['sICA_image_dict']

        self.warps = np.array(data['warps'])

        self.list_to_dict_nparray()

    def export_json(self):
        self.dict_nparray_to_list()

        export_dict = {}
        export_dict['PreProcessingName'] = self.PreProcessingName
        export_dict['param'] = self.param
        export_dict['sICA'] = {}
        export_dict['sICA']['sICA_RUN'] = self.sICA_RUN
        export_dict['sICA']['seed_num'] = self.seed_num
        export_dict['detrending_param'] = self.detrending_param

        export_dict['F_list'] = self.F_list
        export_dict['F_final'] = self.F_final

        export_dict['Lin_reg_masks'] = self.Lin_reg_masks
        export_dict['detrended_signals'] = self.detrended_signals

        export_dict['Movement_dict'] = self.Movement_dict
        export_dict['Movement_dict_subsampled'] = self.Movement_dict_subsampled
        export_dict['MiceDict'] = self.MiceDict
        export_dict['ROI_list'] = self.ROI_list
        export_dict['ROI_color_list'] = self.ROI_color_list
        export_dict['Mouse_ROI_pixels'] = self.Mouse_ROI_pixels
        export_dict['Thresholded_Mouse_ROI_pixels'] = self.Thresholded_Mouse_ROI_pixels
        export_dict['sICA_image_dict'] = self.sICA_image_dict
        export_dict['warps'] = self.warps.tolist()

        with open(self.paths['json_pre_path'], 'w') as json_path:
            json.dump(export_dict, json_path, indent=4)

if __name__ == "__main__":
    # Give PreProcessingName name
    simulation_ID = 41

    # sICA info for pre-processing
    sICA_RUN = 7
    seed_num = 0

    overwrite = False # Overwrite PRE_x.json
    specific_rec = [] # Enter entry for testing purpose (only analyses the specific entry)

    PreProcessingName = "PRE_TEST" if len(specific_rec)!=0 else "PRE_" + str(simulation_ID)

    # Execute pre-processing steps (twice to first store and subsequently load the results stored)
    PreProcessing(PreProcessingName, sICA_RUN=sICA_RUN, seed_num=seed_num,overwrite=overwrite,specific_rec=specific_rec)
    f = PreProcessing(PreProcessingName)

    # Plot most important results
    recs = range(f.num_recs) if len(specific_rec) == 0 else specific_rec

    for rec in recs:
        f.plot_filled_warp(rec)
        f.plot_sICA_masks_combined(rec)
        f.plot_original_vs_detrended_signal(rec)
        f.plot_time_courses(rec)
        pass
