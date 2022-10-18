import copy
import numpy as np
import pandas as pd
import matplotlib
import os
import util
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interp1d

class MovementExtraction(object):

    def __init__(self,TE):
        self.TE = TE

        # Read configurations.ini for accessing fUS server
        self.config = util.read_config_file()

        # Create paths
        self.paths = util.create_paths_dict()

        # Read mice CSV table
        self.MiceDict = util.read_csv_file(self.paths['csv_path'])

        ID = self.MiceDict['recID'][self.TE]
        sec = self.MiceDict['sectionID'][self.TE]
        name = self.MiceDict['animalID'][self.TE]

        self.fps = int(self.config['MotionVideo']['fps'])
        self.Nf = int(self.config['MotionVideo']['Nf_video'])

        # Path to file
        remainder = "BodycamDLC_resnet50_bodycam_bottomJan22shuffle1_1030000.h5"
        self.file_path = os.path.join(self.paths['movement_path'],name,str(ID),str(sec),remainder)

        #labels
        labs = ["left_front", "right_front", "left_hind_paw", "right_hind_paw"]

        if os.path.exists(self.file_path):
            self.df = self.dlc_data_to_dataframe(self.file_path)
            self.labels = self.bodyparts_list(self.df, keyword=labs)
            self.time = self.frames_array(self.df, fps=self.fps)

            self.df_lls = self.extract_dlc_likelihood(self.file_path)

            self.xy_arrays={}
            self.df_lls_thresholded = {}
            for label in self.labels:
                self.df_lls_thresholded[label] = self.get_likelihood_labelarray(self.df_lls, label)
                self.xy_arrays[label] = self.raw_label_array(self.df, label)

            print("Shape of xy_arrays",np.shape(self.xy_arrays[self.labels[0]]))

            # Apply sliding window and thresholding based on variance within window
            self.sliding_window_variance()
            self.threshold_variance()
            self.subsample()

            #
            # self.plot_xy()
            # self.plot_variances()
            # self.plot_movement()

        else:
            chunk_times = [0, 91202, 182404, 273606, 364808, 456010, 547213, 638415]
            self.time = np.linspace(0, (self.Nf-1)/self.fps, self.Nf)
            self.xy_arrays={}
            self.df_lls_thresholded = {}

            for label in labs:
                self.xy_arrays[label] = []
                self.df_lls_thresholded[label] = []

            for chunk in range(1,len(chunk_times)+1):
                time = str(chunk_times[chunk - 1])
                remainder = "msDLC_resnet50_bodycam_bottomJan22shuffle1_1030000.h5"
                self.file_path = os.path.join(self.paths['movement_path'],name,str(ID),str(sec),"Bodycam_chunk" + str(chunk) + "," + time + remainder)

                df = self.dlc_data_to_dataframe(self.file_path)
                self.labels = self.bodyparts_list(df, keyword=labs)
                self.df_lls = self.extract_dlc_likelihood(self.file_path)

                for label, i in zip(self.labels, range(len(self.labels))):
                    if chunk==1:
                        self.xy_arrays[label].extend(self.raw_label_array(df, label).tolist())
                        self.df_lls_thresholded[label].extend(self.get_likelihood_labelarray(self.df_lls, label).tolist())
                    else:
                        raw = self.raw_label_array(df, label)
                        self.xy_arrays[label][0].extend(raw[0].tolist())
                        self.xy_arrays[label][1].extend(raw[1].tolist())
                        ll=self.get_likelihood_labelarray(self.df_lls, label)
                        self.df_lls_thresholded[label].extend(ll.tolist())
                        #self.df_lls_thresholded[label].extend(ll[1].tolist())

            for label in self.labels:
                self.xy_arrays[label] = np.array(self.xy_arrays[label])
                self.df_lls_thresholded[label] = np.array(self.df_lls_thresholded[label])

            self.sliding_window_variance()
            self.threshold_variance()
            self.subsample()

            # self.plot_variances()
            # self.plot_movement()



    def dlc_data_to_dataframe(self,file):
        """
        Import position label data from deeplabcut into pandas dataframe
        ---------------------------------------------------------
        :param file: dlc position data of the labels (hdf5)
        :return: pandas data frame with label positions
        """
        df = pd.read_hdf(file)
        df.drop('likelihood', axis=1, level='coords', inplace=True) # remove likelihood measure
        return df

    def bodyparts_list(self, df, keyword=None):
        """
        List object of all the DLC tracked features/labels.
        There is an option to supply keyword so you can select a subset
        :param dataframe: dlc dataframe with label positions
        :param keyword: can be a list of strings or just one string. Return only specified labels as list
        :return: list of dlc labels
        """
        labels = list(set(list(df.columns.get_level_values('bodyparts'))))
        if keyword is not None:
            label_list = keyword
        else:
            label_list = labels
        return label_list

    def frames_array(self,df, fps=None):
        """
        Create time vector (s) based on fps
        ----------------------------------
        :param dataframe:
        :param fps: frames per second from video acquisition
        :return:
        """
        time = 1
        if isinstance(fps, int): # check if input fps is integer
            time=fps
        return np.array(list(df.index))/time

    def dataframe_per_bodypart(self,df, bodypart):
        """
        Dataframe per bodypart
        -------------------------
        :param df:
        :param bodypart: string of bodypart dlc label
        :return:
        """
        scorer = list(df.columns.get_level_values ('scorer'))[0]
        df_bodypart = df.xs(bodypart, level='bodyparts', axis=1)
        df_bodypart = df_bodypart.xs(scorer, level='scorer', axis=1)
        return df_bodypart

    def bodypart_array(self,df_bodypart, pos='x'):
        return np.array (df_bodypart[pos].values)

    def extract_dlc_likelihood(self,file):
        df_likely = pd.read_hdf (file)
        df_likely.drop(['x', 'y'], axis=1, level='coords', inplace=True)
        scorer = list(df_likely.columns.get_level_values('scorer'))[0]
        coords = list(df_likely.columns.get_level_values('coords'))[0]
        df_likely = df_likely.xs(coords, level='coords', axis=1)
        df_likely = df_likely.xs(scorer, level='scorer', axis=1)
        return df_likely

    def get_likelihood_labelarray(self,df,label,thresh = 0.99):
        return np.array(df[label].tolist()) > thresh

    def raw_label_array(self,df,label):
        """
        Raw DLC label (specfied by name) coordinates
        """
        # raw video x,y pixel coordinates
        x = np.array(self.dataframe_per_bodypart(df, label)["x"].values)
        y = np.array(self.dataframe_per_bodypart(df, label)["y"].values)

        return np.array([x,y])

    def sliding_window_variance(self,window_size=1):
        """
        Compute variances inside a sliding window.
        :param window_size: Size of window [samples].
        :return: Stored in self.variances
        """
        window_size = window_size * self.fps  # in samples now
        self.moving_stds = {}

        for label in self.labels:
            self.moving_stds[label]=np.zeros((2,self.Nf))
            for frame in range(self.Nf):
                xy_windowed = [[], []]
                for i in range(int(-window_size / 2), int(window_size / 2)+1):
                    if (frame+i>=0 and frame+i<self.Nf):
                        if self.df_lls_thresholded[label][frame+i] == True: # Only use reliable data points
                            xy_windowed[0].append(self.xy_arrays[label][0,frame+i])
                            xy_windowed[1].append(self.xy_arrays[label][1,frame+i])
                if np.size(xy_windowed,axis=1) > 1:
                    try:
                        self.moving_stds[label][:,frame] = np.nanvar(xy_windowed, axis=1, ddof=1)
                    except:
                        pass
                else:
                    self.moving_stds[label][:,frame] = np.nan

        self.variances = copy.deepcopy(self.moving_stds)

    def threshold_variance(self,threshold_percentage=0.01):
        for label in self.labels:
            thresholds = threshold_percentage * np.nanmax(self.moving_stds[label],axis=1)[:,None]

            for xy in range(2):
                mask_nan = ~np.isnan(self.moving_stds[label][xy, :]) == True
                mask_threshold = self.moving_stds[label][xy,:] >= thresholds[xy,0]
                mask=mask_nan*mask_threshold
                self.moving_stds[label][xy,mask] = True
                self.moving_stds[label][xy,~mask] = False

        self.movement = np.zeros(self.Nf)
        for label in self.labels:
            self.movement = np.nanmax(np.concatenate(( self.movement[None,:],self.moving_stds[label][0,:][None,:],self.moving_stds[label][1,:][None,:]),axis=0),axis=0)

    def subsample(self):
        movement = self.movement[99:len(self.movement) - int(np.floor(50/4)-1)]
        lin = interp1d(np.arange(len(movement)), movement, kind='nearest')
        self.movement_sub_sampled = lin(np.arange(0, len(movement), self.fps/4))

    # Plotting
    def plot_xy(self):
        # perform extraction of raw coordinates over every label
        for label in self.labels:
            # plot y-information over time in scatterplot
            fig, axs = plt.subplots(2, sharex=True)
            fig.suptitle('X and y positions of paws ' + label)
            fig.supxlabel('Time [s]')
            axs[0].plot(self.time, self.xy_arrays[label][0,:], label=label)
            axs[0].set_ylabel("x")
            axs[1].plot(self.time, self.xy_arrays[label][1,:], label=label)
            axs[1].set_ylabel("y")

        plt.show()

    def plot_lls_mask(self):
        cmap = matplotlib.cm.get_cmap("nipy_spectral").copy()
        color_num = 252

        fig, axs = plt.subplots(len(self.labels), sharex=True)
        for label,i in zip(self.labels,range(len(self.labels))):
            lls_mask = np.ma.masked_where(self.df_lls_thresholded[label] == 0, self.df_lls_thresholded[label])
            cmap.set_bad(color='white')
            axs[i].imshow(color_num * lls_mask[None,:],aspect="auto",cmap=cmap,norm=colors.Normalize(vmin=0, vmax=255),interpolation='None')
        plt.show()

    def plot_variances(self):
        cmap = matplotlib.cm.get_cmap("nipy_spectral").copy()
        color_num = 252

        fig, axs = plt.subplots(len(self.labels), sharex=True)
        for label,i in zip(self.labels,range(len(self.labels))):
            lls_mask = np.ma.masked_where(self.df_lls_thresholded[label] == 0, self.df_lls_thresholded[label])
            cmap.set_bad(color='white')
            min = np.nanmin(self.variances[label][0,:])
            max = np.nanmax(self.variances[label][0,:])
            axs[i].imshow(color_num * lls_mask[None,:],aspect="auto",cmap=cmap,norm=colors.Normalize(vmin=0, vmax=255),extent=[0,self.Nf,min,max],interpolation='None')
            axs[i].plot(self.variances[label][0,:])
        plt.show()

    def plot_movement(self):
        cmap = matplotlib.cm.get_cmap("nipy_spectral").copy()
        color_num = 240
        movement_mask = np.ma.masked_where(self.movement == 0, self.movement)
        cmap.set_bad(color='white')

        plt.imshow(color_num * movement_mask[None,:],aspect="auto",cmap=cmap,norm=colors.Normalize(vmin=0, vmax=255),interpolation='None')
        plt.title("Final extracted movement")
        plt.show()




if __name__ == "__main__":
    TE = 6
    m=MovementExtraction(TE)
    m.subsample()

    # m.plot_variances()
    # m.plot_movement()

    # Create table with movements
    # num_recs = 15
    # mov_time_bauds = np.zeros((num_recs,2))
    # for rec in range(num_recs):
    #     m = MovementExtraction(rec)
    #     mov_time_bauds[rec,0] = np.count_nonzero(m.movement == 1) * 0.02 # Total movement time
    #     mov_time_bauds[rec,1] = (np.diff(m.movement) != 0).sum()/2
    #
    # mov_time_bauds = np.around(mov_time_bauds,2)
    # mov_time_bauds[:,1] = np.ceil(mov_time_bauds[:,1])

