import copy
import numpy as np
import os
import cv2
import util
from scipy.ndimage import binary_dilation,binary_erosion,generate_binary_structure
from data_import import Data
import matplotlib
import plotly.express as px

class GUI_Data(object):
    def __init__(self,TE,ndpi=-1,time_course_data=False):
        if TE == "Dummy":
            config = util.read_config_file()
            self.refresh_time = 0.1
            self.Nf = 100
            self.width = int(config['BorderFromCentre']['left_border']) + int(
                config['BorderFromCentre']['right_border'])
            self.heigth = int(config['BorderFromCentre']['lower_border']) + int(
                config['BorderFromCentre']['upper_border'])
            self.c_min = 0
            self.c_max = 1
            if time_course_data:
                self.GUI_data = np.ones((1, self.heigth, self.width))
            else:
                self.GUI_data = np.ones((1,self.heigth,self.width)).tolist()
            self.x_list = [0]
            self.mean_log_fig = np.ones((self.heigth,self.width)).tolist()

        else:
            paths = util.create_paths_dict()
            MiceDict = util.read_csv_file(paths['csv_path'])
            ID = MiceDict['recID'][TE]
            sectionID = MiceDict['sectionID'][TE]
            obj = Data(ID, sectionID, ndpi)
            obj.subtract_mean_spatial(mask=True,mask_threshold=float(MiceDict['mask_threshold'][TE])) # Linked to CSV table

            if time_course_data:
                self.GUI_data = obj.PDI
            else:
                self.GUI_data = obj.load_GUI_data()

            self.Nf = obj.Nf
            self.refresh_time = int(1/obj.scan_parameters['Fs'] * 1000)
            self.c_min = np.log10(np.min(self.GUI_data))
            self.c_max = np.log10(np.max(self.GUI_data))
            self.x_list = [i for i in range(0,obj.Nf)]
            self.mean_log_fig = np.log10(1 - np.min(obj.mean_fig()) + obj.mean_fig())


class GUI_Warp(object):
    def __init__(self,TE):
        if TE == "Dummy":
            self.norm_img = np.zeros((10,10))
            self.ROI_list = ["No region"]
            self.flood_fill_color_table = 255
            cmap = matplotlib.cm.nipy_spectral
            self.color_list = []
            for i in range(cmap.N):
                rgba = cmap(i)
                # rgb2hex accepts rgb or rgba
                self.color_list.append(matplotlib.colors.rgb2hex(rgba))
        else:
            config = util.read_config_file()
            paths = util.create_paths_dict()
            MiceDict = util.read_csv_file(paths['csv_path'])
            self.ID = MiceDict['recID'][TE]

            self.num_regions = config['Analysis']['num_regions']
            cmap = matplotlib.cm.nipy_spectral
            self.color_list = []
            for i in range(cmap.N):
                rgba = cmap(i)
                # rgb2hex accepts rgb or rgba
                self.color_list.append(matplotlib.colors.rgb2hex(rgba))

            # IMPORT WARP
            self.norm_img = import_warp(self.ID)
            self.ROI_list, self.flood_fill_color_table = util.create_region_color_list()


def import_warp(ID):
    # Import warp
    p = util.create_paths_dict(ID=ID)

    if os.path.exists(p['warp_npy_path']):
        norm_img = np.load(p['warp_npy_path'])
    elif os.path.exists(p['warp_tif_path']):
        img = cv2.imread(p['warp_tif_path'], cv2.IMREAD_GRAYSCALE)
        norm_img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        # Thresholding
        threshold = 254
        norm_img[norm_img < threshold] = 0
        norm_img[norm_img >= threshold] = 255
    elif os.path.exists(p['warp_jpg_path']):
        img = cv2.imread(p['warp_jpg_path'], cv2.IMREAD_GRAYSCALE)
        norm_img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        # Thresholding
        threshold = 254
        norm_img[norm_img < threshold] = 0
        norm_img[norm_img >= threshold] = 255
    else:
        raise ValueError("Path to warp image does not exist")

    return norm_img

def floodfill(img,xy,color):
    y = xy['points'][0]['y']
    x = xy['points'][0]['x']
    seed_point = x,y
    cv2.floodFill(img, None, seed_point, color)

    return img

def dilation(img,roi_color):
    binary_img = copy.deepcopy(img)
    binary_img[img == roi_color] = True
    binary_img[img != roi_color] = False
    struct = generate_binary_structure(2,2)
    dilated_img = binary_dilation(binary_img,structure=struct).astype(binary_img.dtype)
    img[dilated_img==True]=roi_color

    return img

def erosion(img,roi_color):
    binary_img = copy.deepcopy(img)
    binary_img[img == roi_color] = True
    binary_img[img != roi_color] = False
    struct = generate_binary_structure(2, 2)
    erosed_img = binary_erosion(binary_img, structure=struct).astype(binary_img.dtype)
    difference = binary_img-erosed_img
    img[difference == True] = 0 # Give erosed pixels a border color

    return img

def save_ROI_image(img,ID,num_regions):
    paths = util.create_paths_dict()
    path = os.path.join(paths['warp_path'], str(ID),'_warp')
    np.save(path,img)

def update_colors(img):
    _, color_list = util.create_region_color_list()
    warp_colors = np.unique(img[(img != 0) & (img != 255)])
    # # Check if warp is consistent with colors used. If not, update warp with colors in increasing order
    for i in range(len(warp_colors)):
        try:
            img[img==warp_colors[i]] = color_list[i]
        except:
            pass
    return img

def create_option_list():
    option_list = []
    paths = util.create_paths_dict()
    MiceDict = util.read_csv_file(paths['csv_path'])
    for TE in range(0,len(MiceDict['recID'])):
        p = util.create_paths_dict(ID=MiceDict['recID'][TE],slice=MiceDict['sectionID'][TE])
        if os.path.exists(p['srv_pdi_path']) or os.path.exists(p['loc_pdi_path']):
            option_list.append({"label":"TE-"+ str(TE) + " ID-" +MiceDict['recID'][TE] + " SEC-"+MiceDict['sectionID'][TE],"value":str(TE)})
        else:
            option_list.append({"label":"TE-"+ str(TE) + " ID-" +MiceDict['recID'][TE] + " SEC-"+MiceDict['sectionID'][TE],"value":str(TE),"disabled":True})

    return option_list
