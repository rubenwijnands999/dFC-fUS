import configparser
import csv
import os
import ast
import seaborn as sns


# Read config file settings
def read_config_file():
    config = configparser.ConfigParser()
    config.read('configurations.ini')
    return config

# Read CSV file containing mice information
def read_csv_file(path):
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=';')
        keys = reader.fieldnames

        MiceDict = {}
        for key in keys:
            MiceDict[key]=[]
        for row in reader:
            for key in keys:
                MiceDict[key].append(row[key])

        # # Reduce MiceDict and create mapping to original sequence
        # Mapping = []
        # config=read_config_file()
        # for TE in range(len(MiceDict['recID'])):
        #     for region in list(MiceDict.keys())[7:7+int(config['Analysis']['num_regions'])]:
        #         if len(MiceDict[region])==0:
        #             delete_indices.append()
        #             MiceDict
        #             break
        #         Mapping.append(TE)
        # for key in MiceDict:
        #     remove entry



    return MiceDict

def create_paths_dict(ID=None,slice=None,sICA_ID=None,TableEntry=None,PreProcessingName=None,MainProcessingName=None,SimulationName=None):
    config = read_config_file()

    p = {}
    p['main_path'] = os.path.dirname(os.path.realpath(__file__))
    p['data_path'] = os.path.join(p['main_path'], 'data')
    p['csv_path'] = os.path.join(p['data_path'], config['CSV']['file_name'])

    # Figure paths
    p['fig_path'] = os.path.join(p['main_path'], 'figures')
    p['fig_sICA_path'] = os.path.join(p['fig_path'], 'sICA')
    p['fig_pre_path'] = os.path.join(p['fig_path'], 'Pre-processing')
    p['fig_main_path'] = os.path.join(p['fig_path'], 'Main-processing')
    p['fig_main_full_path'] = os.path.join(p['fig_main_path'], 'Full_group')
    p['fig_main_sep_path'] = os.path.join(p['fig_main_path'], 'Separate_groups')

    if SimulationName is not None:
        p['fig_sim_ID_path'] = os.path.join(p['fig_path'], 'SIM', SimulationName)
        if not os.path.exists(p['fig_sim_ID_path']): os.makedirs(p['fig_sim_ID_path'])

    if (sICA_ID is not None) and (TableEntry is not None):
        p['fig_sICA_run_path'] = os.path.join(p['fig_sICA_path'], 'RUN_' + str(sICA_ID))
        p['fig_sICA_entry_path'] = os.path.join(p['fig_sICA_run_path'], 'TE_' + str(TableEntry))
        p['fig_sICA_entry_time_course_path'] = os.path.join(p['fig_sICA_entry_path'], 'TimeCourses')
        if not os.path.exists(p['fig_sICA_run_path']): os.makedirs(p['fig_sICA_run_path'])
        if not os.path.exists(p['fig_sICA_entry_path']): os.makedirs(p['fig_sICA_entry_path'])
        if not os.path.exists(p['fig_sICA_entry_time_course_path']): os.makedirs(p['fig_sICA_entry_time_course_path'])


    # Warp path
    p['warp_path'] = os.path.join(p['data_path'],'Warped_brains',config['Analysis']['num_regions']+'_regions')
    if ID is not None:
        p['warp_npy_path'] = os.path.join(p['warp_path'], str(ID) + '_warp.npy')
        p['warp_tif_path'] = os.path.join(p['data_path'],'Warped_brains', str(ID) + '_warp.tif')
        p['warp_jpg_path'] = os.path.join(p['data_path'],'Warped_brains', str(ID) + '_warp.jpg')

    # Movement video path
    p['movement_path'] = os.path.join(p['data_path'], 'Movement-extraction')

    # Book-keeping paths (including JSON paths)
    p['data_pre_path'] = os.path.join(p['data_path'], 'Pre-processing')
    p['data_main_path'] = os.path.join(p['data_path'], 'Main-processing')
    if PreProcessingName is not None:
        p['json_pre_path'] = os.path.join(p['data_pre_path'], '{}.json'.format(PreProcessingName))
    if MainProcessingName is not None:
        p['json_main_path'] = os.path.join(p['data_main_path'], '{}.json'.format(MainProcessingName))

    if (ID is not None) and (slice is not None):
        # Local data import paths
        p['loc_param_path'] = os.path.join(p['data_path'],str(ID)+"-"+str(slice),"ScanParameters.mat")
        p['loc_pdi_path'] = os.path.join(p['data_path'],str(ID)+"-"+str(slice),"power_doppler_images.dat")
        # Server data import
        p['srv_param_path'] = os.path.join(config["DataImportSettings"]['mount_folder'],config["DataImportSettings"]['username'],str(ID),"ScanParameters.mat")
        p['srv_pdi_path'] = os.path.join(config["DataImportSettings"]['mount_folder'],config["DataImportSettings"]['username'],str(ID),str(slice),"ultrasound","power_doppler_images.dat")

    # If a path to a folder does not exist, then create.
    if not os.path.exists(p['data_path']): os.makedirs(p['data_path'])
    if not os.path.exists(p['fig_path']): os.makedirs(p['fig_path'])
    if not os.path.exists(p['fig_sICA_path']): os.makedirs(p['fig_sICA_path'])
    if not os.path.exists(p['fig_pre_path']): os.makedirs(p['fig_pre_path'])
    if not os.path.exists(p['fig_main_path']): os.makedirs(p['fig_main_path'])
    if not os.path.exists(p['fig_main_full_path']): os.makedirs(p['fig_main_full_path'])
    if not os.path.exists(p['fig_main_sep_path']): os.makedirs(p['fig_main_sep_path'])

    if not os.path.exists(p['warp_path']): os.makedirs(p['warp_path'])
    if not os.path.exists(p['data_pre_path']): os.makedirs(p['data_pre_path'])
    if not os.path.exists(p['data_main_path']): os.makedirs(p['data_main_path'])

    return p

def create_region_color_list():
    config = read_config_file()
    paths = create_paths_dict()
    MiceDict = read_csv_file(paths['csv_path'])

    flood_fill_color_table = ast.literal_eval(config['Analysis']['region_colors'])[:int(config['Analysis']['num_regions'])]
    ROI_list = list(MiceDict.keys())[7:7+int(config['Analysis']['num_regions'])]

    if len(flood_fill_color_table)<int(config['Analysis']['num_regions']):
        raise Exception("Enter more region_colors in configurations.ini")
    if len(ROI_list)<int(config['Analysis']['num_regions']):
        raise Exception("Add more regions in CSV table or decrease num_regions in configurations.ini")

    flood_fill_color_table.extend([255, 0])
    ROI_list.extend(['No region', 'Border'])

    return ROI_list,flood_fill_color_table

def create_color_palette():
    # Color palette
    sns.set_style("white")
    color_names = ["windows blue", "red", "amber", "faded green", "dusty purple", "orange"]
    return sns.xkcd_palette(color_names)
