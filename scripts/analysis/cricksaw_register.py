import os
import socket
import sys
import numpy as np

print('start')
##################### VARIABLES #####################
# Define the path to data set
# Global


chan2use = 3
chan2exclude = []  # MUST be string

machine = socket.gethostname()
if machine == 'C02Z85AULVDC':
    root_directory = '/Users/blota/Data/'
    atlas_directory = '/Users/blota/Data/ARA_CCFv3'
    raw_directory = root_directory
    processed_directory = raw_directory
elif machine == 'chainsaw':
    winstor = '/mnt/blota/winstor/swc/mrsic_flogel/public/projects'
    atlas_directory = '/mnt/data/Antonin/Brainsaw_temp_folder/'
    raw_directory = '/mnt/data/Antonin/Brainsaw_temp_folder/'
elif machine == 'blender':
    print('on blender')
    winstor = '/mnt/blota/winstor/swc/mrsic_flogel/public/projects'
    #root_directory = '/mnt/data/Antonin/Brainsaw_temp_folder/'
    atlas_directory = '/mnt/data/Antonin/Brainsaw_temp_folder/'
    raw_directory = '/mnt/data/Antonin/Brainsaw_temp_folder/'
    processed_directory = raw_directory
    # raw_directory = os.path.join(winstor, 'AnBl_20180101_LPCortexSFTF/Anatomy/retro_RR')
else:
    raise (IOError('Cannot recognise machine %s' % machine))

paramPath = os.path.join(atlas_directory, 'elastix_transforms/')  # path to parameter files

atlas_size = 25
do_downsample = False
do_roi = False
do_roi_processing = False  # For ROI do some extra analysis
do_register = False
do_register_inverse = False
do_trans_template = False
do_trans_data = True
# transform_list = ['translation', 'rigid', 'affine', 'bspline']
transform_list = ['01_ARA_translate', '02_ARA_rigid', '03_ARA_affine','04_ARA_bspline']

ramcheck = True
todo = ['PZAH5.6a']

atlas_directory = os.path.join(atlas_directory, 'ARA_%i_micron_mhd' % atlas_size)
path2template = os.path.join(atlas_directory, 'template.mhd')
path2json = os.path.join(atlas_directory, 'labels.json')
path2atlas = os.path.join(atlas_directory, 'atlas.mhd')
path2borders = os.path.join(atlas_directory, 'borders.mhd')
path2borders_withAL = os.path.join(atlas_directory, 'borders_with_ALPMLP.mhd')

#### CODE ####

# Define the path to code and add to path
from cricksaw_analysis.elastix_registration import register_functions
from cricksaw_analysis.elastix_registration import pre_processing
from cricksaw_analysis.elastix_registration import atlas_utils

if do_trans_template:
    print('Atlas loading')
    path2floatatlas = os.path.join(atlas_directory, 'float_atlas.mhd')
    float_atlas, translator = atlas_utils.create_float_atlas(path2atlas, path2floatatlas)

for mouse_name in todo:
    if not os.path.isdir(os.path.join(raw_directory, mouse_name)):
        print('No folder for %s. Skipping\n' % mouse_name)
        continue
    print('\n\nDoing %s\n' % mouse_name)
    raw_root_path = os.path.join(raw_directory, mouse_name)
    processed_root_path = os.path.join(processed_directory, mouse_name)

    roiPtsFiles = []
    cellPtsFiles = []
    # find all roi points in the masiv directory
    masivPath = os.path.join(processed_directory, mouse_name, mouse_name + '_MaSIV')
    cellPath = os.path.join(processed_directory, 'cellfinder_res')

    if do_roi:
        if not os.path.isdir(masivPath):
            print('No masiv roi folder, skip')
        else:
            for fname in os.listdir(masivPath):
                # find tracks in masiv directory
                if fname.endswith('.yml') and any([m in fname for m in ('injSite', 'injsite', 'track')]):
                    roi_path = os.path.join(masivPath, fname)
                    roiPtsFiles.append(roi_path)
                # find RR cells in masiv directory for manually clicked cells
                elif ('cell' in fname.lower()) and (fname.endswith('yml')):
                    cell_path = os.path.join(masivPath, fname)
                    cellPtsFiles.append(cell_path)

        if not os.path.isdir(cellPath):
            print('No cell folder. Skip')
        else:
            for fname in os.listdir(cellPath):
                # find rab cells in cellfinder directory
                if fname.endswith('.xml') and (mouse_name in fname):
                    cell_path = os.path.join(cellPath, fname)
                    cellPtsFiles.append(cell_path)

    ########## Check everything is in place #############
    path2reg = os.path.join(processed_root_path, 'registration_%i' % atlas_size)
    if not os.path.isdir(path2reg):
        os.mkdir(path2reg)

    # Find data folder
    dataPath = os.path.join(raw_root_path, 'stitchedImages_100')
    assert os.path.isdir(dataPath)

    # Check that I have the recipe file
    recipeFile = [fname for fname in os.listdir(raw_root_path) if fname.startswith('recipe')]
    assert len(recipeFile) == 1
    recipeFile = os.path.join(raw_root_path, recipeFile[0])

    # Count channels
    chanNames = [d for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, d))]
    # chan2use = max([int(c) for c in chanNames if c.isdecimal()])
    chanNames = [c for c in chanNames if c not in chan2exclude]
    if do_register:
        assert str(chan2use) in chanNames

    # Read parameter maps
    pmDict = register_functions.load_parameter_maps(paramPath)

    ############# Downsample ##################
    if do_roi:
        # Start by downsampling the rois
        roi2register = []
        cell2register = []
        if roiPtsFiles:
            print('Downsampling ROIs')
            for roi_path in roiPtsFiles:
                roi_name = os.path.split(roi_path)[1]
                target = os.path.join(path2reg, '%s_downsampled_%ium_roi%s' % (mouse_name, atlas_size, roi_name))
                fulltarget = os.path.join(path2reg, '%s_fullres_roi_%s' % (mouse_name,
                                                                           roi_name.replace('.xml', '.pts')))
                roi_dict = pre_processing.downsample_brainsaw_roi(roi_path, path2recipe=recipeFile,
                                                                  target=target, full_res_target=fulltarget,
                                                                  outsize=atlas_size, force=True)

                # also save in pts format to be able to register
                for ctype, roi_coords in roi_dict.items():
                    target = os.path.join(path2reg, '%s_%s_downsampled_%ium_step00_part00.pts' % (
                        roi_name.replace('.yml', ''), ctype, atlas_size))
                    rois_io.write_pts_file(target, np.asarray([c[0] for c in roi_coords]),
                                           np.asarray([c[1] for c in roi_coords]),
                                           np.asarray([c[2] for c in roi_coords]), force=True, index=False)
                    roi2register.append(target)
        if cellPtsFiles:
            print('Downsampling Cells')
            for cell_path in cellPtsFiles:
                cell_name = os.path.split(cell_path)[1]
                target = os.path.join(path2reg, '%s_downsampled_%ium_cell_%s' % (mouse_name, atlas_size,
                                                                                 cell_name.replace('.xml', '.yml')))
                fulltarget = os.path.join(path2reg, '%s_fullres_cell_%s' % (mouse_name,
                                                                            cell_name.replace('.xml', '.pts')))
                cell_dict = pre_processing.downsample_brainsaw_roi(cell_path, path2recipe=recipeFile,
                                                                   target=target, full_res_target=fulltarget,
                                                                   outsize=atlas_size, force=True)

                # also save in pts format to be able to register
                for ctype, cell_coords in cell_dict.items():
                    target = os.path.join(path2reg, '%s_%s_downsampled_%ium_step00_part00.pts' % (
                        cell_name.replace('.xml', ''), ctype, atlas_size))
                    rois_io.write_pts_file(target, np.asarray([c[0] for c in cell_coords]),
                                           np.asarray([c[1] for c in cell_coords]),
                                           np.asarray([c[2] for c in cell_coords]), force=True, index=False)
                    cell2register.append(target)

    if do_downsample:
        # Start by downsampling all the channels

        print('Doing %s' % mouse_name)
        for cN in chanNames:
            target = os.path.join(path2reg, '%s_downsampled_%ium_chan%s.mhd' % (mouse_name, atlas_size, cN))
            print('Downsampling channel %s...' % cN)
            pre_processing.downsample_brainsaw(path2tiffs=os.path.join(dataPath, cN), path2recipe=recipeFile,
                                               outsize=atlas_size, target=target, check_for_ram=ramcheck)

    if do_register:
        ############# STEP 01 ##################
        movingImageName = path2template
        fixedImage = os.path.join(path2reg, '%s_downsampled_%ium_chan%s.mhd' % (mouse_name, atlas_size, chan2use))

        transformedImage = register_functions.registration_single_step(fixedImage, movingImageName,
                                                                       path_to_save=path2reg,
                                                                       prefix=mouse_name, step_number=1,
                                                                       transform_list=transform_list,
                                                                       fixed_pts=None, moving_pts=None,
                                                                       parameter_map_dictionary=pmDict)
    if do_register_inverse:
        fixedImage = path2template
        movingImageName = os.path.join(path2reg, '%s_downsampled_%ium_chan%s.mhd' % (mouse_name, atlas_size, chan2use))

        transformedImage = register_functions.registration_single_step(fixedImage, movingImageName,
                                                                       path_to_save=path2reg,
                                                                       prefix='%ss_inverse_reg_' % mouse_name,
                                                                       step_number=1,
                                                                       transform_list=transform_list,
                                                                       fixed_pts=None, moving_pts=None,
                                                                       parameter_map_dictionary=pmDict)

    if do_roi:
        ################### PROCESS THE ROIs #############################
        print('Registering ROIs')
        registered_rois = register_functions.register_roi(path2reg, roi2register, mouse_name)
        registered_cells = register_functions.register_roi(path2reg, cell2register, mouse_name)

        if needs_LR_swap[mouse_name]:
            # I need to find the things registered in the atlas and swap left/right
            register_functions.swap_hem_atlas_roi(transform_folder=path2reg, roi_files=registered_rois,
                                                  atlas_size=atlas_size)
            register_functions.swap_hem_atlas_roi(transform_folder=path2reg, roi_files=registered_cells,
                                                  atlas_size=atlas_size)

        if do_roi_processing:
            # extra steps of processing creating new outputs

            print('Processing ROIs')
            roi_to_process = process_rab_roi.get_useful(cell2register, registered_cells)
            for base_name, (raw_roi_file, reg_roi_file) in roi_to_process.items():
                # Find depth
                depth_dict = process_rab_roi.find_depth(reg_roi_file, atlas_path=atlas_directory)
                for what, value in depth_dict.items():
                    np.save(os.path.join(path2reg, base_name + '_%s.npy' % what), value)

                # find cortical cells
                out = process_rab_roi.find_cortical_cells(reg_roi_file, raw_roi_file, path2atlas, path2json)
                is_ctx, layer, dict_by_layer_reg, dict_by_layer_raw = out
                # rename for masiv
                order = ['1', '2/3', '4', '5', '6a', '6b', 'not cortex', 'outside atlas']
                dict_for_masiv_raw = dict()
                dict_for_masiv_reg = dict()
                for layer_name in dict_by_layer_raw:
                    dict_for_masiv_raw['Type%d' % order.index(layer_name)] = dict_by_layer_raw[layer_name][:, [1, 2, 0]]
                    dict_for_masiv_reg['Type%d' % order.index(layer_name)] = dict_by_layer_reg[layer_name][:, [1, 2, 0]]
                rois_io.write_masiv_roi(dict_for_masiv_raw, os.path.join(path2reg, base_name + '_raw_by_layer.yml'),
                                        force=True)
                rois_io.write_masiv_roi(dict_for_masiv_reg,
                                        os.path.join(path2reg, base_name + '_registered_by_layer.yml'),
                                        force=True)
                # sanity check?
                print('ok')
            print('done')

    if do_trans_template:
        # Transform the template to have a background
        # find the size in the mhd file
        # Also apply to the atlas
        with open(path2template, 'r') as mhd_file:
            for line in mhd_file:
                if 'size' not in line.lower():
                    continue
                _, mhd_size = line.split('=')
                mhd_size = tuple(i for i in mhd_size.strip().split(' '))
                break
        spacing = tuple(str(1) for i in range(3))  # was atlas_size / 25 when I register at 25 and transform at 10
        register_functions.apply_registration(path2reg, path2template, mouse_name, suffix='_tpl%ium' % atlas_size,
                                              int_image=False, spacing=spacing, size=mhd_size)
        reg_float_atlas, path_to_reg_float = register_functions.apply_registration(path2reg, path2floatatlas,
                                                                                   root_name=mouse_name,
                                                                                   suffix='_tpl%ium' % atlas_size,
                                                                                   int_image=True,
                                                                                   spacing=spacing, size=mhd_size)
        # now translate back
        path_to_reg_atlas = path_to_reg_float.replace('float_', '')
        atlas_utils.translate_atlas(reg_float_atlas, translator, path_to_reg_atlas)

        register_functions.apply_registration(path2reg, path2borders, mouse_name, suffix='_tpl%ium' % atlas_size,
                                              int_image=False, spacing=spacing, size=mhd_size)
        register_functions.apply_registration(path2reg, path2borders_withAL, mouse_name, suffix='_tpl%ium' % atlas_size,
                                              int_image=False, spacing=spacing, size=mhd_size)

    if do_trans_data:
        spacing = tuple(str(1) for i in range(3))  # was atlas_size / 25 when I register at 25 and transform at 10
        for cN in chanNames:
            downsample_stack = os.path.join(path2reg, '%s_downsampled_%ium_chan%s.mhd' % (mouse_name, atlas_size, cN))
            with open(downsample_stack, 'r') as mhd_file:
                for line in mhd_file:
                    if 'size' not in line.lower():
                        continue
                    _, mhd_size = line.split('=')
                    mhd_size = tuple(i for i in mhd_size.strip().split(' '))
                    break
            register_functions.apply_registration(path2reg, downsample_stack, '%ss_inverse_reg_' % mouse_name,
                                                  suffix='_to_%ium_atlas' % atlas_size,
                                                  int_image=False, spacing=spacing, size=mhd_size)
print('Done!')
