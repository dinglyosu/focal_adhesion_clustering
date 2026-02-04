# this code is adoptted from aics-segmentation repo
# https://github.com/AllenInstitute/aics-segmentation/tree/master/aicssegmentation/core

import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import os
import numpy as np
import cv2
from czifile import CziFile
from tqdm import tqdm

def intensity_normalization(struct_img, scaling_param):

    '''
    Mode 1:  scaling_param = [0]
    Mode 2:  scaling_param = [lower std range, upper std range]
    Mode 3:  scaling_param = [lower std range, upper std range, lower abs intensity, higher abs intensity]
    '''
    assert len(scaling_param) > 0

    if len(scaling_param) == 1:
        if scaling_param[0] < 1:
            print('intensity normalization: using min-max normalization with NO absolute intensity upper bound')
        else:
            print(f'intensity normalization: using min-max normalization with absolute intensity upper bound {scaling_param[0]}')
            struct_img[struct_img > scaling_param[0]] = struct_img.min()
        strech_min = struct_img.min()
        strech_max = struct_img.max()
        struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
    elif len(scaling_param) == 2:
        # print(f'intensity normalization: normalize into [mean - {scaling_param[0]} x std, mean + {scaling_param[1]} x std] ')
        m, s = norm.fit(struct_img.flat)
        # print(m,s)
        # import numpy as np
        # import pdb; pdb.set_trace()
        strech_min = max(m - scaling_param[0] * s, struct_img.min())
        strech_max = min(m + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img > strech_max] = strech_max
        struct_img[struct_img < strech_min] = strech_min
        struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
    elif len(scaling_param) == 4:
        img_valid = struct_img[np.logical_and(struct_img > scaling_param[2], struct_img < scaling_param[3])]
        m, s = norm.fit(img_valid.flat)
        strech_min = max(scaling_param[2] - scaling_param[0] * s, struct_img.min())
        strech_max = min(scaling_param[3] + scaling_param[1] * s, struct_img.max())
        struct_img[struct_img > strech_max] = strech_max
        struct_img[struct_img < strech_min] = strech_min
        struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)

    # print('intensity normalization completes')
    return struct_img


def group_intensity_normalization_meanstd(struct_img, scaling_param, intensity_profile):

    assert len(scaling_param) ==2

    m = intensity_profile.mean
    s = intensity_profile.std
    
    strech_min = m - scaling_param[0] * s
    strech_max = m + scaling_param[1] * s
    struct_img[struct_img > strech_max] = strech_max
    struct_img[struct_img < strech_min] = strech_min
    struct_img = (struct_img - strech_min + 1e-8)/(scaling_param[0] * s + scaling_param[0] * s + 1e-8)

    # print('intensity normalization completes')
    return struct_img

def group_intensity_normalization_percentile(struct_img, intensity_profile, min_percentile, max_percentile):

    strech_min = intensity_profile[f'percentile_{min_percentile}']
    strech_max = intensity_profile[f'percentile_{max_percentile}']
    struct_img[struct_img > strech_max] = strech_max
    struct_img[struct_img < strech_min] = strech_min
    struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)

    # print('intensity normalization completes')
    return struct_img



def intensity_normalization_known_percentile(struct_img, strech_min,strech_max):

    struct_img[struct_img > strech_max] = strech_max
    struct_img[struct_img < strech_min] = strech_min
    struct_img = (struct_img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)

    # print('intensity normalization completes')
    return struct_img



def image_smoothing_gaussian_3d(struct_img, sigma, truncate_range=3.0):

    structure_img_smooth = gaussian_filter(struct_img, sigma=sigma, mode='nearest', truncate=truncate_range)

    return structure_img_smooth


def image_smoothing_gaussian_slice_by_slice(struct_img, sigma, truncate_range=3.0):

    structure_img_smooth = np.zeros_like(struct_img)
    for zz in range(struct_img.shape[0]):
        structure_img_smooth[zz, :, :] = gaussian_filter(struct_img[zz, :, :], sigma=sigma, mode='nearest',
                                                         truncate=truncate_range)

    return structure_img_smooth


def edge_preserving_smoothing_3d(struct_img, numberOfIterations=10, conductance=1.2, timeStep=0.0625):
    import itk
    # numberOfIteration was 5 

    itk_img = itk.GetImageFromArray(struct_img.astype(np.float32))

    gradientAnisotropicDiffusionFilter = itk.GradientAnisotropicDiffusionImageFilter.New(itk_img)
    gradientAnisotropicDiffusionFilter.SetNumberOfIterations(numberOfIterations)
    gradientAnisotropicDiffusionFilter.SetTimeStep(timeStep)
    gradientAnisotropicDiffusionFilter.SetConductanceParameter(conductance)
    gradientAnisotropicDiffusionFilter.Update()

    itk_img_smooth = gradientAnisotropicDiffusionFilter.GetOutput()

    img_smooth_ag = itk.GetArrayFromImage(itk_img_smooth)

    return img_smooth_ag


def suggest_normalization_param(structure_img0):
    m, s = norm.fit(structure_img0.flat)
    print(f'mean intensity of the stack: {m}')
    print(f'the standard deviation of intensity of the stack: {s}')

    p99 = np.percentile(structure_img0, 99.99)
    print(f'0.9999 percentile of the stack intensity is: {p99}')

    pmin = structure_img0.min()
    print(f'minimum intensity of the stack: {pmin}')

    pmax = structure_img0.max()
    print(f'maximum intensity of the stack: {pmax}')

    up_ratio = 0
    for up_i in np.arange(0.5, 1000, 0.5):
        if m+s * up_i > p99:
            if m+s * up_i > pmax:
                print(f'suggested upper range is {up_i-0.5}, which is {m+s*(up_i-0.5)}')
                up_ratio = up_i-0.5
            else:
                print(f'suggested upper range is {up_i}, which is {m+s*up_i}')
                up_ratio = up_i
            break

    low_ratio = 0
    for low_i in np.arange(0.5, 1000, 0.5):
        if m-s*low_i < pmin:
            print(f'suggested lower range is {low_i-0.5}, which is {m-s*(low_i-0.5)}')
            low_ratio = low_i-0.5
            break

    print(f'So, suggested parameter for normalization is [{low_ratio}, {up_ratio}]')
    print('To further enhance the contrast: You may increase the first value (may loss some dim parts), or decrease the second value' +
          '(may loss some texture in super bright regions)')
    print('To slightly reduce the contrast: You may decrease the first value, or increase the second value')



def split_norm_process_all_2D_czi_folders(root_folder, output_dir, low_p=1, high_p=99):
    """Wrapper function to process all subfolders containing .czi files."""
    subfolder_names =  [ d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    
    for subfolder_name in subfolder_names:
        sub_czi_folder = os.path.join(root_folder, subfolder_name)
        print(f"Processing folder: {sub_czi_folder}")
        sub_output_dir = os.path.join(output_dir,subfolder_name)
        os.makedirs(sub_output_dir, exist_ok=True)
        split_norm_process_czi_folder(sub_czi_folder, sub_output_dir, low_p, high_p)
    
    print("All folders processed!")


def split_norm_process_all_2D_czi_folders(root_folder, output_dir, low_p=1, high_p=99):
    """Wrapper function to process all subfolders and nested subfolders containing .czi files."""
    for root, dirs, files in os.walk(root_folder):
        if any(file.endswith(".czi") for file in files):
            relative_path = os.path.relpath(root, root_folder)
            output_subdir = os.path.join(output_dir, relative_path)
            print(f"Processing folder: {root} -> {output_subdir}")
            split_norm_process_czi_folder(root, output_subdir, low_p, high_p)
    
    print("All folders processed!")



def split_norm_process_czi_folder(czi_folder, output_dir, low_p=1, high_p=99):
    """Process .czi files from a folder, normalize using specified percentiles, and save all channels."""
    os.makedirs(output_dir, exist_ok=True)
    all_channels_dir = os.path.join(output_dir, "all_channels")
    os.makedirs(all_channels_dir, exist_ok=True)
    
    # Compute global percentiles
    channel_values = {}
    print(f"Gathering pixel values from {czi_folder}...")
    for file_name in tqdm(os.listdir(czi_folder)):
        if file_name.endswith(".czi"):
            with CziFile(os.path.join(czi_folder, file_name)) as czi:
                img_data = np.squeeze(czi.asarray())                
                num_channels = img_data.shape[0]
                for ch in range(0,num_channels):                    
                    channel_img = img_data[ch, :, :].flatten()                    
                    if ch not in channel_values:
                        channel_values[ch] = channel_img
                    else:
                        channel_values[ch] = np.concatenate((channel_values[ch], channel_img))
    
    # Compute percentiles per channel
    channel_stats = {}
    for ch, values in channel_values.items():
        channel_stats[ch] = {
            'low': np.percentile(values, low_p),
            'high': np.percentile(values, high_p)
        }
    print("Global percentile normalization values computed.")
    
    # Process and save normalized images
    print(f"Processing files from {czi_folder}...")
    for file_name in tqdm(os.listdir(czi_folder)):
        if file_name.endswith(".czi"):
            with CziFile(os.path.join(czi_folder, file_name)) as czi:
                img_data = np.squeeze(czi.asarray()) 
                base_name = os.path.splitext(os.path.basename(file_name))[0]
                for ch in range(img_data.shape[0]):  # Process all channels
                    channel_img = img_data[ch, :, :]
                    # Normalize using computed percentiles
                    low, high = channel_stats[ch]['low'], channel_stats[ch]['high']
                    channel_img = np.clip((channel_img - low) / (high - low), 0, 1) * 255
                    channel_img = channel_img.astype(np.uint8)

                    # Save each channel in its own directory
                    channel_dir = os.path.join(all_channels_dir, f"ch{ch}")
                    os.makedirs(channel_dir, exist_ok=True)

                    channel_path = os.path.join(channel_dir, f"{base_name}_ch{ch}.png")
                    cv2.imwrite(channel_path, channel_img)
            print(f"Saved all channels for {base_name}.")