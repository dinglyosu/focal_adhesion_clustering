import numpy as np
import pandas as pd
from typing import Optional, List, Union
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_dilation, disk
from skimage.filters import sobel_v, sobel_h
from scipy.ndimage import distance_transform_cdt
from utils.pre_processing_utils import intensity_normalization, group_intensity_normalization_meanstd, group_intensity_normalization_percentile,intensity_normalization_known_percentile

from skimage.filters import rank



class FA_obj_feature_extractor:
    def __init__(
        self, 
        pax_seg: Optional[np.ndarray] = None, 
        # raw_multich_img: Optional[np.ndarray] = None, 
        input_multich_img: Optional[np.ndarray] = None, 
        new_cell_mask: Optional[np.ndarray] = None, 
        front_mask: Optional[np.ndarray] = None, 
        filename: str = "unknown", 
        label_filename: str = "unknown", 
        filenameID: int = 0, 
        time_point: int = 0, 
        pixel_size: float = 0.07, 
        intensity_scaling_param: List[int] = [10, 40], 
        major_fa_ch: int = 0, 
        cellmask_ch: int = 3,
        supplementary_fa_ch: int = [0, 2],
        intensity_profile = None,         
        given_percentiles: float = [0.2, 99.8],
    ):
        self.pax_seg: np.ndarray = pax_seg if pax_seg is not None else np.zeros((100, 100), dtype=np.uint8)
        self.new_cell_mask: np.ndarray = new_cell_mask if new_cell_mask is not None else np.zeros((100, 100), dtype=bool)
        self.front_mask: np.ndarray = front_mask if front_mask is not None else new_cell_mask        
        self.major_fa_ch: int = major_fa_ch
        self.cellmask_ch: int = cellmask_ch
        self.intensity_scaling_param: List[int] = intensity_scaling_param
        self.input_multich_img: np.ndarray = (
            input_multich_img if input_multich_img is not None else np.zeros((2, 100, 100), dtype=np.float32)
        )
        self.filename: str = filename
        self.label_filename : str = label_filename
        self.filenameID: int = filenameID
        self.time_point: int = time_point
        self.pixel_size: float = pixel_size
        self.prop_df_pax: pd.DataFrame = pd.DataFrame()    
        self.supplementary_fa_ch: int = supplementary_fa_ch
        self.intensity_profile = intensity_profile  
        self.given_percentiles = given_percentiles        
    
    def extract_features(self):
        raw_pax_img =  self.input_multich_img[self.major_fa_ch]
        input_pax_img =  self.input_multich_img[self.major_fa_ch]
        
        group_norm_pax_img = intensity_normalization_known_percentile(input_pax_img,
                                            self.intensity_profile[f'ch{self.major_fa_ch}_percentile_{self.given_percentiles[0]}'],
                                            self.intensity_profile[f'ch{self.major_fa_ch}_percentile_{self.given_percentiles[1]}'])

        intensity_incell = input_pax_img[self.new_cell_mask>0]
        vmin = np.percentile(intensity_incell,0.2)
        vmax = np.percentile(intensity_incell,99.8)
        indiv_norm_pax_img = (input_pax_img-vmin)/(vmax-vmin)
        indiv_norm_pax_img[indiv_norm_pax_img<0]=0
        indiv_norm_pax_img[indiv_norm_pax_img>1]=1
        
        ######################################################
        print(self.major_fa_ch)
        print(self.supplementary_fa_ch)

        raw_supplementary_img_1 = self.input_multich_img[self.supplementary_fa_ch[0]]
        raw_supplementary_img_2 = self.input_multich_img[self.supplementary_fa_ch[1]]

        group_norm_supplementary_img_1 = intensity_normalization_known_percentile(raw_supplementary_img_1,
                                            self.intensity_profile[f'ch{self.supplementary_fa_ch[0]}_percentile_{self.given_percentiles[0]}'],
                                            self.intensity_profile[f'ch{self.supplementary_fa_ch[0]}_percentile_{self.given_percentiles[1]}'])
    
        group_norm_supplementary_img_2 = intensity_normalization_known_percentile(raw_supplementary_img_2,
                                            self.intensity_profile[f'ch{self.supplementary_fa_ch[1]}_percentile_{self.given_percentiles[0]}'],
                                            self.intensity_profile[f'ch{self.supplementary_fa_ch[1]}_percentile_{self.given_percentiles[1]}'])

        intensity_incell = raw_supplementary_img_1[self.new_cell_mask>0]
        vmin = np.percentile(intensity_incell,0.2)
        vmax = np.percentile(intensity_incell,99.8)
        indiv_norm_supplementary_img_1 = (raw_supplementary_img_1-vmin)/(vmax-vmin)
        indiv_norm_supplementary_img_1[indiv_norm_supplementary_img_1<0]=0
        indiv_norm_supplementary_img_1[indiv_norm_supplementary_img_1>1]=1


        intensity_incell = raw_supplementary_img_2[self.new_cell_mask>0]
        vmin = np.percentile(intensity_incell,0.2)
        vmax = np.percentile(intensity_incell,99.8)
        indiv_norm_supplementary_img_2 = (raw_supplementary_img_2-vmin)/(vmax-vmin)
        indiv_norm_supplementary_img_2[indiv_norm_supplementary_img_2<0]=0
        indiv_norm_supplementary_img_2[indiv_norm_supplementary_img_2>1]=1

        # Label segmentations and calculate the region properties
        label_pax_seg = label(self.pax_seg)
        
        regionprops_pax = regionprops(label_pax_seg, intensity_image=raw_pax_img)

        distance_taxicab = distance_transform_cdt(self.new_cell_mask, metric="taxicab")
        
        local_orientation = calculate_local_orientation(self.new_cell_mask)

        density_d9_map = rank.mean((self.pax_seg>0).astype(np.uint8)*255, disk(9)).astype(float)
        density_d15_map = rank.mean((self.pax_seg>0).astype(np.uint8)*255, disk(15)).astype(float)
        density_d25_map = rank.mean((self.pax_seg>0).astype(np.uint8)*255, disk(25)).astype(float)
        # print([np.median(density_d15_map[density_d15_map>0])])

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=256, facecolor='w', edgecolor='k')
        # ax.imshow(density_d15_map)
        # Initialize feature arrays
        for iL in range(label_pax_seg.max()):  
            # get eh zyxin intensity using the mask

            this_obj_bw = label_pax_seg == iL
            not_this_obj_bw = (self.pax_seg > 0) & (label_pax_seg != iL)
            distance_this_obj_bw = distance_transform_cdt( 1- this_obj_bw, metric="taxicab")
            min_obj_distance = min(distance_this_obj_bw[not_this_obj_bw>0])
        
            sup_1_raw_intensity_set = raw_supplementary_img_1[label_pax_seg == iL]
            sup_2_raw_intensity_set = raw_supplementary_img_2[label_pax_seg == iL]

            sup_1_groupnorm_intensity_set = group_norm_supplementary_img_1[label_pax_seg == iL]
            sup_2_groupnorm_intensity_set = group_norm_supplementary_img_2[label_pax_seg == iL]

            sup_1_indiv_norm_intensity_set = indiv_norm_supplementary_img_1[label_pax_seg == iL]
            sup_2_indiv_norm_intensity_set = indiv_norm_supplementary_img_2[label_pax_seg == iL]


            pax_raw_intensity_set = raw_pax_img[label_pax_seg == iL]
            pax_group_norm_intensity_set = group_norm_pax_img[label_pax_seg == iL]
            pax_indiv_norm_intensity_set = indiv_norm_pax_img[label_pax_seg == iL]
            

            # get the centroid of this obj
            obj_X_iL = regionprops_pax[iL].centroid[0]
            obj_Y_iL = regionprops_pax[iL].centroid[1]

            # a mask to select only the front mask objects, if provided
            if(self.front_mask[int(obj_X_iL), int(obj_Y_iL)]==0):
                continue

            density_d9_this_obj = density_d9_map[int(obj_X_iL), int(obj_Y_iL)]
            density_d15_this_obj = density_d15_map[int(obj_X_iL), int(obj_Y_iL)]
            density_d25_this_obj = density_d25_map[int(obj_X_iL), int(obj_Y_iL)]
            
            # get the distance to cell edge at centroid
            cell_edge_dist = distance_taxicab[int(obj_X_iL), int(obj_Y_iL)]
                        
            # get the orientation obteined from the cell shape, at the centroid of this obj
            cell_edge_orient = local_orientation[int(obj_X_iL), int(obj_Y_iL)]
            
            # get the difference of these two orientation
            diff_orient = cell_edge_orient - regionprops_pax[iL].orientation
            
            # normalize into -pi/2 ~ pi/2, process: add pi/2, warp to 0 ~ pi and then subtract pi/2
            diff_orient = (diff_orient + np.pi / 2) % np.pi - np.pi / 2
            
            # then take the abs of the angle
            diff_orient = np.abs(diff_orient)   
            
            # Append extracted features to DataFrame
            s = pd.Series([
                self.filename, self.label_filename, self.filenameID, self.time_point, self.pixel_size,
                regionprops_pax[iL].area, regionprops_pax[iL].bbox_area, regionprops_pax[iL].convex_area,
                regionprops_pax[iL].eccentricity, regionprops_pax[iL].equivalent_diameter, regionprops_pax[iL].euler_number,
                regionprops_pax[iL].extent, regionprops_pax[iL].filled_area, regionprops_pax[iL].label,
                regionprops_pax[iL].major_axis_length, regionprops_pax[iL].max_intensity, regionprops_pax[iL].mean_intensity,
                regionprops_pax[iL].min_intensity, regionprops_pax[iL].minor_axis_length, regionprops_pax[iL].orientation,
                regionprops_pax[iL].perimeter, regionprops_pax[iL].solidity, 
                cell_edge_dist, cell_edge_orient, diff_orient, 
                pax_raw_intensity_set.mean(), pax_group_norm_intensity_set.mean(),pax_indiv_norm_intensity_set.mean(),
                sup_1_raw_intensity_set.mean(), sup_1_groupnorm_intensity_set.mean(),sup_1_indiv_norm_intensity_set.mean(),
                sup_2_raw_intensity_set.mean(), sup_2_groupnorm_intensity_set.mean(),sup_2_indiv_norm_intensity_set.mean(),
                min_obj_distance,density_d25_this_obj,density_d9_this_obj,density_d15_this_obj
            ], index=[
                'filename','label_filename', 'cell_ID', 'time_point', 'pixel_size', 'area', 'bbox_area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number',
                'extent', 'filled_area', 'label', 'major_axis_length', 'max_intensity', 'mean_intensity',
                'min_intensity', 'minor_axis_length', 'orientation', 'perimeter', 'solidity',
                'cell_edge_dist', 'cell_edge_orient', 'diff_orient', 'pax_raw_int_mean', 'pax_group_norm_int_mean', 'pax_indiv_norm_int_mean', 
                'sup1_raw_int_mean', 'sup1_group_norm_int_mean', 'sup1_indiv_norm_int_mean', 
                'sup2_raw_int_mean', 'sup2_group_norm_int_mean', 'sup2_indiv_norm_int_mean', 
                'min_obj_distance','density_d25_this_obj','density_d9_this_obj','density_d15_this_obj'
            ])
            
            self.prop_df_pax = self.prop_df_pax.append(s, ignore_index=True)
        
        return [self.prop_df_pax, local_orientation]

def calculate_local_orientation(input_mask): 
    for_orent_mask = binary_opening(input_mask, disk(11))
    bigger_for_orent_mask = binary_dilation(for_orent_mask, disk(11))
    bigger_for_orent_distance_taxicab = distance_transform_cdt(bigger_for_orent_mask, metric="taxicab")

    # obtain cell edge orientation based on gradient of the distance map    
    n_v = sobel_v(bigger_for_orent_distance_taxicab)
    n_h = sobel_h(bigger_for_orent_distance_taxicab)
    # the gradients are really small, make them reasonable values
    for_plot_max = 5/min(n_v.max(), n_h.max())
    n_v = n_v*for_plot_max
    n_h = n_h*for_plot_max
            
    # convert the directions into orientation angels   
    local_orientation = np.arctan2(n_v,n_h)

    return local_orientation
