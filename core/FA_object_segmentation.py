import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, binary_dilation, disk
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter 
from typing import Tuple,  List

from utils.pre_processing_utils import intensity_normalization
from utils.vessel_2d import filament_2d_wrapper

class FA_obj_segmenter:
    def __init__(
        self,
        cellmask_ch: int = 0,
        major_fa_ch: int = 1,
        thres_cellmask: float = 97.7,
        thres_major_FA: float = 101,
        close_size: int = 5,
        intensity_scaling_param: Tuple[int, int] = (10, 40),
        log_sigma: float = 1.0,
        response_threshold: float = 0.01,
        major_FA_res_thresholds: List[float] = [0.5, 0.5, 0.5],
        sauvola_window_size: int = 201,
        sauvola_k: float = -1.0,
        min_object_size: int = 6,
        intensity_profile = None,
    ):
        self.cellmask_ch = cellmask_ch
        self.major_fa_ch = major_fa_ch
        self.thres_cellmask = thres_cellmask
        self.thres_major_FA = thres_major_FA
        self.close_size = close_size
        self.intensity_scaling_param = intensity_scaling_param
        self.log_sigma = log_sigma
        self.response_threshold = response_threshold
        self.major_FA_res_thresholds = major_FA_res_thresholds
        self.sauvola_window_size = sauvola_window_size
        self.sauvola_k = sauvola_k
        self.min_object_size = min_object_size
        self.intensity_profile = intensity_profile        
        

    def cellmask_img_correction(self, cellmask_img: np.ndarray) -> np.ndarray:
        kernel_size = cellmask_img.shape[0]
        sigma = 1
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
        gauss = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) 
        cellmask_median_background_img = gauss * 8 + 100

        smooth_cellmask_img = gaussian_filter(cellmask_img, sigma=2, mode='nearest', truncate=3)
        smooth_cellmask_img_corrected = smooth_cellmask_img * 100 / cellmask_median_background_img
    
        return smooth_cellmask_img_corrected

    def target_cell_mask_seg(self, cellmask_img: np.ndarray) -> np.ndarray:

        target_cell_mask = cellmask_img > self.thres_cellmask
        target_cell_mask = remove_small_objects(target_cell_mask > 0, min_size=3, connectivity=1)     
        target_cell_mask = binary_closing(target_cell_mask, disk(self.close_size))
        target_cell_mask = remove_small_objects(target_cell_mask > 0, min_size=10, connectivity=1)
        target_cell_mask = binary_opening(target_cell_mask, disk(3))   
        target_cell_mask = ndi.binary_fill_holes(target_cell_mask)
        target_cell_mask = remove_small_objects(target_cell_mask > 0, min_size=30000, connectivity=1)
        
        # keep the one in the center
        label_cell_seg = label(target_cell_mask)
        regionprops_cells = regionprops(label_cell_seg)
        
        if(len(regionprops_cells)>1):
            distance_array = np.zeros(len(regionprops_cells))+10000
            boarder_flag = np.zeros(len(regionprops_cells))
            
            for ci in range(len(regionprops_cells)):
                distance_array[ci] = abs(regionprops_cells[ci]['Centroid'][0]- float(target_cell_mask.shape[0])/2) +abs(regionprops_cells[ci]['Centroid'][1]- float(target_cell_mask.shape[1])/2)
                if regionprops_cells[ci]['bbox'][0] ==0 or regionprops_cells[ci]['bbox'][1] ==0:
                    boarder_flag[ci]=1
                if regionprops_cells[ci]['bbox'][2] == target_cell_mask.shape[0] or regionprops_cells[ci]['bbox'][3] == target_cell_mask.shape[1]:
                    boarder_flag[ci]=1
        
            sort_ind= np.argsort(distance_array)
    
            target_cell_mask_center  = np.zeros_like(target_cell_mask)
            first_label = regionprops_cells[sort_ind[0]]['label']
            second_label = regionprops_cells[sort_ind[1]]['label']
    
            target_cell_mask_center[label_cell_seg==first_label] = first_label
            # print(first_label)
            if(boarder_flag[sort_ind[1]]==0):
               target_cell_mask_center[label_cell_seg==second_label] = second_label
        else:
               target_cell_mask_center =  target_cell_mask.copy()
              
        return target_cell_mask_center

    def apply_fa_obj_segmentation(self, img: np.ndarray) -> np.ndarray:
        cellmask_img = img[self.cellmask_ch, :, :]
        major_FA_img = img[self.major_fa_ch, :, :]

        # smooth_cellmask_img_corrected = self.cellmask_img_correction(cellmask_img)
      
        target_cell_mask = self.target_cell_mask_seg(cellmask_img)
       
        intensity_incell = major_FA_img[target_cell_mask>0]
        vmin = np.percentile(intensity_incell,0.2)
        vmax = np.percentile(intensity_incell,99.8)
        norm_major_FA_img = (major_FA_img-vmin)/(vmax-vmin)
        norm_major_FA_img[norm_major_FA_img<0]=0
        norm_major_FA_img[norm_major_FA_img>1]=1


        # norm_major_FA_img = intensity_normalization(major_FA_img,self.intensity_scaling_param)

        # vmin = self.intensity_profile[f'ch{self.major_fa_ch}_mean'] - self.intensity_scaling_param[0]*self.intensity_profile[f'ch{self.major_fa_ch}_std']
        # vmax = self.intensity_profile[f'ch{self.major_fa_ch}_mean'] + self.intensity_scaling_param[1]*self.intensity_profile[f'ch{self.major_fa_ch}_std']
        
        # x_clipped = np.clip(major_FA_img, vmin, vmax)
        # norm_major_FA_img = (x_clipped - vmin) / (vmax - vmin)


        # smooth_major_FA_img = gaussian_filter(norm_major_FA_img, sigma=0.5, mode="nearest", truncate=3)
        smooth_major_FA_img = norm_major_FA_img.copy() # gaussian_filter(norm_major_FA_img, sigma=0.5, mode="nearest", truncate=3)

        no_back_smooth_pax_img = smooth_major_FA_img.copy()
        no_back_smooth_pax_img[target_cell_mask == 0] = np.mean(smooth_major_FA_img[target_cell_mask == 1])

        thresh_sauvola = threshold_sauvola(no_back_smooth_pax_img, window_size=self.sauvola_window_size, k=self.sauvola_k)
        binary_sauvola = norm_major_FA_img > thresh_sauvola

        f2_param = [[1, 0.5]]
        major_FA_res_1 = filament_2d_wrapper(smooth_major_FA_img, f2_param)
        f2_param = [[2, 0.5]]
        major_FA_res_2 = filament_2d_wrapper(smooth_major_FA_img, f2_param)
        f2_param = [[3, 0.5]]
        major_FA_res_3 = filament_2d_wrapper(smooth_major_FA_img, f2_param)

        major_FA_seg_1 = major_FA_res_1 > self.major_FA_res_thresholds[0]
        major_FA_seg_2 = major_FA_res_2 > self.major_FA_res_thresholds[1]
        major_FA_seg_3 = major_FA_res_3 > self.major_FA_res_thresholds[2]

        binary_sauvola = remove_small_objects(binary_sauvola > 0, min_size=self.min_object_size, connectivity=1)

        response = -1 * (self.log_sigma ** 2) * ndi.gaussian_laplace(smooth_major_FA_img, self.log_sigma)
        major_FA_seg_dot = response > self.response_threshold
        major_FA_seg_dot = remove_small_objects(major_FA_seg_dot > 0, min_size=self.min_object_size, connectivity=1)

        # major_FA_seg = (major_FA_seg_dot + major_FA_seg_1 + major_FA_seg_2 + major_FA_seg_3 + binary_sauvola) > 0
        major_FA_seg = (major_FA_seg_dot + major_FA_seg_1 + major_FA_seg_2 +  binary_sauvola) > 0
        major_FA_seg = major_FA_seg * target_cell_mask

        return [major_FA_seg, target_cell_mask]  
