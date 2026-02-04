from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import binary_opening, binary_dilation
from skimage.morphology import disk
from scipy.ndimage import distance_transform_cdt
from utils.pre_processing_utils import intensity_normalization

def define_colormap_multipleobjects(cm_map_type: str = "tab10", number_limits: int = 200000) -> ListedColormap:
    """
    Generate a colormap that extends an existing colormap to accommodate a large number of segmented objects.

    Parameters:
    - cm_map_type (str, default "tab10"): Name of the colormap.
    - number_limits (int, default 200000): Number of required colors.

    Returns:
    - ListedColormap: A new colormap with `number_limits` colors.
    """
    # Retrieve the base colormap
    colormap_single = cm.get_cmap(cm_map_type)

    # Extract colors from the colormap
    base_colors = colormap_single(np.linspace(0, 1, colormap_single.N))  # Get discrete colors

    # Repeat colors to reach `number_limits`
    num_repeats = (number_limits // colormap_single.N) + 1  # Ensure enough colors
    extended_colors = np.tile(base_colors, (num_repeats, 1))  # Repeat color set

    # Add black (0,0,0,1) as the first color
    extended_colors = np.vstack([[0, 0, 0, 1], extended_colors[:number_limits]])

    # Return a new colormap with the modified colors
    return ListedColormap(extended_colors)


def display_fa_segmentation_panels(
    input_image: np.ndarray,
    input_cell_mask: np.ndarray,
    input_front_mask: np.ndarray,
    input_obj_seg: np.ndarray,
    local_orientation: np.ndarray,
    plot_output_dir: str,
    filename: str,
    flag_plot_save: int = 1,
    flag_run_all: int = 1,
    intensity_profile = None,
    intensity_scaling_param = [2,10],
    major_fa_ch = 1,
) -> np.ndarray:
    """
    Displays and saves focal adhesion segmentation panels.

    Args:
        input_image (np.ndarray): The input image.
        input_cell_mask (np.ndarray): The binary cell mask.
        input_obj_seg (np.ndarray): Segmented object mask.
        local_orientation (np.ndarray): Orientation map.
        plot_output_dir (str): Directory to save output plots.
        filename (str): Filename identifier.
        flag_plot_save (int, optional): Whether to save plots (1 = Yes, 0 = No). Defaults to 1.
        flag_run_all (int, optional): Whether to close figures after saving (1 = Yes, 0 = No). Defaults to 1.

    Returns:
        np.ndarray: The labeled image overlay.
    """
    
    if flag_plot_save:
        # Prepare directories for different plots
        sub_dirs = ['contour', 'quiver_cell', 'quiver_obj', 'label_color', 'rgb_plot', 'panel_plot']
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(plot_output_dir, sub_dir), exist_ok=True)
        
    # Label segmentations and calculate the region properties
    label_FA_obj = label(input_obj_seg)
    regionprops_pax = regionprops(label_FA_obj, intensity_image=input_image)
    
    extended_colors = define_colormap_multipleobjects()
  
    # vmin = intensity_profile[f'ch{major_fa_ch}_mean'] - intensity_scaling_param[0]*intensity_profile[f'ch{major_fa_ch}_std']
    # vmax = intensity_profile[f'ch{major_fa_ch}_mean'] + intensity_scaling_param[1]*intensity_profile[f'ch{major_fa_ch}_std']
    
    # x_clipped = np.clip(input_image, vmin, vmax)
    # norm_input_image = (x_clipped - vmin) / (vmax - vmin)
    
    # fourtimes_norm_input_image: np.ndarray = norm_input_image * 1
    # fourtimes_norm_input_image[fourtimes_norm_input_image > 1] = 1



    intensity_incell = input_image[input_cell_mask>0]
    vmin = np.percentile(intensity_incell,0.2)
    vmax = np.percentile(intensity_incell,99.8)
    fourtimes_norm_input_image = (input_image-vmin)/(vmax-vmin)
    fourtimes_norm_input_image[fourtimes_norm_input_image<0]=0
    fourtimes_norm_input_image[fourtimes_norm_input_image>1]=1


    
    label_FA_obj_forplot = label_FA_obj.copy()
    
    label_FA_obj_forplot[0,0]=200001
    
    # Prepare the color map for label overlay
    pax_image_label_overlay: np.ndarray = label2rgb(
        label_FA_obj_forplot, image=fourtimes_norm_input_image, kind='overlay', alpha=0.75, colors=extended_colors.colors
    )
    
    for_orent_mask = binary_opening(input_cell_mask, disk(11))
    for_orent_distance_taxicab = distance_transform_cdt(for_orent_mask, metric="taxicab")

    n_v = np.sin(local_orientation)   
    n_h = np.cos(local_orientation) 

    # find the orentation and directional vectors for each object at the centroid        
    # Initialize vectors for object orientation analysis
    num_labels = label_FA_obj.max()
    obj_X = np.zeros((num_labels, 1))
    obj_Y = np.zeros((num_labels, 1))
    cell_U = np.zeros((num_labels, 1))
    cell_V = np.zeros((num_labels, 1))
    obj_U = np.zeros((num_labels, 1))
    obj_V = np.zeros((num_labels, 1))
    cell_tan_U = np.zeros((num_labels, 1))
    cell_tan_V = np.zeros((num_labels, 1))

    for iL in range(num_labels):          

        obj_X[iL] = regionprops_pax[iL]['centroid'][0]
        obj_Y[iL] = regionprops_pax[iL]['centroid'][1]

        obj_U[iL] = np.sin(regionprops_pax[iL]['orientation'])
        obj_V[iL] = np.cos(regionprops_pax[iL]['orientation'])

        cell_edge_orient = local_orientation[int(obj_X[iL]),int(obj_Y[iL])]
        cell_tan_U[iL] = np.sin(cell_edge_orient)
        cell_tan_V[iL] = np.cos(cell_edge_orient)
        
        cell_U[iL] = n_v[int(obj_X[iL]),int(obj_Y[iL])]
        cell_V[iL] = n_h[int(obj_X[iL]),int(obj_Y[iL])]

    X, Y = np.meshgrid(np.arange(0,input_cell_mask.shape[1]), np.arange(0,input_cell_mask.shape[0]))
    # build a sparse grid and only within cell mask for quivering the directions
    grid_mask = np.zeros_like(input_cell_mask)
    grid_mask[::30,::30] = input_cell_mask[::30,::30]
    to_plot_X = X[grid_mask>0]
    to_plot_Y = Y[grid_mask>0]
    to_plot_U = n_v[grid_mask>0]
    to_plot_H = n_h[grid_mask>0]


    # plot the main outputs together for quick viewing
    fig, ax = plt.subplots(2,4, figsize=(16,8), dpi=256, facecolor='w', edgecolor='k')
    ax[0,0].imshow(fourtimes_norm_input_image, cmap=plt.cm.gray,vmax=1,vmin=0)
    ax[0,0].axis('off')
    ax[0,1].imshow(input_cell_mask, cmap=plt.cm.gray,vmax=0.3,vmin=0)
    ax[0,1].axis('off')

    # build a grid for contour plot

    ax[0,2].imshow(for_orent_distance_taxicab,cmap=plt.cm.gray)
    ax[0,2].contour(X, Y, for_orent_distance_taxicab,6,colors=('yellow','green', 'r','blue','cyan'),linewidths=0.7)
    ax[0,2].axis('off')        

    ax[0,3].imshow(for_orent_distance_taxicab,cmap=plt.cm.gray)
    ax[0,3].quiver(to_plot_X,to_plot_Y, -to_plot_U,to_plot_H,color='blue')
    ax[0,3].contour(X, Y, for_orent_distance_taxicab,6,linewidths=0.1)    
    ax[0,3].axis('off')


    ax[1,0].imshow(input_cell_mask, cmap=plt.cm.gray,vmax=0.1,vmin=0)
    ax[1,0].axis('off')
    

    # extended_colors_fit = extended_colors
    # extended_colors_fit.colors = extended_colors_fit.colors[:label_FA_obj.max()+1,:]
    
    ax[1,1].imshow(label_FA_obj*input_front_mask, cmap=extended_colors, interpolation='none',vmax = 200000 + 1,vmin = 0)
    ax[1,1].axis('off')
    
    ax[1,2].imshow(pax_image_label_overlay)
    ax[1,2].axis('off')
    
    ax[1,3].imshow(fourtimes_norm_input_image,cmap=plt.cm.gray,vmax=1,vmin=0)
    # quiver the orientation based on cell shape in blue
    ax[1,3].quiver(obj_Y,obj_X, -cell_U,cell_V,color='blue')
    # quiver the orientation of each object in magenta        
    ax[1,3].quiver(obj_Y,obj_X,  -obj_U,obj_V,color='m')         
    ax[1,3].axis('off')

    # Save plots
    if flag_plot_save:
        plt.savefig(os.path.join(plot_output_dir, 'panel_plot', f'panels_{filename}.png'))

        save_dirs = ['rgb_plot', 'contour',  'label_color', 'quiver_cell','quiver_obj']
        for idx, save_dir in enumerate(save_dirs, start=1):            
            extent = ax[idx % 2, idx // 2 + 1].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(os.path.join(plot_output_dir, save_dir, f'{save_dir}_{filename}.png'), bbox_inches=extent)
   
    if flag_run_all:
        plt.close(fig) 

    return pax_image_label_overlay
    