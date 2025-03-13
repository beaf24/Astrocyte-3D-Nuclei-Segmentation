"""
Postproceesing step, after nnUNet predictions, to create instance segmentation masks from semantic segmentation input
"""

from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_grayscale, pil_to_tensor, to_pil_image
from torchvision.transforms import ToTensor, Compose

import numpy as np
import pickle
import nibabel as nib

import cc3d
import copy
import cv2
import glob

from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pickle
from PIL import Image

from scipy.ndimage import convolve
from scipy import ndimage as ndi
from skimage import measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn import metrics

import sys

import tifffile
import torch
import torch.nn.functional as F


def extract_sub_array(array:np.ndarray, min_coords:tuple, max_coords:tuple) -> tuple:
    """
    Extract a sub-array from the given array using the bounding box coordinates.

    Parameters:
    array (np.ndarray): The input 3D array.
    min_coords (tuple): The minimum coordinates (x_min, y_min, z_min) of the bounding box.
    max_coords (tuple): The maximum coordinates (x_max, y_max, z_max) of the bounding box.

    Returns:
    np.ndarray: The extracted sub-array.
    """
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords

    # Ensure coordinates are within array bounds
    x_min, y_min, z_min = max(x_min-1, 0), max(y_min-1, 0), max(z_min-1, 0)
    x_max, y_max, z_max = min(x_max+1, array.shape[0]), min(y_max+1, array.shape[1]), min(z_max+1, array.shape[2])

    # Extract sub-array using slicing
    return array[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1], ((x_min, y_min, z_min), (x_max, y_max, z_max))

def max_pooling_3d_torch(array:np.ndarray, kernel_size:int, stride:int) -> np.ndarray:
    """
    Apply 3D max-pooling to the given array using PyTorch.

    Parameters:
    array (np.ndarray): The input 3D array.
    kernel_size (int): The size of the pooling kernel.
    stride (int): The stride of the pooling operation.

    Returns:
    np.ndarray: The result of the max-pooling operation.
    """
    # Convert the NumPy array to a PyTorch tensor
    array_tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Apply max pooling
    pooled_tensor = F.max_pool3d(array_tensor, kernel_size, stride, padding=1)

    # Convert the result back to a NumPy array
    pooled_array = pooled_tensor.squeeze().numpy()

    return pooled_array

def remove_dust(input:np.ndarray, values:list) -> np.ndarray:

    output = copy.deepcopy(input)

    for value in values:
        coords = np.argwhere(output == value)

        x_min, x_max = coords[:,0].min(), coords[:,0].max()
        y_min, y_max = coords[:,1].min(), coords[:,1].max()
        z_min, z_max = coords[:,2].min(), coords[:,2].max()

        output = np.where(output == value, 0, output)

        sub_output, coords_sub_output = extract_sub_array(output, (x_min, y_min, z_min), (x_max, y_max, z_max))

        x_min_sub, y_min_sub, z_min_sub = coords_sub_output[0]
        x_max_sub, y_max_sub, z_max_sub = coords_sub_output[1]

        convolved_sub_output = max_pooling_3d_torch(sub_output, kernel_size=3, stride=1)

        output[x_min_sub:x_max_sub+1, y_min_sub:y_max_sub+1, z_min_sub:z_max_sub+1] = convolved_sub_output

    return output

def postprocessing_instance_segmentation(nii_gz:str) -> np.ndarray:
    val_load = nib.load(nii_gz)
    prediction = val_load.get_fdata()
    nii_aff  = val_load.affine
    nii_hdr  = val_load.header
    #print(nii_aff ,'\n',nii_hdr)

    sure_bg = (prediction == 0).astype(np.uint8)
    sure_fg = (prediction == 2).astype(np.uint8)
    unknown = (prediction == 1).astype(np.uint8)

    # Find connected components in the sure foreground ?~@~T assign a label to each
    connected_components = cc3d.connected_components(sure_fg)
    # connected_components = cc3d.dust(sure_fg, threshold=10)

    # Create topological maps ?~@~T distance to background
    dist = ndi.distance_transform_edt(sure_fg)

    # Find local maxima of each connected component
    max_coords = peak_local_max(dist, labels=connected_components,
                                num_peaks_per_label=1,
                                footprint=np.ones((3, 3, 3)))
    local_maxima = np.zeros_like(sure_fg, dtype=bool)
    local_maxima[tuple(max_coords.T)] = True
    markers = ndi.label(local_maxima)[0]

    # Watershed segmentation
    instance_segmentation = watershed(connected_components, markers, mask=1-sure_bg)

    # Clean residues
    region_properties = measure.regionprops_table(instance_segmentation, properties=['label', 'num_pixels'])
    df_region_properties = pd.DataFrame(region_properties)
    df_region_properties = df_region_properties.set_index('label')

    threshold = 20
    dusty = [lbl for lbl in list(df_region_properties.index) if df_region_properties.loc[lbl, 'num_pixels'] < threshold]
    instance_segmentation_clean = remove_dust(np.int16(instance_segmentation), dusty)

    return instance_segmentation_clean

def semantic_segmentation_tensor(image:np.ndarray) -> np.ndarray:
    labels = list(np.unique(image))
    labels.remove(0)

    ss_image = np.zeros((len(labels), image.shape[0], image.shape[1]))

    for i, lbl in enumerate(labels):
        ss_image[i,:,:] = np.where(image == lbl, 1, 0)

    return ss_image

def thresholding(npz_file:str, threshold_foreground:float = 0.5, threshold_border:float = 0.5):
    data = np.load(npz_file)
    probability_map = data['probabilities']

    thresholded_prediction = np.zeros((probability_map.shape[1],
                                        probability_map.shape[2],
                                        probability_map.shape[3]))
    
    thresholded_prediction = np.where(probability_map[3] > threshold_border, 2, thresholded_prediction)
    thresholded_prediction = np.where(probability_map[2] > threshold_foreground, 1, thresholded_prediction)

    return thresholded_prediction


if __name__ == '__main__':

    dataset =  "BeatrizActinGFP__GFAP_SOX9_GFP_DAPI" #"ThomasADLH1L1__DAPI_GFAP" 
                                          #"KathiHumanCells__Marker1_Marker2_Marker3_DAPI" 
                                          #"BeatrizActinGFP__GFAP_SOX9_GFP_DAPI"

    dataset_folder = f"/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/{dataset}"
    predictions_folder = f"{dataset_folder}/prediction"

    # for probabilities_file in glob.glob(f"{predictions_folder}/*.npz"):
        # thresholded_prediction = Postprocessing.thresholding(probabilities_file)  .npz

    for semantic_labeling_file in glob.glob(f"{predictions_folder}/*.gz"):
        instance_segmentation_array = postprocessing_instance_segmentation(semantic_labeling_file)



