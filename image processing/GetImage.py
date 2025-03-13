
# import tkinter as tk
# from tkinter import filedialog

# from aicspylibczi.CziFile import CziFile
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile
import xml.etree.ElementTree as ET
import numpy as np
# import czifile
# import xmltodict
import pathlib
import os
from imio import load, save
import copy
import nibabel as nib
import glob
import json
import ast
import tensorflow as tf

from aicsimageio.readers.tiff_reader import TiffReader
from aicsimageio.readers.czi_reader import CziReader

import cv2
from scipy import ndimage


def image_from_file(image_path:str):
    """get dataframe and metadata of a given image, from a file path"""

    assert os.path.isfile(image_path)

    print(f"Start get data")
    image = getData(image_path=image_path)

    print(f"Start get metadata")
    metadata = getMetadata(image_path=image_path, image=image)

    return image, metadata

def getData(image_path:str) -> np.ndarray:
    """
    Get image tensor
    """
    try:
        # Check if the file is .czi or .tif or .gz
        if not (image_path.endswith('.czi') or image_path.endswith('.tif') or image_path.endswith('.nii.gz')):
            raise ValueError("The file is not a .czi, .tif or .nii.gz file.")
        
        # Open the file
        if pathlib.Path(image_path).suffix == '.tif':
            tif = TiffReader(image_path)
            if len(tif.data.shape) == 4:
                image = tif.data.transpose((1,0,2,3))
            elif len(tif.data.shape) == 3:
                image = tif.data.transpose((0,1,2))

        elif pathlib.Path(image_path).suffix == '.czi':
            czi = CziReader(image_path)
            image = czi.data
        
        elif pathlib.Path(image_path).suffix == '.gz':
            nibdata = nib.load(image_path)
            image = nibdata.get_fdata()#.transpose((3,2,0,1))
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path {image_path} does not exist.")
    except ValueError as ve:
        raise ve
    
    return image
    
def getMetadata(image_path:str, image:np.array) -> dict:
    """
    scale: from file metadata (list)
    dimensions: in the right order from the file metadata (list)
    shape: from metadata (dict={dimension: size})
    markers: from folder name

    If the file is a .gz, it is expected that it already contains the metadata organized this way in the file header.
    """

    scale, dimensions, shape, markers = None, None, None, None

    if pathlib.Path(image_path).suffix == '.tif':
        tif = TiffReader(image_path)
        scale = [tif.physical_pixel_sizes.X, tif.physical_pixel_sizes.Y, tif.physical_pixel_sizes.Z]
        dimensions = list(tif.dims.order)
        shape = {}

        for d, dim in enumerate(dimensions):
            shape[dim] = tif.dims.shape[d]

        directory_path = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        if len(directory_path.split('__')) > 1:
            markers = [marker for marker in directory_path.split('__')[1].split('_')]

    
    elif pathlib.Path(image_path).suffix == '.czi':
        czi = CziReader(image_path)
        scale = [czi.physical_pixel_sizes.X, czi.physical_pixel_sizes.Y, czi.physical_pixel_sizes.Z]
        dimensions = list(czi.dims.order)
        shape = {}

        for d, dim in enumerate(dimensions):
            shape[dim] = czi.dims.shape[d]

        directory_path = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        if len(directory_path.split('__')) > 1:
            markers = [marker for marker in directory_path.split('__')[1].split('_')]

    elif pathlib.Path(image_path).suffix == '.gz':
        nibdata = nib.load(image_path)
        header = nibdata.header

        metadata = None
        for ext in header.extensions:
            if ext.get_code() == 44:  # 44 is the code for JSON data
                # metadata = ast.literal_eval(ext.get_content().decode('utf-8'))

                try:
                    metadata = ast.literal_eval(ext.get_content().decode('utf-8'))
                    scale = metadata['scale']
                    dimensions = metadata['dimensions']
                    shape = metadata['shape']
                    markers = metadata['markers']

                except ValueError as e:
                    print(f"Error evaluating content: {e}")
                    print(ext.get_content().decode('utf-8'))
                    scale, dimensions, shape, markers = None, None, None, None
                    
            else: 
                scale, dimensions, shape, markers = None, None, None, None
    
    metadata = {'scale': scale,
                'dimensions': dimensions,
                'shape': shape,
                'markers': markers}
    
    if metadata['dimensions'] != None:
        dimensions_order = dimensionCorrespondence(image=image, metadata=metadata)
        metadata['dimensions'] = dimensions_order

    print(metadata)

    if metadata['shape'] != None:
        if 'C' in metadata['shape'].keys():
            if metadata['markers'] == None:
                metadata['markers'] = [f"Marker {n_marker}" for n_marker 
                                       in np.arange(metadata['shape']['C'])]

            elif len(metadata['markers']) < metadata['shape']['C']:
                metadata['markers'].append([f"Marker{n_marker}" 
                                            for n_marker in np.arange(metadata['shape']['C']-len(metadata['markers']))])

    return metadata

def dimensionCorrespondence(image:np.array, metadata:dict):
    dimensions_order = ['C', 'Z', 'X', 'Y']

    squeezing_dimensions = [d for d, dim in enumerate(metadata['dimensions']) if dim not in dimensions_order]
    image = image.squeeze(axis=tuple(squeezing_dimensions))

    if len(image.shape) == 3:
        dimensions_order = ['Z', 'X', 'Y']

    for ele in reversed(squeezing_dimensions):
        metadata['shape'].pop(metadata['dimensions'][ele])
        metadata['dimensions'].pop(ele)

    index_correspondence = [dimensions_order.index(metadata['dimensions'][i]) for i in range(len(metadata['dimensions']))
                                                                              if metadata['dimensions'][i] in dimensions_order]
        
    image.transpose(tuple(index_correspondence))

    return dimensions_order


# def updateMetadata(self, metadata:dict):
#     keys = 'scale', 'shape', 'dimensions', 'markers'
#     new_metadata = {key: None for key in keys}

#     for key in metadata.keys():
#         new_metadata[key] = metadata[key]

#     self.metadata = new_metadata
#     return None

# def updateData(self, image:np.array):
#     self.image = image
#     return None
        

def saveNifti(image:np.array = None, metadata:dict = None, output_path:str = None) -> tuple:
    assert len(image.shape) == 4 or len(image.shape) == 3, "The image is not 3D!"

    if len(image.shape) == 4:
        nib_image = nib.nifti1.Nifti1Image(dataobj=image.transpose((3,2,1,0)),
                                           affine=None)
    elif len(image.shape) == 3:
        nib_image = nib.nifti1.Nifti1Image(dataobj=image.transpose((2,1,0)),
                                           affine=None)     

    ## Add metadata to extensions
    if metadata:   
        custom_metadata_json = json.dumps(metadata).encode('utf-8')
        extension = nib.nifti1.Nifti1Extension(44, custom_metadata_json)  # 44 is the code for JSON data
        nib_image.header.extensions.append(extension)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    nib.save(img=nib_image,
             filename=output_path)
    
# def saveNiftiXP(image:np.array = None, path:str = None, metadata:dict = None) -> tuple:
#     assert len(image.shape) == 4 or len(image.shape) == 3, "The image is not 3D!"

#     if len(image.shape) == 4:
#         nib_image = nib.nifti1.Nifti1Image(dataobj=image.transpose((3,2,1,0)),
#                                         affine=None)
#     elif len(image.shape) == 3:
#         nib_image = nib.nifti1.Nifti1Image(dataobj=image.transpose((2,1,0)),
#                                         affine=None)        

#     custom_metadata_json = json.dumps(metadata).encode('utf-8')
#     extension = nib.nifti1.Nifti1Extension(44, custom_metadata_json)  # 44 is the code for JSON data
#     nib_image.header.extensions.append(extension)
    
#     nib.save(img=nib_image,
#                 filename=path)

def resizeImage(image:np.array, metadata:dict, size:tuple, output_path:str, save:bool=False) -> tuple:
    """
    PARAMETERS:
    shape (tuple) - (x,y) dimensions of the resized image
    """

    original_shape = metadata['shape']
    original_scale = metadata['scale']

    scale = [original_scale[0] * original_shape['X']/size[0],
             original_scale[1] * original_shape['Y']/size[1],
             original_scale[2]]
    
    shape = {'Z': original_shape['Z'],
             'C': original_shape['C'],
             'X': size[0],
             'Y': size[1]}

    zoom = tuple([1, 
                  1, 
                  size[0]/original_shape['X'], 
                  size[1]/original_shape['X']])
            
    image_resized = ndimage.zoom(image, zoom, order=1)

    metadata_resized = copy.deepcopy(metadata)
    metadata_resized['scale'] = scale
    metadata_resized['shape'] = shape

    if save:
        saveNifti(image=image_resized,
                  metadata=metadata_resized,
                  output_path=output_path)

    return image_resized, metadata_resized

def cropImage(image:np.array, metadata:dict, box:dict, output_path:str=None, save:bool=False) -> tuple:

    original_shape = metadata['shape']

    for key in box.keys():
        assert key in original_shape.keys(), \
                "The cropping dimensions do not correspond to the image dimensions"

    for dim in box.keys(): assert len(box[dim]) == 2 and isinstance(box[dim], (list, tuple, np.ndarray)), \
                            f"Cropping in dimension {dim} is not in the right format."
        
    shape = copy.deepcopy(original_shape)

    for dim in ['X', 'Y', 'Z']:
        if dim not in box.keys(): 
            box[dim] = [0, int(original_shape[dim])]
        if box[dim][0] == -1 or box[dim][0] < 0: box[dim][0] = 0
        if box[dim][1] == -1 or box[dim][1] >= int(original_shape[dim]): box[dim][1] = int(original_shape[dim])
        shape[dim] = box[dim][1] - box[dim][0]

    image_cropped = image[:, box['Z'][0]:box['Z'][1],
                             box['X'][0]:box['X'][1], 
                             box['Y'][0]:box['Y'][1]]
    
    metadata_cropped = copy.deepcopy(metadata)
    metadata_cropped['shape'] = shape
    
    if output_path and save:
        saveNifti(image=image_cropped,
                  metadata=metadata_cropped,
                  output_path=output_path)

    return image_cropped, metadata_cropped

def channelImage(image:np.array, metadata:dict, channel=None, output_path:str=None, save:bool = False) -> tuple:
    original_shape = metadata['shape']
    markers = metadata['markers']

    print(markers, channel)

    if isinstance(channel, str):
        assert channel in markers, "The channel selected is not one of the markers."
        position = markers.index(channel)
    elif isinstance(channel, int):
        assert 0 <= channel < int(original_shape['C']), "The channel selected is not one of the markers."
        position = channel
    else:
        raise TypeError("Channel must be either a string or an integer.")

    # if path is None and save:
    #     path = f"{'.'.join(image_path.split('.')[:-1])}_000{position}.{image_path.split('.')[-1]}"

    channelImage = np.array(image[position, :, :, :], ndmin=4).squeeze()
    metadata = copy.deepcopy(metadata)

    metadata['markers'] = [markers[position]] if isinstance(channel, str) else metadata['markers'][position]
    metadata['shape']['C'] = 1

    if save:
        saveNifti(image=channelImage,
                  output_path=output_path,
                  metadata=metadata)
    
    return channelImage, metadata






# import tkinter as tk
# from tkinter import filedialog

# # from aicspylibczi.CziFile import CziFile
# from PIL import Image
# from PIL.TiffTags import TAGS
# import tifffile
# import xml.etree.ElementTree as ET
# import numpy as np
# import czifile
# import xmltodict
# import pathlib
# import os
# from imio import load, save
# import copy
# import nibabel as nib
# import glob
# import json
# import ast
# import tensorflow as tf

# from aicsimageio.readers.tiff_reader import TiffReader
# from aicsimageio.readers.czi_reader import CziReader

# import cv2
# from scipy import ndimage

# class GetImage():
#     def __init__(self, image=None, metadata=None, image_path:str = ''):
#         self.image_path = image_path
#         self.image = image
#         self.metadata = metadata

#         if image and metadata and image_path:
#             assert os.path.isfile(image_path)

#         elif image and image_path:
#             assert os.path.isfile(image_path)
#             self.getMetadata()

#         elif metadata and image_path:
#             assert os.path.isfile(image_path)
#             self.getData()

#         elif image_path:
#             assert os.path.isfile(image_path)
#             self.getData()
#             self.getMetadata()


#     def updateMetadata(self, metadata:dict):
#         keys = 'scale', 'shape', 'dimensions', 'markers'
#         new_metadata = {key: None for key in keys}

#         for key in metadata.keys():
#             new_metadata[key] = metadata[key]

#         self.metadata = new_metadata
    
#         return self.metadata
    
#     def updateData(self, image:np.array):
#         self.image = image
            
#     def getData(self) -> np.ndarray:
#         try:
#             # Check if the file is .czi or .tif or .gz
#             if not (self.image_path.endswith('.czi') or self.image_path.endswith('.tif') or self.image_path.endswith('.nii.gz')):
#                 raise ValueError("The file is not a .czi, .tif or .nii.gz file.")
            
#             # Open the file
#             if pathlib.Path(self.image_path).suffix == '.tif':
#                 tif = TiffReader(self.image_path)
#                 print(tif.data.shape)
#                 if len(tif.data.shape) == 4:
#                     self.image = tif.data.transpose((1,0,2,3))
#                 elif len(tif.data.shape) == 3:
#                     self.image = tif.data.transpose((0,1,2))

#             elif pathlib.Path(self.image_path).suffix == '.czi':
#                 czi = CziReader(self.image_path)
#                 self.image = czi.data
            
#             elif pathlib.Path(self.image_path).suffix == '.gz':
#                 nibdata = nib.load(self.image_path)
#                 self.image = nibdata.get_fdata()#.transpose((3,2,0,1))
        
#         except FileNotFoundError:
#             raise FileNotFoundError(f"The file at path {self.image_path} does not exist.")
#         except ValueError as ve:
#             raise ve
        
#         return self.image
        
    
#     def getMetadata(self) -> dict:
#         """
#         scale: from file metadata (list)
#         dimensions: in the right order from the file metadata (list)
#         shape: from metadata (dict={dimension: size})
#         markers: from folder name

#         If the file is a .gz, it is expected that it already contains the metadata organized this way in the file header.
#         """

#         scale, dimensions, shape, markers = None, None, None, None

#         if pathlib.Path(self.image_path).suffix == '.tif':
#             tif = TiffReader(self.image_path)
#             scale = [tif.physical_pixel_sizes.X, tif.physical_pixel_sizes.Y, tif.physical_pixel_sizes.Z]
#             dimensions = list(tif.dims.order)
#             shape = {}

#             for d, dim in enumerate(dimensions):
#                 shape[dim] = tif.dims.shape[d]

#             directory_path = os.path.basename(os.path.dirname(os.path.dirname(self.image_path)))
#             if len(directory_path.split('__')) > 1:
#                 markers = [marker for marker in directory_path.split('__')[1].split('_')]

        
#         elif pathlib.Path(self.image_path).suffix == '.czi':
#             czi = CziReader(self.image_path)
#             scale = [czi.physical_pixel_sizes.X, czi.physical_pixel_sizes.Y, czi.physical_pixel_sizes.Z]
#             dimensions = list(czi.dims.order)
#             shape = {}

#             for d, dim in enumerate(dimensions):
#                 shape[dim] = czi.dims.shape[d]

#             directory_path = os.path.basename(os.path.dirname(os.path.dirname(self.image_path)))
#             if len(directory_path.split('__')) > 1:
#                 markers = [marker for marker in directory_path.split('__')[1].split('_')]

#         elif pathlib.Path(self.image_path).suffix == '.gz':
#             nibdata = nib.load(self.image_path)
#             header = nibdata.header

#             metadata = None
#             for ext in header.extensions:
#                 if ext.get_code() == 44:  # 44 is the code for JSON data
#                     # metadata = ast.literal_eval(ext.get_content().decode('utf-8'))

#                     try:
#                         metadata = ast.literal_eval(ext.get_content().decode('utf-8'))
#                     except ValueError as e:
#                         print(f"Error evaluating content: {e}")
#                         print(ext.get_content().decode('utf-8'))

#                     scale = metadata['scale']
#                     dimensions = metadata['dimensions']
#                     shape = metadata['shape']
#                     markers = metadata['markers']
                
#                 else: 
#                     scale, dimensions, shape, markers = None, None, None, None
        
#         self.metadata = {'scale': scale,
#                         'dimensions': dimensions,
#                         'shape': shape,
#                         'markers': markers}
        
#         if self.metadata['dimensions'] != None:
#             self.dimensionCorrespondence()

#         if self.metadata['markers'] == None:
#             if self.metadata['shape'] != None:
#                 self.metadata['markers'] = [f"Marker {n_marker}" for n_marker in np.arange(self.metadata['shape']['C'])]

#         elif len(self.metadata['markers']) < self.metadata['shape']['C']:
#             self.metadata['markers'].append([f"Marker{n_marker}" 
#                                              for n_marker in np.arange(self.metadata['shape']['C']-len(self.metadata['markers']))])

#         return self.metadata
    
#     def dimensionCorrespondence(self):
#         dimensions_order = ['C', 'Z', 'X', 'Y']

#         squeezing_dimensions = [d for d, dim in enumerate(self.metadata['dimensions']) if dim not in dimensions_order]
#         self.image = self.image.squeeze(axis=tuple(squeezing_dimensions))

#         if len(self.image.shape) == 3:
#             dimensions_order = ['Z', 'X', 'Y']

#         for ele in reversed(squeezing_dimensions):
#             self.metadata['shape'].pop(self.metadata['dimensions'][ele])
#             self.metadata['dimensions'].pop(ele)

        
#         index_correspondence = [dimensions_order.index(self.metadata['dimensions'][i]) for i in range(len(self.metadata['dimensions']))
#                                                                                        if self.metadata['dimensions'][i] in dimensions_order]
            
#         self.image.transpose(tuple(index_correspondence))
#         self.metadata['dimensions'] = dimensions_order
    
#     def saveNifti(self, image:np.array = None, path:str = None, metadata:dict = None) -> tuple:
#         assert len(image.shape) == 4 or len(image.shape) == 3, "The image is not 3D!"

#         if len(image.shape) == 4:
#             nib_image = nib.nifti1.Nifti1Image(dataobj=image.transpose((3,2,1,0)),
#                                             affine=None)
#         elif len(image.shape) == 3:
#             nib_image = nib.nifti1.Nifti1Image(dataobj=image.transpose((2,1,0)),
#                                                affine=None)        

#         custom_metadata_json = json.dumps(metadata).encode('utf-8')
#         extension = nib.nifti1.Nifti1Extension(44, custom_metadata_json)  # 44 is the code for JSON data
#         nib_image.header.extensions.append(extension)
        
#         nib.save(img=nib_image,
#                  filename=path)
        
#     def saveNiftiXP(image:np.array = None, path:str = None, metadata:dict = None) -> tuple:
#         assert len(image.shape) == 4 or len(image.shape) == 3, "The image is not 3D!"

#         if len(image.shape) == 4:
#             nib_image = nib.nifti1.Nifti1Image(dataobj=image.transpose((3,2,1,0)),
#                                             affine=None)
#         elif len(image.shape) == 3:
#             nib_image = nib.nifti1.Nifti1Image(dataobj=image.transpose((2,1,0)),
#                                             affine=None)        

#         custom_metadata_json = json.dumps(metadata).encode('utf-8')
#         extension = nib.nifti1.Nifti1Extension(44, custom_metadata_json)  # 44 is the code for JSON data
#         nib_image.header.extensions.append(extension)
        
#         nib.save(img=nib_image,
#                  filename=path)
    
#     def resizeImage(self, size:tuple, path:str, save:bool) -> tuple:
#         """
#         PARAMETERS:
#         shape (tuple) - (x,y) dimensions of the resized image
#         """
#         original_shape = self.metadata['shape']
#         original_scale = self.metadata['scale']

#         scale = [original_scale[0] * original_shape['X']/size[0],
#                  original_scale[1] * original_shape['Y']/size[1],
#                  original_scale[2]]
        
#         shape = {'Z': original_shape['Z'],
#                  'C': original_shape['C'],
#                  'X': size[0],
#                  'Y': size[1]}

#         zoom = tuple([1, 
#                       1, 
#                       size[0]/original_shape['X'], 
#                       size[1]/original_shape['X']])
                
#         resized_image = ndimage.zoom(self.image, zoom, order=1)

#         metadata = copy.deepcopy(self.metadata)
#         metadata['scale'] = scale
#         metadata['shape'] = shape

#         if save:
#             self.saveNifti(image=resized_image,
#                            path=path,
#                            metadata=metadata)

#         return resized_image, metadata

#     def cropImage(self, box:dict, path:str = None, save:bool = False) -> tuple:

#         original_shape = self.metadata['shape']

#         for key in box.keys():
#             assert key in original_shape.keys(), \
#                    "The cropping dimensions do not correspond to the image dimensions"

#         for dim in box.keys(): assert len(box[dim]) == 2 and isinstance(box[dim], (list, tuple, np.ndarray)), \
#                                f"Cropping in dimension {dim} is not in the right format."
            
#         shape = copy.deepcopy(original_shape)

#         for dim in ['X', 'Y', 'Z']:
#             if dim not in box.keys(): 
#                 box[dim] = [0, int(original_shape[dim])]
#             if box[dim][0] == -1 or box[dim][0] < 0: box[dim][0] = 0
#             if box[dim][1] == -1 or box[dim][1] >= int(original_shape[dim]): box[dim][1] = int(original_shape[dim])
#             shape[dim] = box[dim][1] - box[dim][0]

#         cropped_image = self.image[:, box['Z'][0]:box['Z'][1],
#                                       box['X'][0]:box['X'][1], 
#                                       box['Y'][0]:box['Y'][1]]
        
#         metadata = copy.deepcopy(self.metadata)
#         metadata['shape'] = shape
        
#         if path and save:
#             self.saveNifti(image=cropped_image,
#                            path=path,
#                            metadata=metadata)

#         return cropped_image, metadata
    
#     def channelImage(self, channel=None, path:str = None, save:bool = False) -> tuple:
#         original_shape = self.metadata['shape']
#         markers = self.metadata['markers']

#         print(markers, channel)

#         if isinstance(channel, str):
#             assert channel in markers, "The channel selected is not one of the markers."
#             position = markers.index(channel)
#         elif isinstance(channel, int):
#             assert 0 <= channel < int(original_shape['C']), "The channel selected is not one of the markers."
#             position = channel
#         else:
#             raise TypeError("Channel must be either a string or an integer.")

#         if path is None and save:
#             path = f"{'.'.join(self.image_path.split('.')[:-1])}_000{position}.{self.image_path.split('.')[-1]}"

#         channelImage = np.array(self.image[position, :, :, :], ndmin=4).squeeze()
#         metadata = copy.deepcopy(self.metadata)

#         metadata['markers'] = [markers[position]] if isinstance(channel, str) else self.metadata['markers'][position]
#         metadata['shape']['C'] = 1

#         if save:
#             self.saveNifti(image=channelImage,
#                            path=path,
#                            metadata=metadata)
        
#         return channelImage, metadata


# if __name__ == '__main__':
#     i = 'save'
#     dataset =  "E2WT24H__Hoechst_CalceinAM_PI" #"ThomasADLH1L1__DAPI_GFAP" 
#                                           #"KathiHumanCells__Marker1_Marker2_Marker3_DAPI" 
#                                           #"BeatrizActinGFP__GFAP_SOX9_GFP_DAPI"

#     if i == -1:
#         # originals_folder = "/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/BeatrizActinGFP__GFAP_SOX9_GFP_DAPI/nnunet"

#         # for original_image in glob.glob(f"{originals_folder}/*"):
#         #     nibdata = nib.load(original_image)
#         #     image = nibdata.get_fdata()

#         #     print(image.shape)

#         originals_folder = "/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/BeatrizActinGFP__GFAP_SOX9_GFP_DAPI/originals"

#         for original_image in glob.glob(f"{originals_folder}/*"):
#             tif = TiffReader(original_image)
#             image = tif.data.transpose((1,0,2,3))

#     if i == 0:
#         dataset_folder = f"/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/{dataset}"
#         originals_folder = f"{dataset_folder}/originals"

#         for original_image in glob.glob(f"{originals_folder}/*"):
#             data = GetImage(original_image)


#             image = data.getData()
#             metadata = data.getMetadata()    
    
#     elif i == 'unet':
#         # dataset_folder = filedialog.askdirectory(title="Dataset directory.")
#         # dataset_folder = f"/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/{dataset}"
#         dataset_folder = "/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/algorithms/nnU-Net/nnUNet_raw/Dataset005_AugmentedII__DAPI_GFAP"
#         originals_folder = f"{dataset_folder}/originals"

#         nnUNet_training_shape = {'X': 512, 
#                                  'Y': 512}
        
#         dataset_json_path = f"{dataset_folder}/dataset.json"

#         with open(dataset_json_path) as file:
#             dataset_json = json.load(file)
        
#         for original_image in glob.glob(f"{originals_folder}/*.tif"):
#             data = GetImage(original_image)

#             image = data.getData()
#             metadata = data.getMetadata()

#             shape = metadata['shape']

#             for channel, marker in dataset_json['channel_names'].items():
#                 new_image_name = f"{original_image.split('.')[0]}_000{channel}.nii.gz"

#                 resized_image, metadata_resized = data.resizeImage((nnUNet_training_shape['X'], nnUNet_training_shape['Y']), 
#                                                                    f"{dataset_folder}/originals/{os.path.basename(new_image_name)}", 
#                                                                    False)
                
#                 resized = GetImage()
#                 resized.updateData(image=resized_image)
#                 resized.updateMetadata(metadata=metadata_resized)

#                 final_image = resized.channelImage(marker, 
#                                                    f"{dataset_folder}/nnunet/{os.path.basename(new_image_name)}", 
#                                                    True)

#     elif i == 'cropped':
#         # dataset_folder = filedialog.askdirectory(title="Dataset directory.")
#         dataset_folder = f"/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/{dataset}"
#         originals_folder = f"{dataset_folder}/originals"

#         nnUNet_training_shape = {'X': 512, 
#                                  'Y': 512}
        
#         dataset_json_path = f"{dataset_folder}/dataset.json"

#         with open(dataset_json_path) as file:
#             dataset_json = json.load(file)
        
#         for original_image in glob.glob(f"{originals_folder}/*"):
#             data = GetImage(original_image)

#             image = data.getData()
#             metadata = data.getMetadata()

#             shape = metadata['shape']

#             crops = [[]]

#             boxesX = np.ceil(int(shape['X'])/nnUNet_training_shape['X'])
#             boxesY = np.ceil(int(shape['Y'])/nnUNet_training_shape['Y'])

#             overlapX = np.int64(boxesX*nnUNet_training_shape['X']-int(shape['X']))/(boxesX-1)
#             overlapY = np.int64(boxesY*nnUNet_training_shape['Y']-int(shape['Y']))/(boxesY-1)

#             for boxX in np.arange(boxesX):
#                 for boxY in np.arange(boxesY):
#                     crop = {'X': [int(0 + boxX*(nnUNet_training_shape['X'] - overlapX)), 
#                                   int(nnUNet_training_shape['X'] + boxX*(nnUNet_training_shape['X'] - overlapX))],
#                             'Y': [int(0 + boxY*(nnUNet_training_shape['Y'] - overlapY)), 
#                                   int(nnUNet_training_shape['Y'] + boxY*(nnUNet_training_shape['Y'] - overlapY))]}
                                        
#                     # for channel in np.arange(int(shape['C'])):
#                     new_image_name = f"{original_image.split('.')[0]}_{str(int(boxX))}{str(int(boxY))}.nii.gz"
#                     cropped_image, metadata_cropped = data.cropImage(crop, f"{dataset_folder}/originals cropped/{os.path.basename(new_image_name)}", True)              


#     elif i == 'unet crop':
#         # dataset_folder = filedialog.askdirectory(title="Dataset directory.")
#         dataset_folder = f"/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/{dataset}"
#         originals_folder = f"{dataset_folder}/originals"

#         nnUNet_training_shape = {'X': 512, 
#                                  'Y': 512}
        
#         dataset_json_path = f"{dataset_folder}/dataset.json"

#         with open(dataset_json_path) as file:
#             dataset_json = json.load(file)
        
#         for original_image in glob.glob(f"{originals_folder}/*"):
#             data = GetImage(original_image)

#             image = data.getData()
#             metadata = data.getMetadata()

#             shape = metadata['shape']

#             crops = [[]]

#             boxesX = np.ceil(int(shape['X'])/nnUNet_training_shape['X'])
#             boxesY = np.ceil(int(shape['Y'])/nnUNet_training_shape['Y'])

#             overlapX = np.int64(boxesX*nnUNet_training_shape['X']-int(shape['X']))/(boxesX-1)
#             overlapY = np.int64(boxesY*nnUNet_training_shape['Y']-int(shape['Y']))/(boxesY-1)

#             for boxX in np.arange(boxesX):
#                 for boxY in np.arange(boxesY):
#                     crop = {'X': [int(0 + boxX*(nnUNet_training_shape['X'] - overlapX)), 
#                                   int(nnUNet_training_shape['X'] + boxX*(nnUNet_training_shape['X'] - overlapX))],
#                             'Y': [int(0 + boxY*(nnUNet_training_shape['Y'] - overlapY)), 
#                                   int(nnUNet_training_shape['Y'] + boxY*(nnUNet_training_shape['Y'] - overlapY))]}
                                        
#                     # for channel in np.arange(int(shape['C'])):
#                     new_image_name_1 = f"{original_image.split('.')[0]}_{str(int(boxX))}{str(int(boxY))}.nii.gz"
#                     cropped_image, metadata_cropped = data.cropImage(crop, f"{dataset_folder}/originals cropped/{os.path.basename(new_image_name_1)}", False)   

#                     cropped = GetImage()
#                     cropped.updateData(image=cropped_image)
#                     cropped.updateMetadata(metadata=metadata_cropped)

#                     for channel, marker in dataset_json['channel_names'].items():
#                         new_image_name = f"{new_image_name_1.split('.')[0]}_000{channel}.nii.gz"

#                         resized_image, metadata_resized = cropped.resizeImage((nnUNet_training_shape['X'], nnUNet_training_shape['Y']), 
#                                                                             f"{dataset_folder}/originals/{os.path.basename(new_image_name)}", 
#                                                                             False)
                        
#                         resized = GetImage()
#                         resized.updateData(image=resized_image)
#                         resized.updateMetadata(metadata=metadata_resized)

#                         final_image = resized.channelImage(marker, 
#                                                         f"{dataset_folder}/nnunet cropped/{os.path.basename(new_image_name)}", 
#                                                         True)

#     elif i == 'resized':
#     # dataset_folder = filedialog.askdirectory(title="Dataset directory.")
#         dataset_folder = "/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/BeatrizMiceActinGFP__GFAP_SOX9_GFP_DAPI"
#         originals_folder = f"{dataset_folder}/originals"

#         nnUNet_training_shape = {'X': 512, 
#                                  'Y': 512}
        
#         dataset_json_path = f"{dataset_folder}/dataset.json"

#         with open(dataset_json_path) as file:
#             dataset_json = json.load(file)
        
#         for original_image in glob.glob(f"{originals_folder}/*"):
#             data = GetImage(original_image)

#             image = data.getData()
#             metadata = data.getMetadata()

#             new_image_name = f"{original_image.split('.')[0]}.nii.gz"
#             resized_image, metadata_resized = data.resizeImage((512, 512), f"{dataset_folder}/originals cropped/{os.path.basename(new_image_name)}", True)

        
#     elif i == "save":
#         dataset_folder = "/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/algorithms/nnU-Net/nnUNet_raw/Dataset005_AugmentedII__DAPI_GFAP"
#         folder = f"{dataset_folder}/originals/labels"

#         dataset_json_path = f"{dataset_folder}/dataset.json"

#         with open(dataset_json_path) as file:
#             dataset_json = json.load(file)
        
#         for original_image in glob.glob(f"{folder}/*"):
#             data = GetImage(original_image)

#             image = data.getData()
#             metadata = None

#             print(type(image))

#             new_image_name = f"{original_image.split('.')[0]}.nii.gz"
#             data.saveNifti(image=image, 
#                              path=f"{dataset_folder}/nnunet/labels/{os.path.basename(new_image_name)}", 
#                              metadata=metadata)
