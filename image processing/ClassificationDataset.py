import numpy as np
import os
import pandas as pd

from GetImage import GetImage

# class ClassificationDataset():

#     def __init__(self, annotations_path:str, image_path:str, image_name:str = None):
#         annotations = pd.read_csv(annotations_path)
#         assert image_name in annotations["File Name"].unique()
#         self.image_name = image_name
#         self.bb_data = annotations[annotations["File Name"] == image_name]
#         instances = self.bb_data["Segmentation Number"].unique()

#         self.objimage = GetImage(image_path=image_path)


#         for instance in instances:
            
#             break

        

#         return None      



def maximum_intensity_projection(self, image):
    channel_dim = self.objimage.metadata["dimensions"].index("C")
    mip_image = np.max(image, axis=channel_dim)
    
    return mip_image

def get_voxel(self, instance_number:int):
    filter_instance = self.bb_data[self.bb_data["Segmentation Number"] == instance_number]

    zmin = min(filter_instance["Layer"])
    zmax = max(filter_instance["Layer"])

    xmin, ymin, xmax, ymax = np.inf, np.inf, 0, 0

    for _, x1, y1, _, x2, y2 in filter_instance["BoundingBox"]:
        xmin = min(xmin, x1)
        ymin = min(ymin, y1)
        xmax = max(xmax, x2)
        ymax = max(ymax, y2)

    cmax = self.objimage.metadata["shape"]["C"]

    voxel_dict = dict.fromkeys(self.objimage.metadata["dimensions"])


    voxel_dict['X'] = (xmin,xmax)
    voxel_dict['Y'] = (ymin,ymax)
    voxel_dict['Z'] = (zmin,zmax)
    voxel_dict['C'] = (1   ,cmax)
    
    voxel = self.objimage.image[voxel_dict[0][0]:voxel_dict[0][1],
                                voxel_dict[1][0]:voxel_dict[1][1],
                                voxel_dict[2][0]:voxel_dict[2][1],
                                voxel_dict[3][0]:voxel_dict[3][1]]
    
    return voxel
        