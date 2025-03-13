
import csv
import os

# import nibabel as nib
import glob
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import masks_to_boxes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import torch
import torch.nn.functional as F
import pandas as pd

import Postprocessing
from GetImage import getData, getMetadata

class CreateCSV:
    def __init__(self, dataset_path:str, output_csv_path:str = None):
        self.boundingboxes_csv = pd.DataFrame()

        if not output_csv_path:
            self.output_csv_path = "/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets_segmentation/CortexMaroussia__Marker1_GFP_DAPI_SA/CortexMaroussia_Model4/detected_regions_Maourssia-Model4.csv"
        else:
            self.output_csv_path = output_csv_path

        if not os.path.exists(self.output_csv_path):
            self.header = ["File Name",
                            "Layer",
                            "Segmentation Number", 
                            "Object Number",
                            "Area", 
                            "Centroid", 
                            "BoundingBox",
                            "Morphology Label",
                            "Multinuclei Label"]
            self.boundingboxes_csv = pd.DataFrame(columns=self.header)
        else:
            self.boundingboxes_csv = pd.read_csv(self.output_csv_path)
            self.header = list(self.boundingboxes_csv.columns)
        
        self.dataset_path = dataset_path
        # self.images_path = f"{self.dataset_path}/nnunet"
        # self.originals_path = "/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets_segmentation/Introduction__GFAP_DAPI/originals" #f"{self.dataset_path}/originals cropped"
        self.predictions_path = "/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets_segmentation/CortexMaroussia__Marker1_GFP_DAPI_SA/CortexMaroussia_Model4"#f"{self.dataset_path}/predictions"

        dataset_markers = None

        self.createCsv()

        
    def createCsv(self) -> None:

        preds_path = glob.glob(f"{self.predictions_path}/*.nii.gz")
        label_columns = set()

        print(preds_path)
            
        for predpath in preds_path:
            file_type = ''
            file_name = os.path.basename(predpath)
            img_name = os.path.basename(predpath).split('.')[0]

            print(img_name)
            
            if file_name.endswith('.nii.gz'):
                file_type = 'nii.gz'
                image0 = Postprocessing.postprocessing_instance_segmentation(nii_gz = predpath)
                image = image0.transpose((2,0,1))

            # original_image = getData(image_path=f"{self.originals_path}/{img_name}.tif")
            # metadata = getMetadata(image_path=f"{self.originals_path}/{img_name}.tif", image=original_image)
            # image_scale = metadata['scale']
            
            object_number = 0
            if file_type == 'nii.gz':
                for layer, msk in enumerate(image): 
                    ss_mask = Postprocessing.semantic_segmentation_tensor(msk)
                    tensor_mask = torch.from_numpy(ss_mask).type(torch.uint8)

                    boxes = masks_to_boxes(tensor_mask)

                    for b, bounding_box in enumerate(boxes): 
                        object_number += 1

                        x1, y1, x2, y2 = np.array(bounding_box, dtype=np.uint16)
                        bounding_box = (-1, x1, y1, -1, x2, y2)

                        area = (x2-x1)*(y2-y1)
                        centroid = (x1+(x2-x1)/2, y1+(y2-y1)/2)

                        if area:
                            new_entry = pd.DataFrame([{"File Name": img_name,
                                                           "Layer": layer,
                                                           "Segmentation Number": np.unique(msk[y1:y2,x1:x2])[-1], 
                                                           "Object Number": object_number, 
                                                           "Area": area, 
                                                           "Centroid": centroid, 
                                                           "BoundingBox": bounding_box,
                                                           "Morphology Label": 'Unknown',
                                                           "Multinuclei Label": 'Unknown'}])
                            
                            self.boundingboxes_csv = pd.concat([self.boundingboxes_csv, new_entry])
            
            # image_markers = metadata['markers']
            image_markers = ['GFP', 'FOXJ1', 'GFAP', 'DAPI']
            [label_columns.add(f"{image_markers[n_marker]} Label")
                            if n_marker < len(image_markers)
                            else label_columns.append(f"Marker{n_marker} Label") 
                            for n_marker in np.arange(4)]  #metadata['shape']['C']

        for new_column in list(set(label_columns) - set(self.boundingboxes_csv.columns) - {'DAPI Label'}):
            self.boundingboxes_csv[new_column] = 'Unknown'

        self.boundingboxes_csv.fillna('Unknown', inplace=True)

        self.boundingboxes_csv.to_csv(self.output_csv_path, index=False)

        print("Object-level information saved to CSV file.")

    def show(imgs:list) -> None:
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        plt.show()


if __name__ == '__main__':
    import sys

    sys.stdout.write("Start")
    CreateCSV(dataset_path="/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets_segmentation/CortexMaroussia__Marker1_GFP_DAPI_SA/CortexMaroussia_Model4")#,
              #output_csv_path="/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/BeatrizMiceActinGFP__GFAP_SOX9_GFP_DAPI/detected_regions_on_all_images.csv")