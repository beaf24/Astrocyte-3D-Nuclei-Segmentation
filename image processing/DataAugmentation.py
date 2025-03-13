import tensorflow as tf
import numpy as np
import os
import random
import copy
import glob
from scipy import ndimage

from Main.GetImage import *

# class DataAugmentation():

#     def __init__(self, dataset_path:str): 
#         assert os.path.exists(dataset_path)
#         self.dataset_path = dataset_path

#         self.originals_folder = f"{self.dataset_path}/nnunet"
#         self.labels_folder = f"{self.dataset_path}/nnunet/labels"

def augmentation(images_folder:str, labels_folder:str, size_augmented_dataset:int, probability:float, range_zoom:tuple, output_folder:str, probability_zoom:float=1):

    linked_images = {}

    print(glob.glob(f"{labels_folder}/*.gz"))

    for dataset_image in glob.glob(f"{labels_folder}/*.gz"):
        label_name = os.path.basename(dataset_image).split('.')[0]
        train_images = list(filter(lambda x: label_name in x, glob.glob(f"{images_folder}/*.gz")))

        linked_images[label_name] = {'label': dataset_image, 
                                            'images': train_images}

    print(linked_images.keys())
    print(list(linked_images.keys()))

    augmentation_counter = {k: 0 for k in linked_images.keys()}

    n_image = 0
    while n_image < size_augmented_dataset:
        image_name = [key for key in linked_images.keys()][random.randint(0, len(linked_images)-1)]
        counter = augmentation_counter[image_name] 
        augmentation_counter[image_name]+=1

        dataAugmentation(linked_images=linked_images, image_name=image_name, counter=counter, probability=probability, probability_zoom=probability_zoom, range_zoom=range_zoom, output_folder=output_folder)
        n_image+=1

        print(image_name, counter)

def dataAugmentation(linked_images:dict, image_name:np.array, counter:int, probability:float, probability_zoom:float, range_zoom:tuple, output_folder:str):
    dict_image = linked_images[image_name]
    images_set = {}
    print(dict_image)

    dataset_path = os.path.dirname(dict_image['label'])
    
    image_label = getData(image_path=dict_image['label'])

    images_set['label'] = image_label
    images_set['images'] = []

    for image in np.arange(len(dict_image['images'])):
        image_label = getData(image_path=dict_image['images'][image])

        images_set['images'].append(image_label)#.transpose((1, 2, 0)))
        
    
    print(images_set)

    print(f"shape: {images_set['label'].shape}")

    transformed_set = transform(images_set=images_set, probability=probability, probability_zoom=probability_zoom, range_zoom=range_zoom, shape=(512,512,-1))
    print(transformed_set['label'].shape)

    for image_type in dict_image.keys():
        if image_type == 'label':
            saveNifti(image=transformed_set['label'].transpose((2,1,0)),
                      metadata=None,
                      output_path=f"{output_folder}/labels/AG{counter}_{os.path.basename(dict_image['label'])}")
        else:
            for i, image in enumerate(transformed_set['images']):
                saveNifti(image=image.transpose((2,1,0)),
                            metadata=None,
                            output_path=f"{output_folder}/AG{counter}_{os.path.basename(dict_image['images'][i])}")


def transform(images_set:dict, probability:float, probability_zoom:float=1, range_zoom:tuple=(0.5,1,0.05), shape:tuple=(512,512,-1)):

    assert probability > 0 and probability <= 1

    # Left-Right flip
    if random.random() < probability:
        images_set["label"] = tf.image.flip_left_right(images_set["label"]).numpy()
        for image in np.arange(len(images_set['images'])):
            images_set["images"][image] = tf.image.flip_left_right(images_set["images"][image]).numpy()
    # Up-Down flip
    if random.random() < probability:
        images_set["label"] = tf.image.flip_up_down(images_set["label"]).numpy()
        for image in np.arange(len(images_set['images'])):
            images_set["images"][image] = tf.image.flip_up_down(images_set["images"][image]).numpy()
        # for key in transforming.keys():
        #     transforming[key] = tf.image.flip_up_down(transforming[key]).numpy()
    # 90º rotation
    if random.random() < probability:
        degrees = random.randint(1,4)
        images_set["label"] = tf.image.rot90(images_set["label"], degrees).numpy()
        for image in np.arange(len(images_set['images'])):
            images_set["images"][image] = tf.image.rot90(images_set["images"][image], degrees).numpy()
        # for key in transforming.keys():
        #     transforming[key] = tf.image.rot90(transforming[key], degrees).numpy()
    # zoom
    if random.random() <= probability_zoom:
        zoom = random.choice(np.arange(range_zoom[0], range_zoom[1], range_zoom[2]))
        # for key in transforming.keys():
        #     transforming[key] = tf.image.resize_with_crop_or_pad(transforming[key], size, size).numpy()
        size = int(images_set['label'].shape[1] * zoom)
        images_set["label"] = tf.image.resize_with_crop_or_pad(images_set["label"], size, size).numpy()
        for image in np.arange(len(images_set['images'])):
            size_i = int(images_set["images"][image].shape[1] * zoom)
            images_set["images"][image] = tf.image.resize_with_crop_or_pad(images_set["images"][image], size_i, size_i).numpy()

    # Standard final size
    images_set['label'] = ndimage.zoom(images_set['label'], 
                                       (shape[0]/images_set['label'].shape[0], shape[1]/images_set['label'].shape[1], 1)).astype(np.int32)

    for image in np.arange(len(images_set['images'])):
        print(images_set['images'][image].shape)
        print(images_set['images'][image].shape[1])
        images_set['images'][image] = ndimage.zoom(images_set['images'][image], 
                                                   (shape[0]/images_set['images'][image].shape[0], shape[1]/images_set['images'][image].shape[1], 1))
        
    return images_set


# if __name__ == '__main__':
#     # dataset = DataAugmentation(dataset_path="/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/datasets/Introduction__GFAP_DAPI")
#     dataset_path="/Users/Beatriz/Documents/Biomédica@IST/Mestrado/Tese/algorithms/nnU-Net/nnUNet_raw/Dataset005_AugmentedII__DAPI_GFAP"    
#     images_folder = f"{dataset_path}/nnunet"
#     labels_folder = f"{dataset_path}/nnunet/labels"
    
    
#     augmentation(images_folder, labels_folder, size_augmented_dataset=200, probability=0.4,output_folder=f"{dataset_path}/augmented/")