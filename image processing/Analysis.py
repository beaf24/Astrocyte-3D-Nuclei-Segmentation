import os

from alive_progress import alive_bar

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

os.chdir(os.getcwd())
# from Postprocessing import*

# from Postprocessing import Postprocessing, masks_to_boxes


def compare_tuples(pred, msk, div=1):
    size = 512//div

    predictions, masks = list(np.unique(pred)), list(np.unique(msk))
    predictions.remove(0), masks.remove(0)
    pred_dict = dict()
    msk_dict = dict()

    pred = pred.reshape((-1, 512, 512))
    msk = msk.reshape((-1, 512, 512))

    for p in predictions:
        pred_p = np.sign(np.where(pred==p, 1, 0).sum(axis=0))
        pred_p_reshaped = pred_p.reshape(-1,size,size)

        for k in [i for i, n in enumerate(list(np.sign(pred_p_reshaped.sum(axis=(1,2))))) if n == True]:
            if k in pred_dict.keys():
                pred_dict[k].append(p)
            else:
                pred_dict[k] = list([p])

    for m in masks:
        msk_m = np.sign(np.where(msk==m, 1, 0).sum(axis=0))
        msk_m_reshaped = msk_m.reshape(-1,size,size)

        for k in [i for i, n in enumerate(list(np.sign(msk_m_reshaped.sum(axis=(1,2))))) if n == True]:
            if k in msk_dict.keys():
                msk_dict[k].append(m)
            else:
                msk_dict[k] = list([m])

    combinations = list()
    combinations.append([(x,y) for key in np.intersect1d(list(msk_dict.keys()),list(pred_dict.keys())) for x in pred_dict[key] for y in msk_dict[key] ])

    return set(combinations[0])

def consecutive_labels(input):
    labels = list(np.unique(input))
    labels.remove(0)

    output = copy.deepcopy(input)

    for i, l in enumerate(labels):
        output = np.where(output == l, i+1, output)

    return output

def load_arch_images(arch, arch_path, image_name, fold, merge = None):

    if arch == 'nnUNet':
        # Images paths
        predicted_image_path = f"{arch_path}/nnUNet_results/{dataset}/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/{fold}/validation/{image_name}.nii.gz"
        test_image_path = f"{arch_path}/nnUNet_raw/{dataset}/imagesTr/{image_name}_0000.nii.gz"
        groundtruth_image_path = f"{arch_path}/nnUNet_raw/Dataset001_ActinGFP/labelsIns/{image_name}.tif"

        logging.info("Loading images...")
        # Load images
        test_load = nib.load(test_image_path)
        test_image = test_load.get_fdata()
        nii_aff  = test_load.affine
        nii_hdr  = test_load.header
        scale = test_load.header['pixdim'][1:4]
        x_ani, y_ani, z_ani = scale

        groundtruth_image = tifffile.imread(groundtruth_image_path)
        groundtruth_image = np.transpose(groundtruth_image, (2,1,0))
        groundtruth_image = consecutive_labels(groundtruth_image)

        predicted_image = postprocessing_instance_segmentation(predicted_image_path)
        predicted_image = consecutive_labels(predicted_image)


    elif arch == 'stardist':
        # Images paths
        predicted_image_path = f"{arch_path}/data/{fold}/results/{image_name}.tif"
        test_image_path = f"{arch_path}/data/{fold}/test/images/{image_name}.tif"
        groundtruth_image_path = f"{arch_path}/data/{fold}/test/masks/{image_name}.tif"

        test_image = tifffile.imread(test_image_path)

        predicted_image = tifffile.imread(predicted_image_path)
        predicted_image = consecutive_labels(predicted_image)

        groundtruth_image = tifffile.imread(groundtruth_image_path)
        groundtruth_image = consecutive_labels(groundtruth_image)

        scale = [1,1,1]

    elif arch == 'cellstitch':
        # Images paths
        predicted_image_path = f"{arch_path}/data/{dataset}/{fold}/results/{merge}/{image_name}.npy"
        test_image_path = f"{arch_path}/data/{dataset}/train/images/{image_name}.npy"
        groundtruth_image_path = f"{arch_path}/data/{dataset}/train/masks/{image_name}.npy"

        logging.info("Loading images...")
        # Load images
        test_image = np.load(test_image_path)

        groundtruth_image = np.load(groundtruth_image_path)
        groundtruth_image = consecutive_labels(groundtruth_image)

        predicted_image = np.load(predicted_image_path)
        predicted_image = consecutive_labels(predicted_image)

        scale = [1,1,1]

    return test_image, groundtruth_image, predicted_image

def analysis(arch, dataset, image_name, arch_path, fold, merge=None, scale = [1,1,1]):
    analysis = dict()
    analysis['name'] = image_name
    analysis['dataset'] = dataset

    test_image, groundtruth_image, predicted_image = load_arch_images(arch, arch_path, image_name, fold, merge)

    n_preds = len(list(np.unique(predicted_image)))
    n_gt_masks = len(list(np.unique(groundtruth_image)))

    logging.info(f"{test_image.shape}, {groundtruth_image.shape}, {predicted_image.shape}")
    print(f"{test_image.shape}, {groundtruth_image.shape}, {predicted_image.shape}")

    # plt.figure()
    # plt.imshow(groundtruth_image.sum(axis=2))
    # plt.title(f"Ground truth {image_name}")

    analysis['groundtruth_image'] = groundtruth_image
    analysis['predicted_image'] = predicted_image
    # analysis['test_image'] = test_image

    logging.info("Extrating image properties...")
    print("Extrating image properties...")
    # Image properties
    props_ground_truth = measure.regionprops_table(groundtruth_image, intensity_image=test_image,
                                                   spacing=scale,
                                                   properties=['label', 'num_pixels',
                                                               'area', 'equivalent_diameter',
                                                               'mean_intensity'])
    df_props_ground_truth = pd.DataFrame(props_ground_truth)
    df_props_ground_truth = df_props_ground_truth.set_index('label')

    df_stats_props_ground_truth = pd.DataFrame([df_props_ground_truth.mean(),
                                                df_props_ground_truth.std(),
                                                df_props_ground_truth.min(),
                                                df_props_ground_truth.max()],
                                                index = ['mean', 'std', 'min', 'max'])

    analysis['groudtruth_properties'] = {'properties': df_props_ground_truth,
                                         'statistics': df_stats_props_ground_truth}

    props_prediction = measure.regionprops_table(predicted_image, intensity_image=test_image,
                                                 spacing=scale,
                                                 properties=['label', 'num_pixels',
                                                             'area', 'equivalent_diameter',
                                                             'mean_intensity'])
    df_props_prediction = pd.DataFrame(props_prediction)
    df_props_prediction = df_props_prediction.set_index('label')

    df_stats_props_prediction = pd.DataFrame([df_props_prediction.mean(),
                                              df_props_prediction.std(),
                                              df_props_prediction.min(),
                                              df_props_prediction.max()],
                                              index = ['mean', 'std', 'min', 'max'])

    analysis['prediction_properties'] = {'properties': df_props_prediction,
                                         'statistics': df_stats_props_prediction}

    logging.info("Computing IoU...")
    print("Computing IoU...")
    # Compute IoU

    logging.info("Calculating combinations...")
    print("Calculating combinations...")
    combinations = compare_tuples(groundtruth_image, predicted_image)
    logging.info(f"Total of {len(combinations)} combinations to compare")
    print(f"Total of {len(combinations)} combinations to compare")

    iou_global = np.zeros((n_preds, n_gt_masks))

    logging.info("Starting calculating the IoUs")
    print("Starting calculating the IoUs")
    i=0
    with alive_bar(len(combinations), force_tty = True) as bar:
        for (p,g) in combinations:
            prediction = np.where(predicted_image==p, 1, 0)
            groudtruth = np.where(groundtruth_image==g, 1, 0)

            z_prediction = np.argwhere(prediction==1)[:, 2]
            z_groudtruth = np.argwhere(groudtruth==1)[:, 2]

            x_prediction = np.argwhere(prediction==1)[:, 0]
            x_groudtruth = np.argwhere(groudtruth==1)[:, 0]

            y_prediction = np.argwhere(prediction==1)[:, 1]
            y_groudtruth = np.argwhere(groudtruth==1)[:, 1]

            if len(np.intersect1d(z_prediction, z_groudtruth)) != 0 and \
                len(np.intersect1d(x_prediction, x_groudtruth)) != 0 and \
                len(np.intersect1d(y_prediction, y_groudtruth)) != 0:
                iou_global[p,g] = metrics.jaccard_score(groudtruth.flatten(), prediction.flatten(), average = None)[1]

            logging.info(bar())
            i+=1
            print(f"{i}/{len(combinations)}")

    logging.info("Computing other metrics...")
    print("Computing other metrics...")
    iou_global_labels = iou_global[1:, 1:]

    n_preds, n_masks = iou_global_labels.shape
    ratio = n_preds/n_masks

    mean_iou = iou_global_labels[iou_global_labels!=0].mean()
    sign_iou = np.sign(iou_global_labels)

    correspondence = np.argwhere(sign_iou == 1)
    groudtruth_correspondence = sign_iou.sum(axis=0)
    predictions_correspondence = sign_iou.sum(axis=1)

    fp = list(predictions_correspondence).count(0)
    tp = list(predictions_correspondence).count(1)
    fn = list(groudtruth_correspondence).count(0)

    prediction_values = np.unique(predictions_correspondence).astype('int')

    if np.any(prediction_values >= 2) != 0:
        overpredictions = dict()
        overpredictions['total'] = len(predictions_correspondence) - tp - fp
        overpredictions['count'] = dict()

        for value in prediction_values:
            if value >=2: overpredictions['count'][value] = list(predictions_correspondence).count(value)

    else: overpredictions = None


    analysis['masks_analysis'] = {'groudtruth': n_gt_masks,
                                  'predictions': n_preds,
                                  'correspondence': correspondence,   # (groundtruth,prediction)
                                  'ratio': ratio}

    analysis['iou'] = {'iou_global': iou_global,
                       'mean_iou': mean_iou}

    analysis['metrics'] = {'true_positives': tp,
                           'false_positives': fp,
                           'false_negatives': fn,
                           'overpredictions': overpredictions}
    logging.info(analysis)
    print(analysis)

    logging.info("Saving .pkl")
    print("Saving .pkl")

    if arch == 'nnUNet':
        saving_path = f'{arch_path}/nnUNet_results/{dataset}/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/analysis/{fold}'
    elif arch == 'stardist':
        saving_path = f'{arch_path}/data/analysis/{fold}'
    elif arch == 'cellstitch':
        saving_path = f'{arch_path}/data/{dataset}/analysis/{fold}/{merge}'

    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)

    with open(f'{saving_path}/{image_name}.pkl', 'wb') as fp:
        pickle.dump(analysis, fp)


import argparse
import sys
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Analysis of validation results',
                                     description='What the program does',
                                     epilog='Text at the bottom of help')

    parser.add_argument('-a', '--arch', choices = ['nnUNet', 'stardist', 'cellstitch'], required=True)
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-f', '--fold', nargs='*', default='*', required=False)
    parser.add_argument('-m', '--merge', choices = ['cellstitch', 'cellpose2d', 'cellpose3d'], default='cellstitch', required=False)
    parser.add_argument('-v', '--verbosity', action="count",
                        help="increase output verbosity (e.g., -vv is more than -v)")

    args = parser.parse_args()

    if args.verbosity:
        def _v_print(*verb_args):
            if verb_args[0] > (3 - args.verbosity):
                print(verb_args[1])
    else:
        _v_print = lambda *a: None  # do-nothing function

    global v_print
    dataset = args.dataset
    arch_path = args.path
    arch = args.arch

    logger = logging.getLogger(__name__)
    logger_filename = f"analysis_{dataset}.log"
    logging.basicConfig(filename=logger_filename, encoding='utf-8', level=logging.INFO)

    logging.info(f'Architecture: {arch}\nDataset: {dataset}')

    print("I am alive!")
    print(f'Architecture: {arch}\nDataset: {dataset}')
    print(f"Fold: {args.fold}\nMerge: {args.merge}")

    if args.fold == '*':

        if arch == 'nnUNet':
            validation_path = f"{arch_path}/nnUNet_results/{dataset}/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/*/validation/*.npz"
        elif arch == 'stardist':
            validation_path = f"{arch_path}/data/*/results/*.tif"
        elif arch == 'cellstitch':
            validation_path = f"{arch_path}/data/{dataset}/essay*/results/*/*.npy"

        logging.info(validation_path)
        print(validation_path)

        for relative_filepath in glob.glob(validation_path):
            if arch == 'nnUNet' or arch == 'stardist':
                fold = relative_filepath.split('/')[-3]
                merge = None
            if arch == 'cellstitch':
                merge = relative_filepath.split('/')[-2]
                fold = relative_filepath.split('/')[-4]

            image_name = os.path.basename(relative_filepath).split('.')[0]
            logging.info(f"Starting analysis of {image_name}")
            print(f"Starting analysis of {image_name}")
            analysis(arch = arch, dataset=dataset, image_name=image_name, arch_path=arch_path, fold=fold, merge=merge)

            logging.info("\n")

    else:
        for i in args.fold:
            fold = i
            if arch == 'nnUNet':
                validation_path = f"{arch_path}/nnUNet_results/{dataset}/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_{args.fold[0]}/validation/*.npz"
            elif arch == 'stardist':
                validation_path = f"{arch_path}/data/fold_{args.fold[0]}/results/*.tif"
            elif arch == 'cellstitch':
                validation_path = f"{arch_path}/data/{dataset}/{args.fold[0]}/results/{args.merge}/*.npy"

            logging.info(validation_path)
            print(validation_path)
            print(glob.glob(validation_path))

            for relative_filepath in glob.glob(validation_path):
                if arch == 'nnUNet' or arch == 'stardist':
                    fold = relative_filepath.split('/')[-3]
                elif arch == 'cellstitch':
                    fold = relative_filepath.split('/')[-4]

                image_name = os.path.basename(relative_filepath).split('.')[0]
                logging.info(f"Starting analysis of {image_name}")
                print(f"Starting analysis of {image_name}")
                analysis(arch = arch, dataset=dataset, image_name=image_name, arch_path=arch_path, fold=fold, merge=args.merge)

                logging.info("\n")                                                                      