import numpy as np
import os

import torch

from scipy.spatial.distance import directed_hausdorff
from monai.metrics import HausdorffDistanceMetric

def calculate_metrics_numpy_by_np(prediction , ground_truth):

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)


    TP = np.sum(gt & pred)
    FP = np.sum(~gt & pred)
    TN = np.sum(~gt & ~pred)
    FN = np.sum(gt & ~pred)


    dice = 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) > 0 else 0


    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0


    if np.any(gt) and np.any(pred):
        hausdorff_distance = max(directed_hausdorff(np.argwhere(gt), np.argwhere(pred))[0], directed_hausdorff(np.argwhere(pred), np.argwhere(gt))[0])
    else:
        hausdorff_distance = float('inf')

    return dice, sensitivity, specificity, hausdorff_distance


def calculate_metrics_NoHD95_by_np(prediction , ground_truth):

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)


    TP = np.sum(gt & pred)
    FP = np.sum(~gt & pred)
    TN = np.sum(~gt & ~pred)
    FN = np.sum(gt & ~pred)


    dice = 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) > 0 else 0


    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0


    hausdorff_distance = 0.0

    return dice, sensitivity, specificity, hausdorff_distance

def calculate_metrics_NoHD95_by_torch(prediction, ground_truth):

    gt = ground_truth.bool()
    pred = prediction.bool()


    TP = torch.sum(gt & pred).item()
    FP = torch.sum(~gt & pred).item()
    TN = torch.sum(~gt & ~pred).item()
    FN = torch.sum(gt & ~pred).item()


    dice = 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) > 0 else 0


    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0


    hausdorff_distance = 0.0

    return dice, sensitivity, specificity, hausdorff_distance

def monai_hausdorff(pred, target , ratio=95):

    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=ratio, directed=False)

    pred_mask = pred.unsqueeze(0).unsqueeze(0).float()
    true_mask = target.unsqueeze(0).unsqueeze(0).float()
    hd_value = hd_metric(pred_mask, true_mask)


    hd_metric.reset()



    return hd_value.item()

def calculate_metrics_dice_iou_hd95_by_torch(prediction, ground_truth):


    pred = prediction.bool()
    gt = ground_truth.bool()


    TP = torch.sum(gt & pred).item()
    FP = torch.sum(~gt & pred).item()
    TN = torch.sum(~gt & ~pred).item()
    FN = torch.sum(gt & ~pred).item()


    dice = 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) > 0 else 0

    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0


    if not torch.any(gt) :
        if not torch.any(pred):
            hausdorff_distance = float(0.0)
        else:
            hausdorff_distance = float('nan')
    else:
        if not torch.any(pred):
            hausdorff_distance = float('nan')
        else:
            hausdorff_distance = monai_hausdorff(pred, gt, ratio=95)

    return dice, iou, hausdorff_distance

def calculate_metrics_only_dice_by_torch(prediction, ground_truth):

    gt = ground_truth.bool()
    pred = prediction.bool()


    TP = torch.sum(gt & pred).item()
    FP = torch.sum(~gt & pred).item()
    TN = torch.sum(~gt & ~pred).item()
    FN = torch.sum(gt & ~pred).item()


    dice = 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return dice

def calculate_metrics_only_dice_by_np(prediction , ground_truth):

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)


    TP = np.sum(gt & pred)
    FP = np.sum(~gt & pred)
    TN = np.sum(~gt & ~pred)
    FN = np.sum(gt & ~pred)


    dice = 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) > 0 else 0
    return dice



def cal_mean_std_median_quantile(val_list):

    mean = np.mean(val_list)
    std_dev = np.std(val_list)
    median = np.median(val_list)
    quantile_25 = np.percentile(val_list, 25)
    quantile_75 = np.percentile(val_list, 75)

    val_list.append(mean)
    val_list.append(std_dev)
    val_list.append(median)
    val_list.append(quantile_25)
    val_list.append(quantile_75)

    return val_list

def cal_mean_std_median_quantile_total_inf_nan_valid(val_list, handle_inf='ignore', max_hd=None):

    arr = np.array(val_list, dtype=np.float64)
    # 处理 inf
    if handle_inf == 'ignore':
        arr = arr[~np.isinf(arr)]
    elif handle_inf == 'replace':
        if max_hd is not None:
            arr = np.where(np.isinf(arr), max_hd, arr)
        else:
            raise ValueError('not set max hausdorff')
    else:
        pass

    num_total = len(val_list)
    num_inf = np.sum(np.isinf(arr))
    num_nan = np.sum(np.isnan(arr))
    num_valid = len(arr) - num_inf - num_nan


    mean = np.nanmean(arr) if num_valid > 0 else np.nan
    std_dev = np.nanstd(arr) if num_valid > 0 else np.nan
    median = np.nanmedian(arr) if num_valid > 0 else np.nan
    quantile_25 = np.nanpercentile(arr, 25) if num_valid > 0 else np.nan
    quantile_75 = np.nanpercentile(arr, 75) if num_valid > 0 else np.nan



    val_list.append(mean)
    val_list.append(std_dev)
    val_list.append(median)
    val_list.append(quantile_25)
    val_list.append(quantile_75)

    val_list.append(num_total)
    val_list.append(num_inf)
    val_list.append(num_nan)
    val_list.append(num_valid)

    return val_list