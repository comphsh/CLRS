import sys
from pathlib import Path


proj_root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(proj_root_dir))

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import yaml
import time
from argparse import Namespace
import nibabel as nib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from functools import partial


from math import ceil
from utils.test_metrics import cal_mean_std_median_quantile , cal_mean_std_median_quantile_total_inf_nan_valid
import ipdb
from tqdm import tqdm
from pred_each.pred_aux_tool import  cal_metric_dice_iou_hd_for_test
from pathlib import Path

def get_missing_modality_combination(dataname=""):
    ###modality missing mask
    #         flair   t1     t1c   t2
    if dataname == "BRATS2018":
        masks = [[True, False, False, False],
                 [False, True, False, False],
                 [False, False, True, False],
                 [False, False, False, True],
                 [True, True, False, False],
                 [True, False, True, False],
                 [True, False, False, True],
                 [False, True, True, False],
                 [False, True, False, True],
                 [False, False, True, True],
                 [True, True, True, False],
                 [True, True, False, True],
                 [True, False, True, True],
                 [False, True, True, True],
                 [True, True, True, True]]
        masks_torch = torch.from_numpy(np.array(masks))
        mask_name = [ 'flair',   #1000
                           't1',
                                't1ce',
                                      't2',

                      'flair_t1',
                      'flair_t1ce',
                      'flair_t2',

                      't1_t1ce',
                      't1_t2',
                      't1ce_t2',

                     'flair_t1_t1ce',
                     'flair_t1_t2',
                     'flair_t1ce_t2',
                     't1_t1ce_t2',
                     'flair_t1_t1ce_t2']
    elif dataname == "Prostate2012":
        masks = [[True, False],
                 [False, True],
                 [True, True]]
        masks_torch = torch.from_numpy(np.array(masks))
        mask_name = ['t2',  # 1000
                     'adc',
                     't2_adc']
    elif dataname == "Breast20":
        masks = [[True, False],
                 [False, True],
                 [True, True]]
        masks_torch = torch.from_numpy(np.array(masks))
        mask_name = ['preDCE',  # 1000
                     'DCE',
                     'preDCE_DCE']
    else:
        raise  ValueError('not found methodname')

    str_available_mode_list = []
    used_mode_list = []
    unused_modal_list = []
    for item in masks:
        indice = [str(index) for index , val in enumerate(item) if val==True]
        indice = ",".join(indice)
        str_available_mode_list.append(indice)

        indice_int = [index for index, val in enumerate(item) if val==True]
        used_mode_list.append(indice_int)

        indice_int = [index for index, val in enumerate(item) if val == False]
        unused_modal_list.append(indice_int)

    int_binary_available_mode_list = masks_torch.int()
    print("combinations=",str_available_mode_list)
    print("combinations=",mask_name)
    print("used=",int_binary_available_mode_list)
    print("used=",used_mode_list)
    print("unused=",unused_modal_list)
    return mask_name , masks , int_binary_available_mode_list , used_mode_list , unused_modal_list

def collate_fn(batch):

    batch_dict = {}


    for key in batch[0].keys():
        batch_dict[key] = torch.stack([d[key] for d in batch])
    return batch_dict


def flatten_dict_to_namespace(d, namespace=None):

    if namespace is None:
        namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):

            sub_namespace = flatten_dict_to_namespace(value)
            setattr(namespace, key, sub_namespace)
        else:

            setattr(namespace, key, value)
    return namespace

def load_yaml_as_args_Nesting(config_path):
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

    args = flatten_dict_to_namespace(cfg)
    return args , cfg

def get_order_list(num_modality , first_idx):
    if num_modality == 4:
        if first_idx == 0:
            # 输入顺序 ， 对应生成的id
            return [0,1,2,3] , [1,2,3,0]
        elif first_idx == 1:
            return [1,2,3,0] , [2,3,0,1]
        elif first_idx == 2:
            return [2,3,0,1] , [3,0,1,2]
        elif first_idx == 3:
            return [3,0,1,2] , [0,1,2,3]
        else:
            exit(-1)
    elif num_modality == 2:
        if first_idx == 0:
            return [0,1] ,[1,0]
        elif first_idx == 1:
            return [1,0] , [0,1]
        else:
            exit(-1)

gloal_fea = None
def cal_pred_Dice_IoU_HD_metric(args = None , model=None , model_spec_list=None, test_loader=None , image_savedir='' , dir_dict=None, method_root_path=None,
                                mask_name=None, masks=None , int_binary_available_mode_list=None, used_mode_list=None, unused_modal_list=None,  methodname='' ,
                                class_name_table=[],  model_param=None, is_visualization=False , is_circle=None):

    start_t = time.time()
    for ipos in tqdm(range(len(mask_name))):
        outimg_dir = os.path.join(image_savedir, mask_name[ipos], 'img')
        if not os.path.exists(outimg_dir):
            os.makedirs(outimg_dir)
        name_list = []
        dice_region_list = [[] for _ in range(args.num_cls)]
        iou_region_list = [[] for _ in range(args.num_cls)]
        hd_region_list = [[] for _ in range(args.num_cls)]

        one_epoch_t = time.time()
        for index, batch in enumerate(test_loader):
            image, label, name, affine, croppedsize = batch
            print("------image={} label={}  croppedsize={}".format(image.shape, label.shape , croppedsize))
            assert image.shape[2:] == label.shape[2:], print("image={} label={}".format(image.shape, label.shape))

            dir_dict['case_idx'] = str(index)
            dir_dict['case_name'] = name[0]
            dir_dict['missing_name'] = mask_name[ipos]

            # 0 表示缺失
            image[:, int_binary_available_mode_list[ipos] == 0, :, :, :] = 0

            with torch.no_grad():
                param_dict = {
                    "used_mode_list": used_mode_list[ipos],
                    "unused_modal_list": unused_modal_list[ipos],
                    "bool_available_modal_list": masks[ipos],
                    "is_circle": is_circle,
                    "model_spec_list": model_spec_list,
                    "model_param": model_param,
                }
                output_obj = predict_sliding_brats2018(args=args,net= model, img=image, croppedsize=croppedsize[0], methodname=methodname , param_dict=param_dict , root_path=method_root_path, dir_dict=dir_dict)
                prob_metric = output_obj[0]
                prob_show = output_obj[1]



            dice_region_list,iou_region_list, hd_region_list, name_list, save_pred_tor = cal_metric_dice_iou_hd_for_test(args=args, prob_metric=prob_metric, label=label, name_list=name_list, dice_region_list=dice_region_list, iou_region_list=iou_region_list, hd_region_list=hd_region_list, index=index,  name=name[0], prob_show=prob_show)

            seg_pred = save_pred_tor.cpu().numpy().transpose((1, 2, 0))
            seg_pred = seg_pred.astype(np.int32)
            seg_pred = nib.Nifti1Image(seg_pred, affine[0])

            seg_save_p = os.path.join('{}/{}.nii.gz' .format ( outimg_dir,  name[0] ))
            print("i={}  save {}".format(index + 1 , seg_save_p))
            nib.save(seg_pred, seg_save_p)



        msg_avg = ""
        for iij in range(len(dice_region_list)):
            msg_avg += f'Dice_{class_name_table[iij]} = {np.nanmean(dice_region_list[iij]) :.4f}, '
            msg_avg += f'IoU_{class_name_table[iij]} = {np.nanmean(iou_region_list[iij]) :.4f}, '
            msg_avg += f'HD_{class_name_table[iij]} = {np.nanmean(hd_region_list[iij]) :.4f}, '
        print('Available modality:{}  OneEpochTime:{}s  Average score: {}'.format(mask_name[ipos], time.time()-one_epoch_t , msg_avg))


        name_list.append('Mean')
        name_list.append('StdDev')
        name_list.append('Median')
        name_list.append('25quantile')
        name_list.append('75quantile')

        name_list.append('num_total')
        name_list.append('num_inf')
        name_list.append('num_nan')
        name_list.append('num_valid')

        for iij in range(len(dice_region_list)):
            dice_region_list[iij] = cal_mean_std_median_quantile_total_inf_nan_valid(dice_region_list[iij], handle_inf='ignore')

            iou_region_list[iij] = cal_mean_std_median_quantile_total_inf_nan_valid(iou_region_list[iij], handle_inf='ignore')
            hd_region_list[iij] = cal_mean_std_median_quantile_total_inf_nan_valid(hd_region_list[iij], handle_inf='ignore')


        results_data = {
            'Label' : name_list ,
        }


        for iij in range(len(dice_region_list)):
            key = f'Dice_{class_name_table[iij]}'
            val = dice_region_list[iij]
            results_data.update({key:val})

            key = f'IoU_{class_name_table[iij]}'
            val = iou_region_list[iij]
            results_data.update({key: val})

            key = f'HD_{class_name_table[iij]}'
            val = hd_region_list[iij]
            results_data.update({key: val})


        df = pd.DataFrame(results_data)

        csvpath = os.path.join( image_savedir , f'{mask_name[ipos]}' , 'seg_results.csv')
        df.to_csv(csvpath, float_format='%.6f' , index=False)
        print("saved csv {}".format(csvpath))

    print("all time = {:.4} sec".format(time.time() - start_t))



def load_data(args=None, methodname=None, proj_path=""):
    if args.dataname == 'BRATS2018':
        from data import BraTSDataSet_crop_norm_resize128
        val_dataset = BraTSDataSet_crop_norm_resize128.BraTSValDataSet(args.datapath, args.val_list,  args.input_size , proj_path=proj_path)
        val_loader = torch.utils.data.DataLoader(val_dataset,   batch_size=1,     num_workers=0,  drop_last=False,         shuffle=False,   pin_memory=True)

    elif args.dataname == 'Prostate2012':
        from data import ProstateDataSet_npy_Compose
        val_dataset = ProstateDataSet_npy_Compose.ProstateValDataSet(args.datapath, args.val_list , args.input_size, output_class_num=args.num_cls, proj_path=proj_path)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=0,
                                                 drop_last=False,
                                                 shuffle=False,
                                                 pin_memory=True)

    elif args.dataname == "Breast20_3m":
        from data import BreastDataSet_3m_Compose
        val_dataset = BreastDataSet_3m_Compose.BreastValDataSet(args.datapath, args.val_list , args.input_size, output_class_num=args.num_cls, proj_path=proj_path)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 num_workers=0,
                                                 drop_last=False,
                                                 shuffle=False,
                                                 pin_memory=True)
    else:
        raise NotImplementedError('not found dataname')

    return val_loader

def main():

    method_dir_list = [
        "",
    ]

    method_pth_list = [
        '.pth',
    ]



    metric_str = "dice_iou_hd"
    is_visualization = False
    is_circle = True
    proj_path = os.path.dirname(os.path.dirname(__file__))
    current_time = time.strftime("%Y_%m%d_%H%M%S", time.localtime(time.time()))
    for dir in method_dir_list:
        method_root_path = dir

        for pth_name_path in method_pth_list:
            pth_path = os.path.join(dir, pth_name_path)

            result_dir = os.path.dirname(pth_path)
            pth_name = pth_path.split('/')[-1].split('.')[0]
            image_savedir = os.path.join(result_dir, f'output_{pth_name}_{current_time}_{metric_str}')
            methodname = pth_path.split('/')[-3]
            config_path = os.path.join(result_dir, f'{methodname}_config.yaml')
            args, cfg = load_yaml_as_args_Nesting(config_path)

            dir_dict = {}
            dir_dict['pth_name'] = pth_name

            if not os.path.exists(image_savedir):
                os.makedirs(image_savedir)
            print("----args=",args)
            print("----pth={}".format(pth_path))
            print("----dataname={}".format(args.dataname))
            print("----methodname={}".format(methodname))
            print("----method_pth={}  {}".format(dir , pth_name_path))

            mask_name, masks, int_binary_available_mode_list, used_mode_list, unused_modal_list = get_missing_modality_combination(dataname='BRATS2018')
            val_loader = load_data(args=args, methodname=methodname, proj_path=proj_path)
            from networks.model import CLRSNet
            model = CLRSNet()

            model = model.cuda()

            loaded_parameter = torch.load(pth_path)
            if 'model' in loaded_parameter:
                parameter_of_model = loaded_parameter['model']
                state_dict = parameter_of_model.state_dict()
            elif 'state_dict' in loaded_parameter:
                state_dict = loaded_parameter['state_dict']
            else:
                exit(-1)

            state_dict = {k.replace('module.' ,''):v for k , v in state_dict.items()}
            model.load_state_dict(state_dict , strict=True)
            model.eval()

            if args.dataname == "BRATS2018":
                class_name_table = {0: "ET", 1: 'WT', 2: "TC"}
            else:
                raise ValueError('not found data name')


            params_dict = {
                "args": args,
                "model": model,
                "test_loader": val_loader,
                "image_savedir": image_savedir,
                "dir_dict":dir_dict,
                "method_root_path": method_root_path,
                "mask_name": mask_name,
                "masks": masks,
                "int_binary_available_mode_list" : int_binary_available_mode_list,
                "used_mode_list": used_mode_list,
                "unused_modal_list": unused_modal_list,
                "methodname": methodname ,
                "is_visualization":is_visualization,
                "class_name_table":class_name_table,

                "model_spec_list": model_spec_list,
                "model_param": model_param,
                "is_circle": is_circle,

            }

            cal_pred_Dice_IoU_HD_metric(**params_dict)
            print('\n\n')



if __name__ == '__main__':
    pass

    main()

