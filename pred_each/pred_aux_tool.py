from utils.test_metrics import calculate_metrics_only_dice_by_torch, calculate_metrics_dice_iou_hd95_by_torch
import logging
import torch

def cal_metric_for_validation(args=None , label_dict=None, pred_dict=None, num_cls=0):
    dice_region_list = []
    if args.dataname == "BRATS2018":

        seg_pred_tor_list = []
        seg_pred_tor_list.append(pred_dict['ET'])
        seg_pred_tor_list.append(pred_dict['WT'])
        seg_pred_tor_list.append(pred_dict['TC'])

        seg_gt_tor_list = []
        seg_gt_tor_list.append(label_dict['ET'])
        seg_gt_tor_list.append(label_dict['WT'])
        seg_gt_tor_list.append(label_dict['TC'])



        dice_ET_i = calculate_metrics_only_dice_by_torch(seg_pred_tor_list[0], seg_gt_tor_list[0])
        dice_WT_i = calculate_metrics_only_dice_by_torch(seg_pred_tor_list[1], seg_gt_tor_list[1])
        dice_TC_i = calculate_metrics_only_dice_by_torch(seg_pred_tor_list[2], seg_gt_tor_list[2])
        logging.info( '-------- Dice_ET = {}, Dice_WT = {}, Dice_TC = {}'.format(dice_ET_i, dice_WT_i, dice_TC_i))

        dice_region_list.append(dice_ET_i)
        dice_region_list.append(dice_WT_i)
        dice_region_list.append(dice_TC_i)

    elif args.dataname == "Prostate2012":

        if num_cls == 2:
            seg_gt_PZ_tor = label_dict['PZ']
            seg_gt_TZ_tor = label_dict['TZ']

            seg_pred_PZ_tor = pred_dict['PZ']
            seg_pred_TZ_tor = pred_dict['TZ']

            dice_PZ_i = calculate_metrics_only_dice_by_torch(seg_pred_PZ_tor, seg_gt_PZ_tor)
            dice_TZ_i = calculate_metrics_only_dice_by_torch(seg_pred_TZ_tor, seg_gt_TZ_tor)

            logging.info('-------- Dice_PZ = {}, Dice_TZ = {}'.format(dice_PZ_i, dice_TZ_i))

            dice_region_list.append(dice_PZ_i)
            dice_region_list.append(dice_TZ_i)
        else:
            seg_gt_PZ_tor = label_dict['PZ']
            seg_gt_TZ_tor = label_dict['TZ']
            seg_gt_BG_tor = label_dict['BG']

            seg_pred_PZ_tor = pred_dict['PZ']
            seg_pred_TZ_tor = pred_dict['TZ']
            seg_pred_BG_tor = pred_dict['BG']

            dice_PZ_i = calculate_metrics_only_dice_by_torch(seg_pred_PZ_tor, seg_gt_PZ_tor)
            dice_TZ_i = calculate_metrics_only_dice_by_torch(seg_pred_TZ_tor, seg_gt_TZ_tor)
            dice_BG_i = calculate_metrics_only_dice_by_torch(seg_pred_BG_tor, seg_gt_BG_tor)


            logging.info('-------- Dice_PZ = {}, Dice_TZ = {}, Dice_BG = {}'.format(dice_PZ_i, dice_TZ_i , dice_BG_i))

            dice_region_list.append(dice_PZ_i)
            dice_region_list.append(dice_TZ_i)
            dice_region_list.append(dice_BG_i)
    elif args.dataname == 'Breast20':


        seg_gt_BG_tor = label_dict['BG']
        seg_gt_Tumor_tor = label_dict['Tumor']
        seg_gt_Beast_tor = label_dict['Breast']


        seg_pred_BG_tor = pred_dict['BG']
        seg_pred_Tumor_tor = pred_dict['Tumor']
        seg_pred_Breast_tor = pred_dict['Breast']

        dice_BG_i = calculate_metrics_only_dice_by_torch(seg_pred_BG_tor, seg_gt_BG_tor)
        dice_Tumor_i = calculate_metrics_only_dice_by_torch(seg_pred_Tumor_tor, seg_gt_Tumor_tor)
        dice_Breast_i = calculate_metrics_only_dice_by_torch(seg_pred_Breast_tor, seg_gt_Beast_tor)

        logging.info('-------- Dice_BG = {}, Dice_Tumor = {}, Dice_Breast = {}'.format(dice_BG_i, dice_Tumor_i, dice_Breast_i))

        dice_region_list.append(dice_BG_i)
        dice_region_list.append(dice_Tumor_i)
        dice_region_list.append(dice_Breast_i)

    elif args.dataname == 'Breast20_3m':


        seg_gt_BG_tor = label_dict['BG']
        seg_gt_Tumor_tor = label_dict['Tumor']
        seg_gt_Beast_tor = label_dict['Breast']


        seg_pred_BG_tor = pred_dict['BG']
        seg_pred_Tumor_tor = pred_dict['Tumor']
        seg_pred_Breast_tor = pred_dict['Breast']

        dice_BG_i = calculate_metrics_only_dice_by_torch(seg_pred_BG_tor, seg_gt_BG_tor)
        dice_Tumor_i = calculate_metrics_only_dice_by_torch(seg_pred_Tumor_tor, seg_gt_Tumor_tor)
        dice_Breast_i = calculate_metrics_only_dice_by_torch(seg_pred_Breast_tor, seg_gt_Beast_tor)

        logging.info('-------- Dice_BG = {}, Dice_Tumor = {}, Dice_Breast = {}'.format(dice_BG_i, dice_Tumor_i, dice_Breast_i))

        dice_region_list.append(dice_BG_i)
        dice_region_list.append(dice_Tumor_i)
        dice_region_list.append(dice_Breast_i)
    else:
        raise ValueError('not found data name!!')

    return dice_region_list



def cal_metric_dice_iou_hd_for_test(args=None , prob_metric=None, label=None, name_list=None , dice_region_list=None, iou_region_list=None, hd_region_list=None, index=-1, name=None, prob_show=None):
    if args.dataname == "BRATS2018":

        seg_pred = torch.round(prob_metric).to(torch.uint8).cuda()

        seg_pred_list = []
        seg_pred_list.append(seg_pred[:, :, :, 0])
        seg_pred_list.append(seg_pred[:, :, :, 1])
        seg_pred_list.append(seg_pred[:, :, :, 2])

        seg_gt = label[0].to(torch.int32).cuda()
        seg_gt_list = []
        seg_gt_list.append(seg_gt[0, :, :, :])
        seg_gt_list.append(seg_gt[1, :, :, :])
        seg_gt_list.append(seg_gt[2, :, :, :])


        dice_ET_i, iou_ET_i, hd_ET_i = calculate_metrics_dice_iou_hd95_by_torch(seg_pred_list[0], seg_gt_list[0])
        dice_WT_i, iou_WT_i, hd_WT_i = calculate_metrics_dice_iou_hd95_by_torch(seg_pred_list[1], seg_gt_list[1])
        dice_TC_i, iou_TC_i, hd_TC_i = calculate_metrics_dice_iou_hd95_by_torch(seg_pred_list[2], seg_gt_list[2])
        print( 'i={} Processing {}: Dice_ET = {}, Dice_WT = {}, Dice_TC = {}'.format(index, name, dice_ET_i, dice_WT_i, dice_TC_i))
        name_list.append(name)

        dice_region_list[0].append(dice_ET_i)
        dice_region_list[1].append(dice_WT_i)
        dice_region_list[2].append(dice_TC_i)

        iou_region_list[0].append(iou_ET_i)
        iou_region_list[1].append(iou_WT_i)
        iou_region_list[2].append(iou_TC_i)

        hd_region_list[0].append(hd_ET_i)
        hd_region_list[1].append(hd_WT_i)
        hd_region_list[2].append(hd_TC_i)



        predshow = torch.round(prob_show).to(torch.uint8).cuda()
        save_pred_tor = torch.zeros_like(predshow[:, :, :, 0], dtype=torch.int32)
        save_pred_tor[predshow[:, :, :, 1] == 1] = 2
        save_pred_tor[predshow[:, :, :, 2] == 1] = 1
        save_pred_tor[predshow[:, :, :, 0] == 1] = 4

    elif args.dataname == "Prostate2012":

        seg_pred = torch.round(prob_metric).to(torch.uint8).cuda()
        seg_pred_PZ = seg_pred[:, :, :, 0]
        seg_pred_TZ = seg_pred[:, :, :, 1]

        seg_gt = label[0].to(torch.int32).cuda()
        seg_gt_PZ = seg_gt[0, :, :, :]
        seg_gt_TZ = seg_gt[1, :, :, :]

        dice_PZ_i, iou_PZ_i, hd_PZ_i = calculate_metrics_dice_iou_hd95_by_torch(seg_pred_PZ, seg_gt_PZ)
        dice_TZ_i, iou_TZ_i, hd_TZ_i = calculate_metrics_dice_iou_hd95_by_torch(seg_pred_TZ, seg_gt_TZ)

        print('i={} Processing {}: Dice_PZ = {}, Dice_TZ = {}'.format(index, name, dice_PZ_i, dice_TZ_i))

        name_list.append(name)
        dice_region_list[0].append(dice_PZ_i)  # 与class_name_table的顺序相呼应
        dice_region_list[1].append(dice_TZ_i)

        iou_region_list[0].append(iou_PZ_i)
        iou_region_list[1].append(iou_TZ_i)

        hd_region_list[0].append(hd_PZ_i)
        hd_region_list[1].append(hd_TZ_i)


        predshow = torch.round(prob_show).to(torch.uint8).cuda()
        save_pred_tor = torch.zeros_like(predshow[:, :, :, 0], dtype=torch.int32)
        save_pred_tor[predshow[:, :, :, 0] == 1] = 1
        save_pred_tor[predshow[:, :, :, 1] == 1] = 2

    elif args.dataname == 'Breast20':


        seg_pred = torch.round(prob_metric).to(torch.uint8).cuda()
        seg_pred_BG_tor = seg_pred[:, :, :, 0]
        seg_pred_Tumor_tor = seg_pred[:, :, :, 1]
        seg_pred_Breast_tor = seg_pred[:, :, :, 2]


        seg_gt = label[0].to(torch.int32).cuda()
        seg_gt_BG_tor = seg_gt[0, :, :, :]
        seg_gt_Tumor_tor = seg_gt[1, :, :, :]
        seg_gt_Beast_tor = seg_gt[2, :, :, :]


        dice_BG_i, iou_BG_i, hd_BG_i = calculate_metrics_dice_iou_hd95_by_torch(seg_pred_BG_tor, seg_gt_BG_tor)
        dice_Tumor_i, iou_Tumor_i, hd_Tumor_i = calculate_metrics_dice_iou_hd95_by_torch(seg_pred_Tumor_tor, seg_gt_Tumor_tor)
        dice_Breast_i, iou_Breast_i, hd_Breast_i = calculate_metrics_dice_iou_hd95_by_torch(seg_pred_Breast_tor, seg_gt_Beast_tor)

        print("GT={} Pred={}".format(seg_gt.unique() , seg_pred.unique()))
        print("GT num, 0:{}  1:{}  2:{}".format(torch.sum(seg_gt[0,...]) , torch.sum(seg_gt[1,...]) , torch.sum(seg_gt[2,...]) , ))
        print("Pred num, 0:{}  1:{}  2:{}".format(torch.sum(seg_pred[ ...,0]), torch.sum(seg_pred[ ...,1]), torch.sum(seg_pred[ ...,2]), ))

        print('i={} Processing {}: Dice_BG = {}, Dice_Tumor = {}, Dice_Breast = {}'.format(index, name, dice_BG_i, dice_Tumor_i, dice_Breast_i))

        name_list.append(name)
        dice_region_list[0].append(dice_BG_i)
        dice_region_list[1].append(dice_Tumor_i)
        dice_region_list[2].append(dice_Breast_i)

        iou_region_list[0].append(iou_BG_i)
        iou_region_list[1].append(iou_Tumor_i)
        iou_region_list[2].append(iou_Breast_i)

        hd_region_list[0].append(hd_BG_i)
        hd_region_list[1].append(hd_Tumor_i)
        hd_region_list[2].append(hd_Breast_i)

        predshow = torch.round(prob_show).to(torch.uint8).cuda()
        save_pred_tor = torch.zeros_like(predshow[:, :, :, 0], dtype=torch.int32)
        save_pred_tor[predshow[:, :, :, 0] == 1] = 0
        save_pred_tor[predshow[:, :, :, 2] == 1] = 2
        save_pred_tor[predshow[:, :, :, 1] == 1] = 1

    else:
        raise ValueError('not found data name!!')

    return dice_region_list, iou_region_list, hd_region_list, name_list, save_pred_tor