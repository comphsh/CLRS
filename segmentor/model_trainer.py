#coding=utf-8
import argparse
import os
import time
import setproctitle

import logging
import random
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from utils._my_losses import MyDice_Loss, MyBCE_or_CELoss

from networks.model import CLRSNet
from networks.synthesis import G_Model
from pred_each.pred_aux_tool import cal_metric_for_validation
from monai.losses import DiceCELoss
from utils.my_lr_scheduler import lr_schedule_election
from torch.cuda.amp import autocast as autocast
from torch.cuda import amp
scaler = amp.GradScaler(enabled=True)

def get_circle_modality(modal_num=1):
    if modal_num == 4:
        input_order_matrix = {
            0:[0,1,2,3],
            1: [1, 2, 3, 0],
            2: [2, 3, 0, 1],
            3: [3, 0, 1, 2],
        }
        supervised_order_matrix = {
            0: [1, 2, 3 , 0],
            1: [2, 3, 0 , 1],
            2: [3, 0, 1 , 2],
            3: [0, 1, 2 , 3],
        }

    elif modal_num == 3:
        input_order_matrix = {
            0: [0, 1, 2],
            1: [1, 2, 0],
            2: [2, 0, 1],

        }
        supervised_order_matrix = {
            0: [1, 2, 0],
            1: [2, 0, 1],
            2: [0, 1, 2],
        }
    elif modal_num == 2:
        input_order_matrix = {
            0: [0, 1],
            1: [1, 0],

        }
        supervised_order_matrix = {
            0: [1, 0],
            1: [0, 1],
        }
    else:
        raise ValueError('not found circle vector')

    return input_order_matrix , supervised_order_matrix


def update_weight_by_tanh(current_w, cur_iter, max_iter, k=1.0):

    if cur_iter > max_iter:
        cur_iter = max_iter

    tensor_val = torch.tensor(k * cur_iter / max_iter)
    delta = (1.0 - current_w) *  torch.tanh(tensor_val)
    return torch.clamp(current_w + delta, max=1.0)

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor

def train(args=None, train_loader=None, val_loader=None, writer=None, masks=None , mask_name=None, proj_name=''):
    ckpts = args.snapshot_dir

    model = CLRSNet(n_channels=args.num_modality, n_classes=args.num_cls, n_modality=args.num_modality, normalization=args.normalization , spec_normalization=args.spec_normalization, has_dropout=False, args=args)
    device = torch.device('cuda:{}'.format(args.local_rank))
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model.train()

    optimizer , lr_scheduler = lr_schedule_election(model , args)

    is_circle = True
    if is_circle == True:
        model_spec_list = []
        optimizer_spec_list = []
        lr_scheduler_spec_list = []
        for num in range(args.num_modality):
            model_spec = G_Model(in_channels=64, base_channels=64, p_dim=64)
            device = torch.device('cuda:{}'.format(args.local_rank))
            model_spec.to(device)
            model_spec = nn.parallel.DistributedDataParallel(model_spec, device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)
            model_spec.train()
            optimizer_spec, lr_scheduler_spec = lr_schedule_election(model_spec, args.aux)

            model_spec_list.append(model_spec)
            optimizer_spec_list.append(optimizer_spec)
            lr_scheduler_spec_list.append(lr_scheduler_spec)


    loss_dice_fn = MyDice_Loss(sigmoid=True)
    loss_ce_fn = MyBCE_or_CELoss(mode='BCE')
    loss_spec_modal_fn = nn.CrossEntropyLoss()
    loss_construct_fn = nn.MSELoss()

    gpu_list = [int(gpu_id) for gpu_id in args.num_gpus.split(',')]
    num_gpu = len(gpu_list)
    world_size = dist.get_world_size()

    print("----------------------###################---------------------\n"
          "=====================  GPU_num:{}  gpu_list:{}  world_size={}  current_local_rank={} =================\n"
          "----------------------###################---------------------\n".format(num_gpu , gpu_list, world_size, args.local_rank))

    input_order_matrix, supervised_order_matrix = get_circle_modality(args.num_modality)
    if args.num_modality == 4:
        spec_modal_label = torch.tensor([0, 1, 2 , 3]).cuda()
        mode_tot_list = [0, 1, 2, 3]
        mode_used_split = [0, 1, 2, 3]

        direction_list = ['flair->t1', 't1->t1c', 't1c->t2', 't2->flair']
    elif args.num_modality == 3:
        spec_modal_label = torch.tensor([0, 1, 2]).cuda()
        mode_tot_list = [0, 1, 2]  #
        mode_used_split = [0, 1, 2]  #
        direction_list = ['pDCE->DCE1', 'DCE1->DCE2', 'DCE2->pDCE']

    elif args.num_modality == 2:
        spec_modal_label = torch.tensor([0, 1]).cuda()
        mode_tot_list = [0, 1]
        mode_used_split = [0, 1]
        if args.dataname == "Prostate2012":
            direction_list = ['t2->ADC', 'ADC->t2']
        elif args.dataname == 'Breast20':
            direction_list = ['preDCE->DCE', 'DCE->preDCE']
        else:
            raise ValueError("not set direction")
    else:
        raise ValueError("not set spec modal label")


    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')

    iter_per_epoch = len(train_loader)
    max_iters = iter_per_epoch * args.num_epochs
    step = 1
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        setproctitle.setproctitle('{}/{}: {}/{}'.format(proj_name, args.method_name, epoch + 1, args.num_epochs))

        model.train()
        if is_circle == True:
            for iij in range(len(model_spec_list)):
                model_spec_list[iij].train()

        epoch_loss = 0.0
        epoch_iter = 0
        b = time.time()
        for i_iter , batch in enumerate(train_loader):

            images = torch.from_numpy(batch['image'])
            target = torch.from_numpy(batch['label'])
            images = images.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)

            _images, mode_used_split, used_mode_binary, first_index = random_mask_fn(images=images, mode_tot_list=mode_tot_list, num_modality=args.num_modality, epoch=epoch)

            N, C, D, W, H = _images.shape

            feature_list, generated_map, spec_fm_cat, spec_logtis_cat, spec_info_vector = model.module.forward_encoder(_images)

            total_loss = 0.0
            if is_circle == True:

                loss_synthsis_list , syntheis_image_dict = synthesize_modality(images=generated_map, model_spec_list=model_spec_list, loss_construct_fn=loss_construct_fn, input_order_matrix=input_order_matrix,
                                    supervised_order_matrix=supervised_order_matrix, first_index=first_index, args=args)

                loss_synthsis = 0.0
                for item in loss_synthsis_list:
                    loss_synthsis = loss_synthsis + item
                loss_synthsis = loss_synthsis / args.num_modality

                shared_map = torch.zeros((len(generated_map), generated_map[0].shape[1] , generated_map[0].shape[2] , generated_map[0].shape[3] , generated_map[0].shape[4])).cuda(args.local_rank, non_blocking=True)

                for iij in range(args.num_modality):

                    shared_map[iij] += syntheis_image_dict[iij][0].cuda(args.local_rank, non_blocking=True)

                shared_map = torch.cat((shared_map[0],shared_map[1],shared_map[2],shared_map[3] ), dim=0).unsqueeze(0).detach()

                feature_list[-1] += shared_map


                total_loss += loss_synthsis

                # for iij in range(len(optimizer_spec_list)):
                #     optimizer_spec_list[iij].zero_grad()
                # loss_synthsis.backward()
                # for iij in range(len(optimizer_spec_list)):
                #     optimizer_spec_list[iij].step()


            fuse_out_obj = model.module.forward_decoder(feature_list)

            if isinstance(fuse_out_obj , list):
                fuse_logits = fuse_out_obj[0]
            elif isinstance(fuse_out_obj , dict ):
                fuse_logits = fuse_out_obj['logits']
            else:
                exit('not found model ouput type!')

            modify_weight = [1.0, 1.0, 1.0, 1.0]

            fuse_cross_loss = loss_ce_fn(fuse_logits, target, num_cls=args.num_cls, weight=modify_weight)
            fuse_dice_loss = loss_dice_fn(fuse_logits, target, num_cls=args.num_cls, weight=modify_weight)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            epsilon_mat = torch.normal(mean=torch.zeros(spec_info_vector.shape), std=1.0).cuda()
            spec_modal_loss = loss_contrastive_modal_fn(spec_info_vector, epsilon_mat + spec_info_vector,  temperature=0.05)

            spec_logit_cls_loss = loss_spec_modal_fn(spec_logtis_cat, spec_modal_label)

            loss = fuse_loss  + args.coe_specloss * spec_modal_loss  + args.coe_consist * spec_logit_cls_loss

            total_loss += loss

            dist.barrier()
            reduce_total_loss = all_reduce_tensor(total_loss, world_size=world_size)
            reduce_loss = all_reduce_tensor(loss, world_size=world_size)  #
            reduce_fuse_loss = all_reduce_tensor(fuse_loss, world_size=world_size)
            reduce_fuse_cross_loss = all_reduce_tensor(fuse_cross_loss, world_size=world_size)
            reduce_fuse_dice_loss = all_reduce_tensor(fuse_dice_loss, world_size=world_size)
            reduce_spec_logit_cls_loss = all_reduce_tensor(spec_logit_cls_loss, world_size=world_size)
            reduce_spec_modal_loss = all_reduce_tensor(spec_modal_loss, world_size=world_size)
            reduce_loss_synthsis = all_reduce_tensor(loss_synthsis, world_size=world_size)


            if is_circle == True:
                for opt in optimizer_spec_list:
                    opt.zero_grad()
            optimizer.zero_grad()

            total_loss.backward()

            if is_circle == True:
                for opt in optimizer_spec_list:
                    opt.step()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_iter += 1

            if args.local_rank == args.print_rank:
                if step % 20 == 0:
                    cls_pred = F.sigmoid(fuse_logits)
                    overlap_pred = torch.round(cls_pred).long()
                    assert overlap_pred.shape == target.shape
                    val_overlap(args=args, fuse_pred=overlap_pred, label=target)

                writer.add_scalar('tot_loss', loss.item(), global_step=step)
                writer.add_scalar('loss_seg', fuse_loss.item(), global_step=step)
                writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
                writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)

                writer.add_scalar('spec_modal_loss', spec_modal_loss.item(), global_step=step)
                writer.add_scalar('spec_logit_cls_loss', spec_logit_cls_loss.item(), global_step=step)

                circlegenerate_loss_msg = ""
                if is_circle == True:
                    for ipos , loss in enumerate(loss_synthsis_list):
                        writer.add_scalar(f'loss_const_{direction_list[ipos]}', loss.item(), global_step=step)
                        circlegenerate_loss_msg += f"loss {direction_list[ipos]}:{loss.item():.6f}, "


                    logging.info('Epoch {}/{}, Iter {}/{},Total Loss {:.6f}, Loss_seg {:.6f},fusecross:{:.6f}, fusedice:{:.6f},spec_modal_loss:{} , circleloss[{}], use_modality:{}'.format((epoch + 1), args.num_epochs, (i_iter + 1), iter_per_epoch,  loss.item(), fuse_loss.item(),

                                                                                                            fuse_cross_loss.item(), fuse_dice_loss.item() , spec_modal_loss.item(), circlegenerate_loss_msg, mode_used_split))
                else:
                    logging.info('Epoch {}/{}, Iter {}/{}, Total Loss {:.6f}, Loss_seg {:.6f}, fusecross:{:.6f}, fusedice:{:.6f},spec_modal_loss:{:}, spec_logit_cls_loss:{:}, use_modality:{}'.format(
                            (epoch + 1), args.num_epochs, (i_iter + 1), iter_per_epoch, loss.item(), fuse_loss.item(),
                            fuse_cross_loss.item(), fuse_dice_loss.item(),spec_modal_loss.item() , spec_logit_cls_loss.item(), mode_used_split))

            step += 1


        lr_scheduler.step()
        step_lr = optimizer.param_groups[0]['lr']

        if is_circle == True:
            for iij in range(len(lr_scheduler_spec_list)):
                lr_scheduler_spec_list[iij].step()

        avg_loss = epoch_loss / epoch_iter

        if args.local_rank == args.print_rank:
            writer.add_scalar('epoch_avg_loss', avg_loss, global_step=(epoch + 1))
            writer.add_scalar('lr', step_lr, global_step=(epoch + 1))
            logging.info('train time per epoch: {}  cur_lr:{}'.format(time.time() - b , step_lr))

            ##########model save
            file_name = os.path.join(ckpts, 'model_last.pth')
            torch.save({
                'lr': step_lr,
                'step' : step,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'scheduler_dict': lr_scheduler.state_dict(),
            }, file_name)

            if avg_loss < best_loss:
                best_loss = avg_loss
                file_name = os.path.join(ckpts, 'model_best_loss.pth')
                torch.save({
                    'lr': step_lr,
                    'step': step,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'scheduler_dict': lr_scheduler.state_dict(),
                }, file_name)

            if (epoch + 1) % 20 == 0 or (epoch >= (args.num_epochs - 5)):
                file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch + 1))
                torch.save({
                    'lr': step_lr,
                    'step': step,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'scheduler_dict': lr_scheduler.state_dict(),
                }, file_name)

            if is_circle == True:
                # save generate model
                for ipos , (model_spec , optimizer_spec , lr_scheduler_spec) in enumerate(zip(model_spec_list , optimizer_spec_list , lr_scheduler_spec_list)):
                    file_name_ = os.path.join(ckpts, f'model_spec{ipos+1}_last.pth')
                    torch.save({
                        'lr': step_lr,
                        'step': step,
                        'epoch': epoch,
                        'state_dict': model_spec.state_dict(),
                        'optim_dict': optimizer_spec.state_dict(),
                        'scheduler_dict': lr_scheduler_spec.state_dict(),
                    }, file_name_)
                if (epoch + 1) % 20 == 0 or (epoch >= (args.num_epochs - 5)):
                    for ipos, (model_spec, optimizer_spec , lr_scheduler_spec) in enumerate(zip(model_spec_list, optimizer_spec_list , lr_scheduler_spec_list)):
                        file_name1 = os.path.join(ckpts, 'model_spec{}_{}.pth'.format(ipos + 1 , epoch + 1))
                        torch.save({
                            'lr': step_lr,
                            'step': step,
                            'epoch': epoch,
                            'state_dict': model_spec.state_dict(),
                            'optim_dict': optimizer_spec.state_dict(),
                            'scheduler_dict': lr_scheduler_spec.state_dict(),
                        }, file_name1)

    logging.info('total time: {:.4f} hours'.format((time.time() - start) / 3600))


def synthesize_modality(images=None , model_spec_list=None , loss_construct_fn=None , input_order_matrix=None , supervised_order_matrix=None ,first_index=None ,args=None):

    N,C = args.batch_size, args.num_modality
    input_order = input_order_matrix[first_index]
    supervised_order = supervised_order_matrix[first_index]

    torch.autograd.set_detect_anomaly(True)

    loss_reconst_list = []

    syntheis_image_dict = {}
    syntheis_pred = None
    for ipos, modal_id in enumerate(input_order):
        if ipos == 0:
            _image1 = images[modal_id]
            noise = torch.randn(_image1.shape).cuda()
            # input_data = torch.cat([_image1.detach(), _image1.detach() + noise], dim=0).cuda()
            input_data =  (_image1.detach() +  _image1.detach() + noise ) / 2.0
            syntheis_pred = model_spec_list[modal_id]( input_data.detach(), modal_id)
            syntheis_pred = torch.tanh(syntheis_pred)

            next_modal = images[supervised_order[ipos]]

            reconstruct_val = loss_construct_fn(syntheis_pred,  next_modal.detach().clone())
            loss_reconst_list.append(reconstruct_val)

            syntheis_image_dict[supervised_order[ipos]] = syntheis_pred

        else:

            _image1 = images[modal_id]
            _image2 = syntheis_pred
            assert _image1.shape == _image2.shape
            input_data = (_image1.clone() +  _image2.clone() ) / 2.0
            syntheis_pred = model_spec_list[modal_id](input_data.clone(), modal_id)
            syntheis_pred = torch.tanh(syntheis_pred)
            next_modal =  images[supervised_order[ipos]]

            reconstruct_val = loss_construct_fn(syntheis_pred,   next_modal.detach().clone())
            loss_reconst_list.append(reconstruct_val)

            syntheis_image_dict[supervised_order[ipos]] = syntheis_pred

    return loss_reconst_list , syntheis_image_dict


def loss_contrastive_modal_fn(anchor_list, feature_list, temperature=0.05):
    anchor_list = F.normalize(anchor_list, p=2, dim=-1)
    feature_list = F.normalize(feature_list, p=2, dim=-1)

    logits = torch.div(torch.matmul(anchor_list, feature_list.T), temperature)


    N = logits.shape[0]
    mask = torch.eye(N).cuda()
    exp_logits = torch.exp(logits)


    p_iv = (mask * logits).sum(dim=1)
    sum_p_iv = exp_logits.sum(dim=1)



    log_prob = p_iv - torch.log(sum_p_iv + 1e-6)

    mean_log_prob_pos = (log_prob).sum() / N

    loss = -mean_log_prob_pos
    loss = loss.mean()
    return loss

def random_mask_fn(images=None, mode_tot_list=None, num_modality=0, epoch=-1):

    used_mode_binary = [1 for _ in range(num_modality)]
    if num_modality == 4:
        if epoch <= 5:
            mode_used_split = random.sample(mode_tot_list, num_modality)
        elif epoch <= 10:
            mode_used_split = random.sample(mode_tot_list, num_modality - 1)
        elif epoch <= 15:
            mode_used_split = random.sample(mode_tot_list, num_modality - 2)
        elif epoch <= 20:
            mode_used_split = random.sample(mode_tot_list, num_modality - 3)
        else:
            mode_num = random.randint(1, num_modality)
            mode_used_split = random.sample(mode_tot_list, mode_num)

        # delete modality
        _images = images.clone()
        for m in [0, 1, 2, 3]:
            if m not in mode_used_split:
                _images[:, m, :, :, :] = 0
                used_mode_binary[m] = 0
            else:
                first_index = m

        return _images, mode_used_split, used_mode_binary, first_index

    elif num_modality == 3:
        if epoch <= 5:
            mode_used_split = random.sample(mode_tot_list, num_modality)
        elif epoch <= 10:
            mode_used_split = random.sample(mode_tot_list, num_modality - 1)
        elif epoch <= 15:
            mode_used_split = random.sample(mode_tot_list, num_modality - 2)
        else:
            mode_num = random.randint(1, num_modality)
            mode_used_split = random.sample(mode_tot_list, mode_num)

        # delete modality
        _images = images.clone()
        for m in [0, 1, 2]:
            if m not in mode_used_split:
                _images[:, m, :, :, :] = 0
                used_mode_binary[m] = 0
            else:
                first_index = m
        return _images, mode_used_split, used_mode_binary , first_index

    elif num_modality == 2:
        if epoch <= 5:
            mode_used_split = random.sample(mode_tot_list, num_modality)
        else:
            mode_num = random.randint(1, num_modality)
            mode_used_split = random.sample(mode_tot_list, mode_num)

        # delete modality
        _images = images.clone()
        for m in [0, 1]:
            if m not in mode_used_split:
                _images[:, m, :, :, :] = 0
                used_mode_binary[m] = 0
            else:
                first_index = m
        return _images, mode_used_split, used_mode_binary, first_index
    else:
        raise NotImplementedError("Error, num_modality must be 2 or 3 or 4")


def val_overlap(args=None, fuse_pred=None, label=None):

    if args.dataname == "BRATS2018":
        print("pred ET:{}  WT:{}  TC:{}".format(torch.sum(fuse_pred[0][0]), torch.sum(fuse_pred[0][1]), torch.sum(fuse_pred[0][2])))
        print("gt ET:{}  WT:{}  TC:{}".format(torch.sum(label[0][0]), torch.sum(label[0][1]), torch.sum(label[0][2])))


        label_dict = {}
        label_dict['ET'] = label[0][0]
        label_dict['WT'] = label[0][1]
        label_dict['TC'] = label[0][2]


        pred_dict = {}
        pred_dict['ET'] = fuse_pred[0][0]
        pred_dict['WT'] = fuse_pred[0][1]
        pred_dict['TC'] = fuse_pred[0][2]

        cal_metric_for_validation(args=args, label_dict=label_dict, pred_dict=pred_dict)
    elif args.dataname == "Prostate2012":
        if label.shape[1] == 2:
            print("pred PZ:{}  TZ:{}".format(torch.sum(fuse_pred[0][0]), torch.sum(fuse_pred[0][1]) ))
            print("gt PZ:{}  TZ:{}".format(torch.sum(label[0][0]), torch.sum(label[0][1]) ))


            label_dict = {}
            label_dict['PZ'] = label[0][0]
            label_dict['TZ'] = label[0][1]


            pred_dict = {}
            pred_dict['PZ'] = fuse_pred[0][0]
            pred_dict['TZ'] = fuse_pred[0][1]
            cal_metric_for_validation(args=args, label_dict=label_dict, pred_dict=pred_dict,num_cls=label.shape[1])
        else:
            print("pred PZ:{}  TZ:{}  BG:{}".format(torch.sum(fuse_pred[0][0]), torch.sum(fuse_pred[0][1]),
                                                    torch.sum(fuse_pred[0][2])))
            print("gt PZ:{}  TZ:{}  BG:{}".format(torch.sum(label[0][0]), torch.sum(label[0][1]),
                                                  torch.sum(fuse_pred[0][2])))

            label_dict = {}
            label_dict['PZ'] = label[0][0]
            label_dict['TZ'] = label[0][1]
            label_dict['BG'] = label[0][2]


            pred_dict = {}
            pred_dict['PZ'] = fuse_pred[0][0]
            pred_dict['TZ'] = fuse_pred[0][1]
            pred_dict['BG'] = fuse_pred[0][2]

            cal_metric_for_validation(args=args, label_dict=label_dict, pred_dict=pred_dict,num_cls=label.shape[1])


    elif args.dataname == 'Breast20':
        print("pred Tumor:{}  Breast:{}  BG:{}".format(torch.sum(fuse_pred[0][0]), torch.sum(fuse_pred[0][1]), torch.sum(fuse_pred[0][2])))
        print("gt Tumor:{}  Breast:{}  BG:{}".format(torch.sum(label[0][0]), torch.sum(label[0][1]), torch.sum(label[0][2])))


        label_dict = {}
        label_dict['BG'] = label[0][0]
        label_dict['Tumor'] = label[0][1]
        label_dict['Breast'] = label[0][2]



        pred_dict = {}
        pred_dict['BG'] = fuse_pred[0][0]
        pred_dict['Tumor'] = fuse_pred[0][1]
        pred_dict['Breast'] = fuse_pred[0][2]

        cal_metric_for_validation(args=args, label_dict=label_dict, pred_dict=pred_dict)

    elif args.dataname == 'Breast20_3m':
        print("pred Tumor:{}  Breast:{}  BG:{}".format(torch.sum(fuse_pred[0][0]), torch.sum(fuse_pred[0][1]), torch.sum(fuse_pred[0][2])))
        print("gt Tumor:{}  Breast:{}  BG:{}".format(torch.sum(label[0][0]), torch.sum(label[0][1]), torch.sum(label[0][2])))


        label_dict = {}
        label_dict['BG'] = label[0][0]
        label_dict['Tumor'] = label[0][1]
        label_dict['Breast'] = label[0][2]



        pred_dict = {}
        pred_dict['BG'] = fuse_pred[0][0]
        pred_dict['Tumor'] = fuse_pred[0][1]
        pred_dict['Breast'] = fuse_pred[0][2]

        cal_metric_for_validation(args=args, label_dict=label_dict, pred_dict=pred_dict)