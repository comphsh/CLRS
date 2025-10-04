#coding=utf-8
import argparse
import os
import time
import setproctitle

import logging
import torch
import torch.optim

import ipdb
from networks import VNet

from monai.losses import DiceCELoss
from utils.my_lr_scheduler import lr_schedule_election

from torch.cuda.amp import autocast as autocast
from torch.cuda import amp
from pred_each.pred_aux_tool import cal_metric_for_validation
scaler = amp.GradScaler(enabled=True)

def loss_kl_divergence_fn(mean , logvar):

    return -0.5 * torch.sum(1 + logvar - mean.pow(2)  - logvar.exp())


def train(args=None, train_loader=None, val_loader=None, writer=None, masks=None , mask_name=None, proj_name=''):
    ckpts = args.snapshot_dir

    model = VNet.VNet(n_channels=4, n_classes=3, normalization='batchnorm', has_dropout=False)
    model = torch.nn.DataParallel(model).cuda()
    optimizer , lr_scheduler = lr_schedule_election(model , args)

    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)


    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('############# training ############')
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = len(train_loader)
    step = 1
    best_loss = float('inf')  #

    for epoch in range(args.num_epochs):
        setproctitle.setproctitle('{}/{}: {}/{}'.format(proj_name, args.method_name, epoch + 1, args.num_epochs))
        model.train()

        epoch_loss = 0.0
        epoch_iter = 0
        b = time.time()
        for i , batch in enumerate(train_loader):

            images = torch.from_numpy(batch['image']).cuda()
            target = torch.from_numpy(batch['label']).cuda()

            layer_dict= model.module.forward_encoder(images)
            fuse_out_obj = model.module.forward_decoder(layer_dict)

            if isinstance(fuse_out_obj , list):
                fuse_logits = fuse_out_obj[0]
            elif isinstance(fuse_out_obj , dict ):
                fuse_logits = fuse_out_obj['logits']
            else:
                exit('not found model ouput type!')

            # monai standard DiceCE
            fuse_cross_loss = torch.tensor(0.0)
            fuse_dice_loss = torch.tensor(0.0)
            fuse_loss = loss_fn(fuse_logits , target)

            loss = fuse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_iter += 1
            ###log
            writer.add_scalar('tot_loss', loss.item(), global_step=step)
            writer.add_scalar('loss_seg', fuse_loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)



            logging.info('Epoch {}/{}, Iter {}/{}, Total Loss {:.6f}, Loss_seg {:.6f}, fusecross:{:.6f}, fusedice:{:.6f}'.format(
                    (epoch + 1), args.num_epochs, (i + 1), iter_per_epoch, loss.item(), fuse_loss.item(),
                    fuse_cross_loss.item(), fuse_dice_loss.item()))

            step += 1


        lr_scheduler.step()
        step_lr = optimizer.param_groups[0]['lr']

        avg_loss = epoch_loss / epoch_iter
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


    logging.info('total time: {:.4f} hours'.format((time.time() - start) / 3600))


