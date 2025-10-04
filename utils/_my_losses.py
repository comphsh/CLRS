
import os
import time
import setproctitle

import logging
import random
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss
from monai.losses import DiceCELoss , DiceLoss
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss


def cal_bce(logits, target, num_cls):


    assert logits.shape == target.shape
    logit = torch.sigmoid(logits)
    epslon = 1e-8
    loss = 0.0
    for i in range(num_cls):
        logit_i = logit[:, i, ...]
        target_i = target[:, i, ...]
        val_i = -( target_i * torch.log(logit_i + epslon) + (1 - target_i) * torch.log(1 - logit_i + epslon) )
        loss_i = torch.sum(val_i)
        loss += loss_i

    num_elements = torch.numel(logits)
    loss = loss / num_elements


    return loss

def cal_bce_trick_for_value_stable(logits, target, num_cls):
    # logit : [B,C,H,W,D]
    # target: [B,C,H,W,D]
    assert logits.shape == target.shape
    # logit = torch.sigmoid(logits)
    loss = 0.0
    for i in range(num_cls):
        logit_i = logits[:, i, ...]
        target_i = target[:, i, ...]
        # val_i = -(target_i * torch.log(logit_i + epslon) + (1 - target_i) * torch.log(1 - logit_i + epslon))

        # loss = -( ylogp + (1-y)log(1-p) ) ===> loss = max(logit,0) - logit* y + log(1 + exp{-|logit|})
        val_i = torch.max(logit_i, torch.zeros_like(logit_i)) - logit_i * target_i + torch.log(1 + torch.exp(-torch.abs(logit_i)))

        loss_i = torch.sum(val_i)
        loss += loss_i

    num_elements = torch.numel(logits)  # 除以 [B,C,H,W,D]
    loss = loss / num_elements

    return loss


def cal_bce_trick_for_value_stable_weight(logits, target, num_cls, weight):
    # logit : [B,C,H,W,D]
    # target: [B,C,H,W,D]
    assert logits.shape == target.shape
    # logit = torch.sigmoid(logits)
    loss = 0.0
    norm_weight = weight

    for i in range(num_cls):
        logit_i = logits[:, i, ...]
        target_i = target[:, i, ...]

        # val_i = -(target_i * torch.log(logit_i + epslon) + (1 - target_i) * torch.log(1 - logit_i + epslon))

        # loss = -( ylogp + (1-y)log(1-p) ) ===> loss = max(logit,0) - logit* y + log(1 + exp{-|logit|})
        val_i = torch.max(logit_i, torch.zeros_like(logit_i)) - logit_i * target_i + torch.log(1 + torch.exp(-torch.abs(logit_i)))

        loss_i = torch.sum(val_i)
        loss += loss_i * norm_weight[i]

    num_elements = torch.numel(logits)  # 除以 [B,C,H,W,D]
    loss = loss / num_elements

    return loss

def cal_ce_target_onehot_trick_for_value_stable(logits, target, num_cls):
    # logits = [B,C,H,W,D]
    # target = [B ,C, H W , D]
    assert logits.shape == target.shape
    max_vals = torch.max(logits, dim=1, keepdim=True).values  # [B,1,H,W,D]

    stable_logits = logits - max_vals  #stable_logits [B,C,H,W,D]


    log_probs = stable_logits - torch.log(torch.sum(torch.exp(stable_logits), dim=1, keepdim=True))
    loss = -torch.sum(target * log_probs, dim=1)
    # loss_sum = loss.sum()
    loss_mean = loss.mean()

    # loss = 0.0
    # for cls in range(num_cls):
    #     target_i = target[:, cls, ...]
    #     stable_logit_i = stable_logits[: , cls, ...]
    #     val = stable_logit_i - torch.log(torch.sum(  torch.exp(stable_logits)  , dim=1 ))
    #     loss += torch.sum(val * target_i)
    # loss_sum = - loss
    # loss_mean = - loss / torch.numel(target[:,0,...])


    # built_loss_sum = F.cross_entropy(logits, target, reduction='sum')
    # built_loss_mean = F.cross_entropy(logits, target , reduction='mean')
    return  loss_mean

def cal_ce_target_onehot_trick_for_value_stable_weight(logits, target, num_cls, weight):
    # logits = [B,C,H,W,D]
    # target = [B ,C, H W , D]
    assert logits.shape == target.shape
    norm_weight = weight

    max_vals = torch.max(logits, dim=1, keepdim=True).values  # [B,1,H,W,D]

    stable_logits = logits - max_vals  #stable_logits [B,C,H,W,D]


    # log_probs = stable_logits - torch.log(torch.sum(torch.exp(stable_logits), dim=1, keepdim=True))
    # loss = -torch.sum(target * log_probs, dim=1)
    # # loss_sum = loss.sum()
    # loss_mean = loss.mean()



    loss = 0.0
    for cls in range(num_cls):
        target_i = target[:, cls, ...]
        stable_logit_i = stable_logits[: , cls, ...]
        val = stable_logit_i - torch.log(torch.sum(  torch.exp(stable_logits)  , dim=1 ))
        loss += torch.sum(val * target_i) * norm_weight[cls]
    # loss_sum = - loss
    loss_mean = - loss / torch.numel(target[:,0,...])


    # built_loss_sum = F.cross_entropy(logits, target, reduction='sum')
    # built_loss_mean = F.cross_entropy(logits, target , reduction='mean')
    return  loss_mean


def cal_standard_ce(logits, target, num_cls):
    # logits = [B,C,H,W,D]
    # target = [B , H W , D]
    assert logits.shape[2:] == target.shape[1:] and logits.shape[0] == target.shape[0]
    epslon = 1e-8
    probs = torch.softmax(logits, dim=1)
    loss = 0.0
    for cls in range(num_cls):
        prob_i = probs[: , cls, ...]
        target_i = (target == cls)
        val = target_i * torch.log(prob_i + epslon)
        loss += torch.sum(val)

    loss_mean = - loss / torch.numel(target)

    # built_loss_sum = F.cross_entropy(logits, target, reduction='sum')
    # built_loss_mean = F.cross_entropy(logits, target , reduction='mean')
    return loss_mean

def cal_ce_target_onehot(logits, target, num_cls):
    # logits = [B,C,H,W,D]
    # target = [B ,C, H W , D]
    assert logits.shape == target.shape
    epslon = 1e-8
    probs = torch.softmax(logits, dim=1)
    loss = 0.0
    for cls in range(num_cls):
        prob_i = probs[: , cls, ...]
        target_i = target[:, cls, ...]
        val = target_i * torch.log(prob_i + epslon)
        loss += torch.sum(val)

    loss_mean = - loss / torch.numel(target[: , 0 , ...])

    # built_loss_sum = F.cross_entropy(logits, target, reduction='sum')
    # built_loss_mean = F.cross_entropy(logits, target , reduction='mean')
    return loss_mean


def single_SoftDice(prob , target):

    smooth = 1.0
    epslon = 1e-10
    prob = prob.float()
    target = target.float()
    intersection = torch.sum(prob  * target)
    ss = prob.sum() + target.sum()
    dice = (2 * intersection + epslon) / (ss + epslon)
    return dice

def cal_soft_diceloss_self(logits , target, num_cls=0 , sigmoid=False, softmax=False):
    assert logits.shape == target.shape
    if sigmoid == True:
        prob = torch.sigmoid(logits)
    elif softmax == True:
        prob = torch.softmax(logits, dim=1)
    else:
        prob = logits

    SoftDice = 0.0
    for cls in range(num_cls):
        SoftDice += single_SoftDice(prob[: , cls , ...] , target[:, cls , ...])

    SoftDice /= num_cls
    loss = 1 - SoftDice
    return loss

def cal_soft_diceloss_self_weight(logits , target, num_cls=0 , norm_weight=None, sigmoid=False, softmax=False):
    assert logits.shape == target.shape
    if sigmoid == True:
        prob = torch.sigmoid(logits)
    elif softmax == True:
        prob = torch.softmax(logits, dim=1)
    else:
        prob = logits

    loss = 0.0
    for cls in range(num_cls):
        SoftDice = single_SoftDice(prob[: , cls , ...] , target[:, cls , ...])
        loss_i = (1.0 - SoftDice) * norm_weight[cls]
        loss += loss_i

    loss /= num_cls
    return loss

class MyDice_BinaryCELoss(nn.Module):
    def __init__(self, to_onehot_y=False, sigmoid=False, softmax=False, mode='none'):
        super().__init__()  # Call nn.Module's constructor

        self.sigmoid = sigmoid
        self.softmax = softmax
        self.mode = mode  # include CE BCE none

    # logtis: [B C H W D]   target: [B C H W D] C个0 1 mask
    def forward(self, logits, target, num_cls=0):

        dice_loss =  cal_soft_diceloss_self(logits, target , num_cls=logits.shape[1] , sigmoid=self.sigmoid , softmax=self.softmax) # self.dice_loss(input_dice, target)
        if self.mode == 'BCE':
            # bce_loss = cal_bce(logits, target, logits.shape[1])
            bce_loss = cal_bce_trick_for_value_stable(logits, target, num_cls=logits.shape[1])
            return dice_loss + bce_loss

        elif self.mode == 'CE':
            # ce_loss = cal_ce_target_onehot(logits, target, logits.shape[1])
            ce_loss = cal_ce_target_onehot_trick_for_value_stable(logits, target, num_cls=logits.shape[1])
            return dice_loss + ce_loss
        else:
            raise ValueError('error')

class MyDice_Loss(nn.Module):
    def __init__(self, to_onehot_y=False, sigmoid=False, softmax=False):
        super().__init__()  # Call nn.Module's constructor

        self.sigmoid = sigmoid
        self.softmax = softmax


    def forward(self, logits, target, num_cls=0, weight=None):

        # dice_loss =  cal_soft_diceloss_self(logits, target , num_cls=logits.shape[1] , sigmoid=self.sigmoid , softmax=self.softmax) # self.dice_loss(input_dice, target)
        dice_loss = cal_soft_diceloss_self_weight(logits, target, num_cls=logits.shape[1], norm_weight=weight ,sigmoid=self.sigmoid, softmax=self.softmax)

        return dice_loss

class MyBCE_or_CELoss(nn.Module):
    def __init__(self, to_onehot_y=False,  mode='none'):
        super().__init__()  # Call nn.Module's constructor
        self.mode = mode  # include CE BCE none


    def forward(self, logits, target, num_cls=0, weight=None):

        if self.mode == 'BCE':
            # bce_loss = cal_bce(logits, target, logits.shape[1])
            # bce_loss = cal_bce_trick_for_value_stable(logits, target, num_cls=logits.shape[1])
            bce_loss = cal_bce_trick_for_value_stable_weight(logits, target, num_cls=logits.shape[1], weight=weight)

            return bce_loss

        elif self.mode == 'CE':
            # ce_loss = cal_ce_target_onehot(logits, target, logits.shape[1])
            # ce_loss = cal_ce_target_onehot_trick_for_value_stable(logits, target, num_cls=logits.shape[1])
            ce_loss = cal_ce_target_onehot_trick_for_value_stable_weight(logits, target, num_cls=logits.shape[1], weight=weight)
            return ce_loss
        else:
            raise ValueError('error')

def merge_test(logits, label_onehot):
    monai_loss_fn_dice_ce_sig = DiceCELoss(to_onehot_y=False, sigmoid=True)
    monai_loss_fn_dice_ce_soft = DiceCELoss(to_onehot_y=False, softmax=True)

    self_loss_fn_dice_ce_sig = MyDice_BinaryCELoss(sigmoid=True, mode='CE')
    self_loss_fn_dice_ce_soft = MyDice_BinaryCELoss(softmax=True, mode='CE')

    monai_dicebce_sig = monai_loss_fn_dice_ce_sig(logits , label_onehot)
    monai_dicebce_soft = monai_loss_fn_dice_ce_soft(logits, label_onehot)


    self_dicebce_sig = self_loss_fn_dice_ce_sig(logits, label_onehot)
    self_dicebce_soft = self_loss_fn_dice_ce_soft(logits, label_onehot)

    return {
        "self dice bce sig":self_dicebce_sig,
        "self dice bce soft": self_dicebce_soft,

        "monai dice bce sig": monai_dicebce_sig,
        "monai dice bce soft": monai_dicebce_soft,
    }

def test_():

    logits = torch.tensor(list(range(1*3*128*128*128))).reshape(1,3,128,128,128).float()
    label_cls = torch.randint(0,3 , (1,128,128,128))
    #  torch.Size([1, 2, 2, 3]) -->  torch.Size([1, 3, 2, 2])
    label_onehot = F.one_hot(label_cls , 3).permute(0, 4, 1, 2, 3)

    label_cls = label_cls.float()
    label_onehot = label_onehot.float()

    print("logits",logits.shape)
    print(label_cls.unique() , label_cls.shape)
    print(label_onehot.unique(), label_onehot.shape)

    print("-----------------")

    v6 = merge_test(logits, label_onehot)
    print("v6=", v6)


if __name__ == "__main__":
    pass

    test_()