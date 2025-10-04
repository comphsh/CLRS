#coding=utf-8
import numpy as np
import torch
import torch.optim
import math
import logging
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# class PolyLR(_LRScheduler):
#     def __init__(self, optimizer, max_epochs, power=0.9, min_lr=1e-6, last_epoch=-1):
#         self.power = power
#         self.max_epochs = max_epochs  # 只使用 max_epochs
#         self.min_lr = min_lr
#         super().__init__(optimizer, last_epoch)  #自动存储了 初始学习率 self.base_lrs
#
#     def get_lr(self):
#         # 正确处理多个参数组
#         return [
#             max(base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power, self.min_lr)
#             for base_lr in self.base_lrs  # 使用 self.base_lrs
#         ]
#
#     def step(self, epoch=None):  # 添加 step 方法
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch  # 更新 last_epoch
#         super().step(epoch)  # 调用父类的 step 方法 (很重要)

def lr_schedule_election(model , args):
    # optimizer selection
    pg = [p for p in model.parameters() if p.requires_grad]
    if args.optim == 'sgd':
        # optimizer =  optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.99, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=args.weight_decay)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(pg, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0)
    else:
        exit('not found optimizer !!!')

    # lr scheduler selection
    if args.lr_scheduler == 'PolyLR':
        lambda_poly = lambda epoch: (1 - epoch / args.num_epochs) ** 0.9
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)
        logging.info("lr_scheduler 1={}".format(scheduler))

    elif args.lr_scheduler == 'LambdaLR':
        lambda_cosine = lambda epoch_x: ((1 + math.cos(epoch_x * math.pi / args.num_epochs)) / 2) * (  1 - args.learning_rate_fate) + args.learning_rate_fate
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)
        logging.info("lr_scheduler lambda cos 2={}".format(scheduler))

    elif args.lr_scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        logging.info("lr_scheduler 3={}".format(scheduler))
    elif args.lr_scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        logging.info("lr_scheduler 4={}".format(scheduler))

    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2)
        logging.info("lr_scheduler 5={}".format(scheduler))
    elif args.lr_scheduler == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs,  max_epochs=args.num_epochs)
        logging.info("lr_scheduler 6={}".format(scheduler))
    else:
        logging.info('not found scheduler !!!')

    return optimizer , scheduler

if __name__ == "__main__":
    pass