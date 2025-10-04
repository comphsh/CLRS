#coding=utf-8
import numpy as np
import torch
import torch.optim
import math
import logging
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


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