#coding=utf-8

import os
import argparse
import time
import logging
import random
import numpy as np

import yaml
from argparse import Namespace
import shutil

from configPathSet import ConfigPath
from monai.utils.misc import set_determinism

from segmentor import  model_trainer
import torch.distributed as dist

def setup_args():
    dos_param = argparse.ArgumentParser()
    dos_param.add_argument("--run_times", default=1, type=int)
    dos_param.add_argument("--alpha", default=0.1, type=float)
    dos_param.add_argument("--beta", default=0.1, type=float)
    dos_param.add_argument('--lr', default=0.0002, type=float)
    dos_param.add_argument('--gpu', default="0", type=str)

    dos_param.add_argument('--local_rank', "--local-rank", default=0, type=int)


    dos_args, unknown = dos_param.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = dos_args.gpu
    dos_args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    return dos_args

dos_args = setup_args()




import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter




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

    elif dataname == "Breast20_3m":
        masks = [[True, False,False],
                 [False, True,False],
                 [False, False, True],
                 [True, True, False],
                 [True, False, True],
                 [False, True, True],
                 [True, True, True]]
        masks_torch = torch.from_numpy(np.array(masks))
        mask_name = ['preDCE',  # 1000
                     'DCE1',
                     'DCE2',
                     'preDCE_DCE1',
                     'preDCE_DCE2',
                     'DCE1_DCE2',
                     'preDCE_DCE1_DCE2']

    else:
        raise ValueError('not found mask')
    print(masks_torch.int())
    return masks , mask_name



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

def get_args_cfg_setlogger(proj_path=None, configpath=None, dataset_name=''):
    # reading config.yaml
    import logging
    config_path = os.path.join(proj_path, configpath)
    methodname = config_path.split('/')[-1].split('_config')[0]
    args, cfg = load_yaml_as_args_Nesting(config_path)

    print("configpath = {}  methodname={}  dataset_name={}".format(config_path, methodname, dataset_name))
    print("args = {}  cfg={}".format(args, cfg))

    ### --------------config----------------------------
    current_model = f"{dataset_name}_{methodname}"
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))

    current_train = current_time + "__" + f"run_times{dos_args.run_times}"+ "_" + f"alpha{dos_args.alpha}"+ "_" + f"beta{dos_args.beta}" +"__" + current_model


    args.snapshot_dir = f'{proj_path}/results/{dataset_name}/{methodname}/{current_train}'
    os.makedirs(args.snapshot_dir, exist_ok=True)
    print(f"snapshot_dir = {args.snapshot_dir}  datasetpath={args.datapath}")

    # ------------ set logging----------------  .%(msecs)03d:
    logging.basicConfig(filename=args.snapshot_dir + "/log.txt", level=logging.INFO, format='[%(asctime)s] %(message)s',
                        datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s] %(message)s'))
    logging.getLogger('').addHandler(console)
    shutil.copy(config_path, args.snapshot_dir)

    ## tensorboard writer
    writer = SummaryWriter(os.path.join(args.snapshot_dir, 'logger'))
    return args, cfg, writer, methodname

def main():
    ##########setting seed
    seed = 1024 + dos_args.local_rank + dos_args.run_times * 10
    set_determinism(seed=seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    dist.init_process_group(
        backend="nccl",
        init_method="env://"

    )


    num_gpus = torch.cuda.device_count()
    print(f'Available GPUs: {num_gpus}')

    # -------- setting config path and project path
    proj_path = os.path.dirname(__file__)
    proj_name = os.path.basename(proj_path)
    print("proj_path=",proj_path)
    print("proj_name=",proj_name)

    # BraTS2018_resize128 ， Breast20_3m_resize128， Prostate_resize128
    dataset_name = 'BraTS2018_resize128'
    configpath = ConfigPath(datasetname=dataset_name)
    configpathlist = [
        configpath.configpath_clrsnet,
    ]


    for i in range(len(configpathlist)):
        print("configpathlist[i][0]=",configpathlist[i][0])
        args, cfg, writer, methodname = get_args_cfg_setlogger(proj_path=proj_path, configpath=configpathlist[i][0], dataset_name=dataset_name)
        # -------------- dos-args to  args -------------
        args.local_rank = dos_args.local_rank
        args.num_gpus = dos_args.gpu

        args.coe_specloss = dos_args.alpha
        args.coe_consist = dos_args.beta
        args.lr = dos_args.lr

        torch.cuda.set_device(args.local_rank)


        print("----------------------###################---------------------\n"
              "-----------------current local rank = {}  worlsize={}----------------\n"
              "----------------------###################---------------------".format(args.local_rank, dist.get_world_size()))

        #---------------------------------seg args value------------------------------------------
        args.num_workers = 4
        # args.num_cls = 3
        # args.num_modality = 4
        # args.normalization = 'batchnorm'
        # args.spec_normalization = ''
        # args.coe_specloss = coe_val
        #### -------------------------------modify cfg----------------------------------

        cfg['snapshot_dir'] = args.snapshot_dir

        with open(os.path.join(args.snapshot_dir, f'{methodname}_config.yaml'), "w") as file:
            yaml.dump(cfg, file,
                      sort_keys=False,
                      default_flow_style=False,
                      )

        logging.info('second args = {}'.format(str(args)))
        logging.info('second cfg = {}'.format(cfg))

        train_loader , val_loader = load_data(args=args , methodname=methodname, proj_path=proj_path)
        masks , mask_name = get_missing_modality_combination(dataname=args.dataname)
        #------------------------Start to training-------------------------
        if methodname == 'CLRSNet':
            model_trainer.train(args=args, train_loader=train_loader, val_loader=val_loader, writer=writer,  masks=masks, mask_name=mask_name, proj_name=proj_name)
        else:
            raise NotImplementedError(f'not found method, error in {methodname}')

def load_data(args=None , methodname=None, proj_path=''):
    if args.dataname == 'BRATS2018':
        from data import BraTSDataSet_crop_norm_resize128
        #  20000 * 1  = args.num_steps * args.batch_size
        train_dataset = BraTSDataSet_crop_norm_resize128.BraTSDataSet(args.datapath, args.train_list, max_iters=None,
                                                         crop_size=args.input_size, scale=args.random_scale,
                                                         mirror=args.random_mirror, proj_path=proj_path)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   drop_last=False,
                                                   # shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=BraTSDataSet_crop_norm_resize128.my_collate,
                                                   sampler=train_sampler)
        val_loader = None

    elif args.dataname == 'Prostate2012':
        from data import ProstateDataSet_npy_Compose
        #  20000 * 1  = args.num_steps * args.batch_size
        train_dataset = ProstateDataSet_npy_Compose.ProstateDataSet(args.datapath, args.train_list, max_iters=None,
                                                         crop_size=args.input_size, scale=args.random_scale,
                                                         mirror=args.random_mirror, output_class_num=args.num_cls, proj_path=proj_path)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=0,
                                                   drop_last=False,
                                                   # shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=ProstateDataSet_npy_Compose.my_collate,
                                                   sampler=train_sampler)
        val_loader = None


    elif args.dataname == "Breast20_3m":

        from data import BreastDataSet_3m_Compose
        #  20000 * 1  = args.num_steps * args.batch_size
        train_dataset = BreastDataSet_3m_Compose.BreastDataSet(args.datapath, args.train_list, max_iters=None,
                                                         crop_size=args.input_size, scale=args.random_scale,
                                                         mirror=args.random_mirror, output_class_num=args.num_cls,proj_path=proj_path)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=0,
                                                   drop_last=False,
                                                   # shuffle=True,
                                                   pin_memory=True,
                                                   collate_fn=BreastDataSet_3m_Compose.my_collate,
                                                   sampler=train_sampler)
        val_loader = None

    return train_loader , val_loader


if __name__ == '__main__':
    main()


