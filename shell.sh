#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=1  --master_port 20081 train.py  --gpu=0 --run_times 1 --alpha 0.1  --beta 0.1  --lr 0.0002
python3 -m torch.distributed.launch --nproc_per_node=1  --master_port 20082 train.py  --gpu=0 --run_times 2 --alpha 0.1  --beta 0.1  --lr 0.002
python3 -m torch.distributed.launch --nproc_per_node=1  --master_port 20083 train.py  --gpu=1 --run_times 3 --alpha 0.1  --beta 0.1  --lr 0.00002

#brats
setsid nohup python3 -m torch.distributed.launch --nproc_per_node=2  --master_port 20092 train_brain.py  --gpu=2,3 --run_times 1 --alpha 0.1  --beta 1.0 > train_run1_$(date +"%Y%m%d_%H%M%S").log 2>&1 &
setsid nohup python3 -m torch.distributed.launch --nproc_per_node=2  --master_port 20093 train_brain.py  --gpu=2,3 --run_times 2 --alpha 0.1  --beta 1.0 > train_run2_$(date +"%Y%m%d_%H%M%S").log 2>&1 &
setsid nohup python3 -m torch.distributed.launch --nproc_per_node=2  --master_port 20094 train_brain.py  --gpu=2,3 --run_times 3 --alpha 0.1  --beta 1.0 > train_run3_$(date +"%Y%m%d_%H%M%S").log 2>&1 &




