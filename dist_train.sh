rlaunch --cpu=16 --gpu=8 --memory=$((1024*128)) -- python -m torch.distributed.launch --nproc_per_node=8 dist_train.py
