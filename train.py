import argparse
import os

import torch
from apex import amp
from apex.parallel import DistributedDataParallel
from math import isfinite
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from data import DataIterator
from model import FCOS

warmup = 500
warmup_ratio = 1 / 3
gamma = 0.1
milestores = [8, 11]
batch_size = 2
stride = 32
lr = 0.01
weight_decay = 1e-4
momentem = 0.9
epochs = 12
shuffle = False
resize = 800
max_size = 1333
resnet_dir = '/home/nick/.cache/torch/checkpoints/resnet50-19c8e357.pth'
coco_dir = '/home/nick/datasets/coco2017'
mb_to_gb_factor = 1024 ** 3
dist = False
world_size = 1


def train(model, rank=0):
    model.cuda()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentem)

    model.train()
    if rank == 0:
        print('preparing dataset...')
    data_iterator = DataIterator(path=coco_dir, batch_size=batch_size, stride=stride, shuffle=shuffle, resize=resize, max_size=max_size,
                                 dist=dist, world_size=world_size)
    if rank == 0:
        print('finish loading dataset!')

    def schedule_warmup(i):
        return warmup_ratio

    def schedule(epoch):
        return gamma ** len([m for m in milestores if m <= epoch])

    scheduler_warmup = LambdaLR(optimizer, schedule_warmup)
    scheduler = LambdaLR(optimizer, schedule)
    if rank == 0:
        print('starting training...')

    for epoch in range(1, epochs + 1):
        cls_losses, box_losses, centerness_losses = [], [], []
        if epoch != 1:
            scheduler.step(epoch)
        for i, (data, target) in enumerate(data_iterator, start=1):
            optimizer.zero_grad()
            cls_loss, box_loss, centerness_loss = model([data, target])
            loss = cls_loss + box_loss + centerness_loss
            loss.sum().backward()

            optimizer.step()
            if epoch == 1 and i <= warmup:
                scheduler_warmup.step(i)

            cls_loss, box_loss, centerness_loss = cls_loss.mean().clone(), box_loss.mean().clone(), centerness_loss.mean().clone()

            if rank == 0:
                cls_losses.append(cls_loss)
                box_losses.append(box_loss)
                centerness_losses.append(centerness_loss)

            if rank == 0 and not isfinite(cls_loss + box_loss + centerness_loss):
                raise RuntimeError('Loss is diverging!')

            del cls_loss, box_loss, centerness_loss, target, data

            if rank == 0 and i % 10 == 0:
                focal_loss = torch.FloatTensor(cls_losses).mean().item()
                box_loss = torch.FloatTensor(box_losses).mean().item()
                centerness_loss = torch.FloatTensor(centerness_losses).mean().item()
                learning_rate = optimizer.param_groups[0]['lr']

                msg = '[{:{len}}/{}]'.format(epoch, epochs, len=len(str(epochs)))
                msg += '[{:{len}}/{}]'.format(i, len(data_iterator), len=len(str(len(data_iterator))))
                msg += ' focal loss: {:.3f}'.format(focal_loss)
                msg += ', box loss: {:.3f}'.format(box_loss)
                msg += ', centerness loss: {:.3f}'.format(centerness_loss)
                msg += ', lr: {:.2g}'.format(learning_rate)
                msg += ', cuda_memory: {:.3g} GB'.format(torch.cuda.memory_cached() / mb_to_gb_factor)
                print(msg, flush=True)
                del cls_losses[:], box_losses[:], focal_loss, box_loss

        if rank == 0:
            print('saving model for epoch {}'.format(epoch))
            torch.save(model.state_dict(), './checkpoints/epoch-{}.pth'.format(epoch))

    if rank == 0:
        print('finish training, saving the final model...')
        torch.save(model.state_dict(), './checkpoints/final.pth')
        print('-' * 10 + 'completed!' + '-' * 10)


if __name__ == '__main__':
    model = FCOS(state_dict_path=resnet_dir)
    train(model)
