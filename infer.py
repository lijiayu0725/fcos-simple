import torch
from torch.nn import DataParallel

from data import DataIterator
from model import RetinaNet

batch_size = 2
stride = 32
resize = 800
max_size = 1333
resnet_dir = '/home/nick/.cache/torch/checkpoints/resnet50-19c8e357.pth'
coco_dir = '/home/nick/datasets/coco2017'
mb_to_gb_factor = 1024 ** 3
dist = False
model_state_dict = 'checkpoints/final.pth'
training = False


def infer(model, rank=0):
    model = model.cuda()
    model = DataParallel(model)
    model.load_state_dict(torch.load(model_state_dict))
    model.eval()
    if rank == 0:
        print('preparing dataset...')
    data_iterator = DataIterator(coco_dir, resize=resize, max_size=max_size, batch_size=batch_size, stride=stride,
                                 training=training, dist=dist)
    if rank == 0:
        print('finish loading dataset!')

    results = []
    with torch.no_grad():
        for i, (data, ids, ratios) in enumerate(data_iterator, start=1):
            scores, boxes, classes = model(data)
            results.append([scores, boxes, classes, ids, ratios])
            if rank == 0:
                size = len(data_iterator.ids)
                msg = '[{:{len}}/{}]'.format(min(i * batch_size, size), size, len=len(str(size)))
                print(msg, flush=True)

    results = [torch.cat(r, dim=0) for r in zip(*results)]
    results = [r.cpu() for r in results]


if __name__ == '__main__':
    model = RetinaNet(state_dict_path=resnet_dir)
    infer(model)
