import random
from contextlib import redirect_stdout

import cv2
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from transform import random_horizontal_flip, totensor, normalize, pad


class CocoDataset(Dataset):
    '''
    CocoDataset:
        process image and annotations belongs to it
        transform image in process of horizontal flip, totensor, normalize, pad so that can be divided by stride
        transform annotations with resized image
        collate_fn:
            collate batched images and annotations into batch formatting
    '''
    def __init__(self,
                 img_path='/Users/nick/datasets/coco2017/images',
                 resize=(640, 1024),
                 max_size=1333,
                 stride=32,
                 annotation_path='/Users/nick/datasets/coco2017/annotations',
                 training=True):
        super(CocoDataset, self).__init__()
        img_path = img_path + ('/train2017' if training else '/val2017')
        annotation_path = annotation_path + ('/instances_train2017.json' if training else '/instances_val2017.json')
        self.path = img_path
        self.resize = resize
        self.max_size = max_size
        self.training = training
        self.stride = stride
        self.transform = [random_horizontal_flip, totensor, normalize, pad] if training else [totensor, normalize, pad]
        with redirect_stdout(None):
            self.coco = COCO(annotation_path)
        self.ids = list(self.coco.imgs.keys())
        self.categories_inv = {k: i for i, k in enumerate(self.coco.getCatIds())}

    def __len__(self):
        '''
        :return: num of images
        '''
        return len(self.ids)

    def _get_target(self, idx):
        '''
        :param idx: index of image
        :return: (if find boxes, boxes -> Tensor(num_bbox, 4), categories -> Tensor(num_bbox, 1)
                  else, boxes->Tensor[[1], [1], [1], [1]]), categories->Tensor[[-1]]
        '''
        ann_ids = self.coco.getAnnIds(imgIds=idx)
        annotations = self.coco.loadAnns(ann_ids)
        boxes, categories = [], []
        for ann in annotations:
            # ignore invalid boxes
            if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            target = torch.FloatTensor(boxes), torch.FloatTensor(categories).unsqueeze(1)
        else:
            target = (torch.ones([1, 4]), torch.ones([1, 1]) * -1)
        return target

    def __getitem__(self, idx):
        '''
        get item of image and target
        :param idx: index of image
        :return: processed image data -> Tensor(3, H, W), targets -> Tensor(N, 5) 4 coodinates + 1 categoryId
        '''
        # load image
        id = self.ids[idx]
        image_name = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread('{}/{}'.format(self.path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # random resize shorter edge of image
        resize = self.resize
        if type(self.resize) in [list, tuple]:
            resize = random.randint(min(self.resize), max(self.resize))

        ratio = resize / min(image.shape[:2])
        if ratio * max(image.shape[:2]) > self.max_size:
            ratio = self.max_size / max(image.shape[:2])
        new_size = int(image.shape[1] * ratio), int(image.shape[0] * ratio)
        image = cv2.resize(image, new_size, cv2.INTER_LINEAR)
        data = image
        if self.training:
            # get annotations
            boxes, categories = self._get_target(id)
            boxes *= ratio

            # transform for training image
            for f in self.transform:
                data, boxes = f(data, boxes)

            target = torch.cat([boxes, categories], dim=1)
            return data, target
        else:
            # transform for validating image
            for f in self.transform:
                    data = f(data)
            return data, id, ratio

    def collate_fn(self, batch):
        '''
        :param batch: list of return in __getitem__
        :return: formatted batch data, if training: data -> Tensor(B, 3, H, W) , target -> Tensor(B, N, 5)
        '''
        # pad annotations in batch
        if self.training:
            data, targets = zip(*batch)
            max_det = max([t.shape[0] for t in targets])
            targets = [torch.cat(
                [t,
                 torch.ones(
                     [max_det - t.shape[0], 5]
                            ) * -1
                 ]
            ) for t in targets]
            targets = torch.stack(targets, dim=0)
        else:
            data, idxs, ratios = zip(*batch)

        # pad data for batch
        sizes = [d.shape[-2:] for d in data]
        max_h, max_w = (max(dim) for dim in zip(*sizes))

        data_stack = []
        for d in data:
            pw, ph = max_w - d.shape[-1], max_h - d.shape[-2]
            data_stack.append(
                F.pad(d, [0, pw, 0, ph]) if max(ph, pw) > 0 else d
            )
        data = torch.stack(data_stack)
        if self.training:
            return data, targets
        else:
            ratios = torch.FloatTensor(ratios).view(-1, 1, 1)
            return data, torch.IntTensor(idxs), ratios


class DataIterator():

    def __init__(self,
                 path='/Users/nick/datasets/coco2017',
                 resize=(640, 1024),
                 max_size=1333,
                 batch_size=2,
                 world_size=8,
                 stride=32,
                 training=True,
                 shuffle=False,
                 dist=False):
        self.resize = resize
        self.max_size = max_size
        img_path = path + '/images'
        annotation_path = path + '/annotations'
        self.dataset = CocoDataset(img_path=img_path,
                                   resize=resize,
                                   max_size=max_size,
                                   stride=stride,
                                   annotation_path=annotation_path,
                                   training=training)
        self.ids = self.dataset.ids
        self.coco = self.dataset.coco
        self.sampler = DistributedSampler(self.dataset) if dist else None
        batch_size = batch_size // world_size if dist else batch_size
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batch_size,
                                     collate_fn=self.dataset.collate_fn,
                                     shuffle=shuffle,
                                     num_workers=2,
                                     pin_memory=True,
                                     sampler=self.sampler)
        self.training = training

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for output in self.dataloader:
            if self.training:
                data, target = output
            else:
                data, ids, ratio = output

            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)

            if self.training:
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)
                yield data, target
            else:
                if torch.cuda.is_available():
                    ids = ids.cuda(non_blocking=True)
                    ratio = ratio.cuda(non_blocking=True)
                yield data, ids, ratio




if __name__ == '__main__':
    dataset = CocoDataset()
    data1 = dataset.__getitem__(0)
    data2 = dataset.__getitem__(1)
    res = dataset.collate_fn([data1, data2])