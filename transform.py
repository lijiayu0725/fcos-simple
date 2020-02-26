import random

import cv2
import torch
import torch.nn.functional as F


def random_horizontal_flip(image, boxes):
    if random.randint(0, 1):
        image = cv2.flip(image, 1)
        boxes[:, 0] = image.shape[1] - boxes[:, 0] - boxes[:, 2]
    return image, boxes


def totensor(image, boxes=None):
    data = torch.from_numpy(image.transpose(2, 0, 1))
    data = data.float() / 255
    if boxes is not None:
        return data, boxes
    else:
        return data

def normalize(data, boxes=None, mean=(0.481, 0.455, 0.404), std=(1.0, 1.0, 1.0)):
    mean = torch.FloatTensor(mean)
    std = torch.FloatTensor(std)
    data = (data - mean[:, None, None]) / std[:, None, None]
    if boxes is not None:
        return data, boxes
    else:
        return data

def pad(data, boxes=None, stride=32):
    ph, pw = (stride - d % stride for d in data.shape[1:])
    data = F.pad(data, [0, pw, 0, ph])
    if boxes is not None:
        return data, boxes
    else:
        return data


if __name__ == '__main__':
    pass