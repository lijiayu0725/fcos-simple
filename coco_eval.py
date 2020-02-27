import argparse
import json

import torch
from apex import amp
from pycocotools.cocoeval import COCOeval

from data import DataIterator
from model import FCOS

batch_size = 16
stride = 32
resize = 800
max_size = 1333
resnet_dir = '/home/lijiayu/.cache/torch/checkpoints/resnet50-19c8e357.pth'
coco_dir = '/data/datasets/coco2017'
mb_to_gb_factor = 1024 ** 3
dist = True
training = False
detection_file = 'detections.json'
world_size = 8


def infer(model, args):
    rank = args.local_rank
    epoch_name = args.epoch
    model_state_dict_dir = 'checkpoints/final.pth' if epoch_name == 'final' else 'checkpoints/epoch-{}.pth'.format(
        epoch_name)

    load = torch.load(model_state_dict_dir, map_location='cpu')
    load = {k.replace('module.', ''): v for k, v in load.items()}
    model_state_dict = load
    model.load_state_dict(model_state_dict)

    model = model.cuda()
    model = amp.initialize(model,
                           opt_level='O2',
                           keep_batchnorm_fp32=True,
                           verbosity=0)
    # model = DistributedDataParallel(model)
    model.eval()
    if rank == 0:
        print('preparing dataset...')
    data_iterator = DataIterator(coco_dir, resize=resize, max_size=max_size, batch_size=batch_size, stride=stride,
                                 training=training, dist=dist, world_size=world_size)
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
    if rank == 0:
        print('gathering results...')
    results = [torch.cat(r, dim=0) for r in zip(*results)]

    for r, result in enumerate(results):
        all_result = [torch.ones_like(result, device=result.device) for _ in range(world_size)]
        torch.distributed.all_gather(list(all_result), result)
        results[r] = torch.cat(all_result, dim=0)

    if rank == 0:
        results = [r.cpu() for r in results]
        detections = []
        processed_ids = set()
        for scores, boxes, classes, image_id, ratios in zip(*results):
            image_id = image_id.item()
            if image_id in processed_ids:
                continue
            processed_ids.add(image_id)

            keep = (scores > 0).nonzero()
            scores = scores[keep].view(-1)
            boxes = boxes[keep, :].view(-1, 4) / ratios
            classes = classes[keep].view(-1).int()

            for score, box, cat in zip(scores, boxes, classes):
                x1, y1, x2, y2 = box.data.tolist()
                cat = cat.item()

                cat = data_iterator.coco.getCatIds()[cat]
                detections.append({
                    'image_id': image_id,
                    'score': score.item(),
                    'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                    'category_id': cat
                })
        if detections:

            print('writing {}...'.format(detection_file))
            detections = {'annotations': detections}
            detections['images'] = data_iterator.coco.dataset['images']
            detections['categories'] = [data_iterator.coco.dataset['categories']]
            json.dump(detections, open(detection_file, 'w'), indent=4)

            print('evaluating model...')
            coco_pred = data_iterator.coco.loadRes(detections['annotations'])
            coco_eval = COCOeval(data_iterator.coco, coco_pred, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        else:
            print('no detections!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epoch", default='final', type=str)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    model = FCOS(state_dict_path=resnet_dir, stride=stride)
    if args.local_rank == 0:
        print('FPN initialized!')
    infer(model, args)
