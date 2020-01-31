"""
Example script to run HR-net on a subset of a video. You'll need person detections from another source (Mask-RCNN is a good choice).
First clone the following git project:
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

It needs a pytorch1 environment. To run go to object-passing base dir and call the following:

HRNET_PATH=<path-to-hrnet lib dir> python scripts/hrnet_predict.py --cfg ~/pkgs/deep-hrnet/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os

HRNET_PATH = os.environ['HRNET_PATH']
sys.path.append(os.path.join(HRNET_PATH, 'lib'))


from scripts import hrnet_dataset

# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import argparse
import time
import os

import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np
import models
from config import cfg
from config import update_config
from core.function import AverageMeter
from utils.utils import create_logger
from core.inference import get_final_preds
from utils.transforms import flip_back


from util.misc import load, ensuredir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--imgs',
                        help='image folder',
                        required=True,
                        type=str)

    parser.add_argument('--bbox',
                        help='image folder',
                        required=True,
                        type=str)

    parser.add_argument('--out',
                        help='image folder',
                        required=True,
                        type=str)
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly - can't remove these as they are expected by update_config
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def predict(config, val_loader, val_dataset, model):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_names = []
    orig_boxes = []

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            num_images = input.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score

            names = meta['image']
            image_names.extend(names)
            orig_boxes.extend(meta['origbox'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time)
                print(msg)

        return all_preds, all_boxes, image_names, orig_boxes


def predict_imgs(model, img_folder, bbox_folder, output_file, normalize, detection_thresh):
    detections = {}
    for file in os.listdir(bbox_folder):
        dets = load(os.path.join(bbox_folder, file))
        assert dets.shape[1] == 5
        img_name = file[:-4]  # remove extension
        detections[img_name] = dets

    valid_dataset = hrnet_dataset.ImgFolderDataset(cfg, img_folder,
                                                   detections,
                                                   normalize, detection_thresh)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        pin_memory=True
    )

    start = time.time()
    preds, boxes, image_names, orig_boxes = predict(cfg, valid_loader, valid_dataset, model)
    end = time.time()
    print("Time in prediction: " + str(end - start))

    ensuredir(os.path.dirname(output_file))
    valid_dataset.rescore_and_save_result(output_file, preds, boxes, image_names, orig_boxes)


def main():
    args = parse_args()
    update_config(cfg, args)
    cfg.defrost()
    cfg.TEST.MODEL_FILE = HRNET_PATH + '/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    cfg.TEST.USE_GT_BBOX = False
    cfg.GPUS = (0,)
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Compose([transforms.ToTensor(), normalize])

    predict_imgs(model, args.imgs, args.bbox, args.out, normalize, 0.85)

if __name__ == '__main__':
    main()
