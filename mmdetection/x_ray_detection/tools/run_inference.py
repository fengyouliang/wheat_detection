import os
import time

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import json

available_gpu_ids = [2]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, available_gpu_ids)))

from mmdet.apis import init_detector, inference_detector


def submission_test(config_file, checkpoint_file, test_path, save_name):

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    results = []

    bar = tqdm(sorted(os.listdir(test_path)))
    for file in bar:
        bar.set_description(file)
        img = f'{test_path}/{file}'
        result = inference_detector(model, img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        result = bbox_result
        pre_image_res = [item.tolist() for item in result]
        results.append(pre_image_res)
    save_path = '/home/fengyouliang/code/mmdetection/x_ray_detection/submission_json'
    dump_ = f'{save_path}/{save_name}.json'
    json.dump(results, open(dump_, 'w'), ensure_ascii=False, indent=4)


def main():
    config_file = '../configs/cascade_rcnn/renext101_64_cascade_2x.py'
    checkpoint_file = '../work_dirs/renext101_64_cascade_2x/epoch_22.pth'
    test_path = '/home/fengyouliang/datasets/x-ray/test1'

    submission_test(config_file, checkpoint_file, test_path, save_name='renext101_64_cascade_2x_epoch_22')


if __name__ == '__main__':
    main()
