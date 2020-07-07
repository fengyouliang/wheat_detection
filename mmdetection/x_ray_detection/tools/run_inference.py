import os
import time

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

avalible_gpu_ids = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, avalible_gpu_ids)))


from mmdet.apis import init_detector, inference_detector

res_vis_path = '/home/fengyouliang/code/mmdetection/wheat_detection/vis_test_result'
if not os.path.exists(res_vis_path):
    os.makedirs(res_vis_path)


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


def submission_test(config_file, checkpoint_file):

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    results = []
    test_path = '/kaggle/input/global-wheat-detection/test'
    test_path = '/home/fengyouliang/datasets/WHD/test'

    bar = tqdm(os.listdir(test_path))
    for file in bar:
        img = f'{test_path}/{file}'
        result = inference_detector(model, img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        result = [_ for _ in bbox_result if len(_) != 0]
        boxes = []
        scores = []

        for class_result in result:
            for item in class_result:
                box = item[:4]
                x1, y1, x2, y2 = box
                score = item[4]
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(score))

        cur_image_result = {
            'image_id': file[:-4],
            'PredictionString': format_prediction_string(boxes, scores)
        }
        results.append(cur_image_result)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    return test_df


def final_test(config, checkpoint, test_path, save_vis=False, save_name='nms'):

    model = init_detector(config, checkpoint, device='cuda:0')
    out = f'{res_vis_path}/{save_name}'

    results = []
    bar = tqdm(os.listdir(test_path))
    for file in bar:
        img = f'{test_path}/{file}'
        result = inference_detector(model, img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        results.append([img, bbox_result])
        if save_vis:
            model.show_result(img, result, text_color='red', thickness=2, out_file=f'{out}/{file}')

    return results


def main():
    config_file = '/home/fengyouliang/code/mmdetection/multi_work_dirs/maskrcnn/maskrcnn.py'
    checkpoint_file = '/home/fengyouliang/code/mmdetection/multi_work_dirs/maskrcnn/epoch_12.pth'
    test_path = '/home/fengyouliang/datasets/WHD/test'

    # results = final_test(config_file, checkpoint_file, test_path, save_vis=True, save_name='maskrcnn')
    # print(results)
    submission_test(config_file, checkpoint_file)


if __name__ == '__main__':
    main()
