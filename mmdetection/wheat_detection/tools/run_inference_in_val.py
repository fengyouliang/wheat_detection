import os
import time

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

avalible_gpu_ids = [1]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, avalible_gpu_ids)))


from mmdet.apis import init_detector, inference_detector

save_path = '/home/youliang/code/mmdetection/wheat_detection/submission'
res_vis_path = '/home/youliang/code/mmdetection/wheat_detection/vis_result/test'
if not os.path.exists(res_vis_path):
    os.makedirs(res_vis_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


def to_csv(results, show_my_result=True, csv_flag=False):
    # my result vis
    df_line = {}
    for img, result in results:
        result = [_ for _ in result if len(_) != 0]
        boxes = []
        scores = []

        if show_my_result:
            image = cv2.imread(img)

        for class_result in result:
            for item in class_result:
                box = item[:4]
                x1, y1, x2, y2 = box
                score = item[4]
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(score))
                if show_my_result:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255))

        if show_my_result:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.show()

        image_row = {
            'image_id': img.split('/')[-1][:-4],
            'PredictionString': format_prediction_string(boxes, scores)
        }

        df_line.append(image_row)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    cur_time = time.strftime('%m%d_%H_%M')
    if csv_flag:
        test_df.to_csv(f'{save_path}/submission_{cur_time}.csv', index=False)
    print(test_df)


def final_test(config, checkpoint, test_path, score_thr=0.3):

    model = init_detector(config, checkpoint, device='cuda:0')

    color = 'green' if 'vis_result' in test_path else 'red'

    results = []
    bar = tqdm(os.listdir(test_path))
    for file in bar:
        img = f'{test_path}/{file}'
        result = inference_detector(model, img)
        # mmdet show result
        model.show_result(img, result, score_thr=score_thr, bbox_color=color, text_color=color, thickness=1, out_file=f'{res_vis_path}/{file}')
        results.append([img, result])

    return results


def show_val_results(config, checkpoint, test_path, score_thr=0.3):

    val_vis_path = '../wheat_detection/vis_result/val'
    os.makedirs(val_vis_path, exist_ok=True)

    model = init_detector(config, checkpoint, device='cuda:0')

    val_image_bbox = get_image_name()
    bar = tqdm(val_image_bbox.items())
    for file_name, bboxes in bar:
        img = f'{test_path}/{file_name}'

        # draw pred
        result = inference_detector(model, img)
        model.show_result(img, result, score_thr=score_thr, bbox_color='green', text_color='green', thickness=1, out_file=f'{val_vis_path}/{file_name}')

        # draw gt
        image = cv2.imread(f'{val_vis_path}/{file_name}')
        assert image is not None
        for box in bboxes:
            x, y, w, h = list(map(int, box))
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
        cv2.imwrite(f'{val_vis_path}/{file_name}', image)


def get_image_name(json_file='/home/youliang/wheat_detection/annotations/val.json'):
    # 从annotation json文件中取出image name
    import json
    j = json.load(open(json_file, 'r'))

    images = j['images']
    images_dict = {item['id']: item['file_name'] for item in images}
    annotations = j['annotations']

    results = dict()
    for ann in annotations:
        image_id = ann['image_id']
        bbox = ann['bbox']
        bboxes = results.get(images_dict[image_id], [])
        bboxes.append(bbox)
        results[images_dict[image_id]] = bboxes

    return results


def main():
    config_file = '/home/youliang/code/mmdetection/work_dirs/base_faster_multi_scale/base_faster_multi_scale.py'
    checkpoint_file = '/home/youliang/code/mmdetection/work_dirs/base_faster_multi_scale/latest.pth'
    train_path = '/home/youliang/wheat_detection/train/'
    test_path = '/home/youliang/wheat_detection/test/'

    # show_val_results(config_file, checkpoint_file, train_path, score_thr=0.3)
    ret = final_test(config_file, checkpoint_file, test_path, score_thr=0.3)


if __name__ == '__main__':
    main()
