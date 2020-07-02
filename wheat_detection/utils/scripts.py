import json
import os

import cv2
import numpy as np
from tqdm import tqdm

data_root_path = '/home/youliang/wheat_detection'


def cvt_csv(whd_df):
    # change dtype
    whd_df[['bbox_xmin', 'bbox_ymin', 'bbox_width', 'bbox_height']] = whd_df['bbox'].str.split(',', expand=True)
    whd_df['bbox_xmin'] = whd_df['bbox_xmin'].str.replace('[', '').astype(float)
    whd_df['bbox_ymin'] = whd_df['bbox_ymin'].str.replace(' ', '').astype(float)
    whd_df['bbox_width'] = whd_df['bbox_width'].str.replace(' ', '').astype(float)
    whd_df['bbox_height'] = whd_df['bbox_height'].str.replace(']', '').astype(float)

    # add xmax, ymax, and area columns for bounding box
    whd_df['bbox_xmax'] = whd_df['bbox_xmin'] + whd_df['bbox_width']
    whd_df['bbox_ymax'] = whd_df['bbox_ymin'] + whd_df['bbox_height']
    whd_df['bbox_area'] = whd_df['bbox_height'] * whd_df['bbox_width']
    return whd_df


def cala_mean_std(dir=f'{data_root_path}/train'):
    img_filenames = os.listdir(dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(dir + '/' + img_filename)
        # img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])  # [80.39894693, 80.89959808, 54.71170866] [0.31528999 0.31725333 0.21455572]
    print(s[0][::-1])  # [52.90205616, 53.51707949, 44.88701229] [0.20745904 0.2098709  0.1760275 ]


# https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.63.125b13e2hDzVMn&postId=86906
def add_seg(json_anno):
    new_json_anno = []
    for c_ann in json_anno:
        c_category_id = c_ann['category_id']
        if not c_category_id:
            continue
        bbox = c_ann['bbox']
        c_ann['segmentation'] = []
        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])

        c_ann['segmentation'].append(seg)
        new_json_anno.append(c_ann)
    return new_json_anno


def tset_add_seg():
    json_file = '../chongqing1_round1_train1_20191223/annotations.json'
    with open(json_file) as f:
        a = json.load(f)
    a['annotations'] = add_seg(a['annotations'])

    with open("new_ann_file.json", "w") as f:
        json.dump(a, f)


def main():
    pass


if __name__ == '__main__':
    main()
