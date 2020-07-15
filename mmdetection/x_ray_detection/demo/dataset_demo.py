from mmdet.datasets import build_dataset

batch_size = 4
fold_index = 0
# dataset_type = 'XRayDataset'
dataset_type = 'xray_demo'
classes = ('knife', 'scissors', 'lighter', 'zippooil', 'pressure', 'slingshot', 'handcuffs', 'nailpolish', 'powerbank',
           'firecrackers')
data_root = '/home/fengyouliang/datasets/x-ray/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        # classes=classes,
        ann_file=data_root + f'annotations/fold{fold_index}/train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
)


datasets = [build_dataset(data['train'])]

print(datasets)
