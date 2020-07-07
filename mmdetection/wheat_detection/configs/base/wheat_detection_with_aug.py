_base_ = '../base/wheat_detection.py'

# use some data augmentation
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='Cutout',
    ),
    # dict(
    #     type='CoarseDropout',
    # ),
    # dict(
    #     type='GaussianBlur',
    # ),
    # dict(
    #     type='GaussNoise',
    # ),
    # dict(
    #     type='RandomGamma',
    # ),
    # dict(
    #     type='Rotate', limit=90,
    # ),
    # dict(
    #     type='OpticalDistortion',
    # ),
    # dict(
    #     type='GridDistortion',
    # ),
    # dict(
    #     type='ElasticTransform',
    # ),
    # dict(
    #     type='HueSaturationValue',
    # ),
    # dict(
    #     type='RGBShift',
    # ),
    # dict(
    #     type='RandomBrightness',
    # ),
    # dict(
    #     type='RandomContrast',
    # ),
    # dict(
    #     type='CLAHE',
    # ),
    # dict(
    #     type='InvertImg',
    # ),
    # dict(
    #     type='ChannelShuffle',
    # ),

]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomCrop', crop_size=[512, 512]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]
data = dict(train=dict(pipeline=train_pipeline))

