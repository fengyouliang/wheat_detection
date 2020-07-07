_base_ = [
    '../base/base_faster_rcnn.py',
    '../base/wheat_detection.py',
    '../base/other.py',
]

fp16 = dict(loss_scale=512.)
data = dict(
    samples_per_gpu=8
)
