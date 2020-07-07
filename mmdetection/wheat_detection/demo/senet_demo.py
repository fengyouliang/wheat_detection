from mmdet.models.builder import build_seblock, build_backbone, build_detector
from mmdet.models.builder import SE_BLOCK
from mmcv import Config

import torch

# seb = SE_BLOCK.get('SEResNetBottleneck')
# print(seb)

# config_py = '/home/fengyouliang/code/mmdetection/wheat_detection/configs/base/senet.py'
config_py = '/home/fengyouliang/code/mmdetection/wheat_detection/configs/faster_rcnn/base_faster_senet.py'
cfg = Config.fromfile(config_py)
# senet = build_backbone(cfg.backbone)
faster = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

x = torch.rand(1, 3, 224, 224)
backbone_out = faster.backbone(x)
neck_out = faster.neck(backbone_out)
print()
