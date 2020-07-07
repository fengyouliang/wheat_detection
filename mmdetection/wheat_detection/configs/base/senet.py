# backbone = dict(
#     type='SENet',
#     block='SEResNetBottleneck',
#     layers=[3, 4, 6, 3],
#     groups=1,
#     reduction=16,
#     dropout_p=None,
#     inplanes=64,
#     input_3x3=False,
#     downsample_kernel_size=1,
#     downsample_padding=0,
#     num_classes=1
# ),

backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch'
),
