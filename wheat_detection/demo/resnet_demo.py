from mmdet.models import ResNet
import torch


def resnet():
    self = ResNet(depth=18, dilations=(2, 2, 2, 2))
    print(self)
    self.eval()
    inputs = torch.rand(1, 3, 32, 32)
    level_outputs = self.forward(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))


def main():
    resnet()


if __name__ == '__main__':
    main()
