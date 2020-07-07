from pycocotools.coco import COCO
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

coco_json = '/home/open_datasets/coco/annotations/instances_val2017.json'
x_ray = '/home/fengyouliang/datasets/x-ray/coco/annotations/val.json'
coco = COCO(x_ray)
# catIds = coco.getCatIds()
imgIds = coco.getImgIds()
index = imgIds[np.random.randint(0, len(imgIds))]
img = coco.loadImgs(ids=index)[0]
id = img['id']
image_path = f"/home/fengyouliang/datasets/x-ray/coco/images/{img['filename']}"
I = cv.imread(image_path)
I = cv.cvtColor(I, cv.COLOR_RGB2BGR)
plt.axis('off')
plt.imshow(I)
# plt.show()

annIds = coco.getAnnIds(imgIds=id, iscrowd=None)
anns = coco.loadAnns(annIds)
print(f'anns: {len(anns)}')
coco.showAnns(anns, draw_bbox=True)
plt.show()

print()
