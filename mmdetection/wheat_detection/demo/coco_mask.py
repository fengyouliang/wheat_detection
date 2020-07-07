import cv2 as cv
import numpy as np

image = np.zeros((500, 500))
points = [
    347.65,
    187.71,
    347.22,
    187.06,
    346.95,
    185.92,
    346.95,
    184.47,
    347,
    183.44,
    347.38,
    183.06,
    348.46,
    182.52,
    349.6,
    182.52,
    350.35,
    182.52,
    355.81,
    182.47,
    355.86,
    185.28,
    356.24,
    186.36,
    356.18,
    187.81,
    355.05,
    187.76,
    354.51,
    187,
    352.08,
    187.06,
    351.76,
    187.92,
    350.89,
    187.27,
    349.27,
    187.27,
    348.52,
    187.92,
    348.25,
    188.03
]
points = [int(item) for item in points]
p = np.array(points).reshape(-1, 2)
for idx, pp in enumerate(p[::10]):
    cv.putText(image, str(idx), (pp[0], pp[1]), cv.FONT_HERSHEY_SIMPLEX, 2, 255, thickness=1)

cv.imwrite('demo.jpg', image)
