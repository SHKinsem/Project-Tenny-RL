import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(THIS_DIR, "..", "assets")

import cv2
import numpy as np

crop_w, crop_h = 100, 100
imgc = cv2.imread(os.path.join(ASSETS, 'martian.png'))
img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
print(img.shape)
crop_img = img[0:crop_h, 0:crop_w]
extend_img = np.zeros((crop_h*3, crop_w*3), dtype=np.uint8)
extend_img[crop_h:crop_h*2, crop_w:crop_w*2] = crop_img
for i in range(5):
    extend_img = cv2.blur(extend_img, (3, 3))
extbnd = 2
crop_img = extend_img[crop_h-extbnd:crop_h*2+extbnd, crop_w-extbnd:crop_w*2+extbnd]
crop_img = cv2.resize(crop_img, (crop_w, crop_h))
cv2.imwrite(os.path.join(ASSETS, 'martian_gray_100x100_bndblur.png'), crop_img)
# cv2.imshow('image', crop_img)
# cv2.waitKey(0)
# img = cv2.blur(img, (9, 9))
# # cv2.imwrite('martian_gray.png', img)
# cv2.imshow('image', img)
# cv2.waitKey(0)