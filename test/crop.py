import cv2

img = cv2.imread("/home/jade/桌面/lc/卫星立体像对/JAX_170_018_002_RIGHT_RGB.tif")
print(img.shape)
cropped = img[0:512, 0:512]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite("/home/jade/桌面/lc/卫星立体像对/cv_cut_right.jpg", cropped)