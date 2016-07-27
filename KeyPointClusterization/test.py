import TestWrap
import cv2

image = cv2.imread('D:\\SCherkashin\\TrainingFolder\\A4\\DSC_0083.JPG')

im2 = TestWrap.methodWrap(image)

cv2.imwrite('1.jpg', im2)