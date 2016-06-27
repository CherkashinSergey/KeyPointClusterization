import cv2
img = cv2.imread('D:\\SCherkashin\\test\\DSC_02.jpg')
crop_img = img[100:100, 100:100] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

cv2.imwrite('D:\\SCherkashin\\test\\crop.jpg',crop_img)

cv2.imshow('cropped', crop_img)
#cv2.waitKey(0)
