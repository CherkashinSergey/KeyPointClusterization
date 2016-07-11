import cv2
import os

ROOT_Dir = 'D:\\SCherkashin\\TrainingFolder\\Test\\'
new_Dir = 'new'
os.chdir(ROOT_Dir)
img = cv2.imread('1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(new_Dir + '\\' + '2.jpg', img)