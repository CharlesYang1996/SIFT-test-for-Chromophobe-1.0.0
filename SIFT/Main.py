import numpy as np
import cv2
#https://blog.csdn.net/zhangziju/article/details/79754652
from matplotlib import pyplot as plt

imgname1 = 'pic/SIFT1.jpg'
imgname2 = 'pic/SIFT2.jpg'




sift = cv2.SIFT_create()

img1 = cv2.imread(imgname1)
img1 = cv2.resize(img1, (800,round(800*img1.shape[0]/img1.shape[1]),), interpolation=cv2.INTER_CUBIC)
print("Img size: [Width :",img1.shape[0],"]","[Height :",img1.shape[1],"]")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
kp1, des1 = sift.detectAndCompute(img1,None)   #des是描述子

img2 = cv2.imread(imgname2)
img2 = cv2.resize(img2, (800, round(800*img2.shape[0]/img2.shape[1])), interpolation=cv2.INTER_CUBIC)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
kp2, des2 = sift.detectAndCompute(img2,None)  #des是描述子

hmerge = np.hstack((gray1, gray2)) #水平拼接
cv2.imshow("gray", hmerge) #拼接显示为gray
cv2.waitKey(0)

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) #画出特征点，并显示为红色圆圈
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) #画出特征点，并显示为红色圆圈
hmerge = np.hstack((img3, img4)) #水平拼接
cv2.imshow("point", hmerge) #拼接显示为gray
cv2.waitKey(0)
# BFMatcher解决匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# 调整ratio
good = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append([m])

img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imshow("BFmatch", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()