import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data
import cv2 as cv
import numpy as np

#read image
img = cv.imread('./00-08-52-932_points_cut.png')
#
# #convert from BGR to RGB
height, width = img.shape[:2]

img = cv.resize(img, (np.int32(width/4), np.int32(height/4)))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
fig = plt.figure()

ax1 = fig.add_subplot(111, projection='3d')
# cset = ax1.contourf(np.arange(img.shape[1]), np.arange(img.shape[0]), img, 100, zdir='z', offsset=0.5, cmap=cm.BrBG)

# fn = get_sample_data('00-08-52-932_points_cut.png', asfileobj=False)
# print(fn)
# # arr = read_png(fn)
# arr = fn

# arr = plt.imread('00-08-52-932_points_cut.png')

# arr = cv.resize(arr, (257, 720))
# 10 is equal length of x and y axises of your surface
# stepX, stepY = 10. / arr.shape[0], 10. / arr.shape[1]

# X1 = np.arange(-5, 5, stepX)
# Y1 = np.arange(-5, 5, stepY)
# X1 = np.arange(height/4)
# Y1 = np.arange(width/4)
X1 = np.arange(-width/8, width/8)
Y1 = np.arange(-height/8, height/8)
X1, Y1 = np.meshgrid(X1, Y1)
Z1 = np.zeros_like(X1)
# stride args allows to determine image quality
# stride = 1 work slow
# print(arr.shape)
# print(X1.shape)
# ax1.plot_surface(X1, Y1, np.zeros_like(X1), rstride=1, cstride=1, facecolors=arr.reshape((257,720,3)))
ax1.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, facecolors=img/255)
# cv.imwrite('lalala.png', arr.reshape((257,720,3)))
cv.imwrite('lalala.png', img)

plt.show()
