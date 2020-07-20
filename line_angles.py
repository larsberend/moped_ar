import numpy as np
import cv2 as cv
from scipy.stats import linregress


def line_angle(img, color=[0,255,0]):
    line = np.where(np.all(img == color, axis=-1))
    # print(line)
    # # quit()
    p1 = (line[1][0], line[0][0])
    p2 = (line[1][-1], line[0][-1])
    # # print(p1)
    # # print(p2)
    # print((p1[1] - p2[1], p1[0] - p2[0]))
    # angle = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    # return angle

    # slope, intersect = linregress(line[0], line[1])[:2]
    slope = (p1[1]-p2[1])/(p1[0]-p2[0])
    angle = np.arctan(slope)
    print(slope)
    return angle

if __name__=='__main__':
    old_angle = 180
    for i in range(91):
        print(i)
        image = cv.imread('../100GOPRO/testfahrt_1006/kandidaten/%s_processed/%s.png'%('3_2', i))
        angle = line_angle(image)
        print(angle)
        print(np.degrees(-angle-np.pi/2))
        # print(angle)
        # print(np.degrees(old_angle - angle))
        # old_angle = angle
