import numpy as np
import cv2 as cv
from scipy.stats import linregress
import matplotlib.pyplot as plt

def line_angle(img, color=[0,0,255]):
    line = np.where(np.all(img == color, axis=-1))
    # img[line[0], line[1]] = [0,0,255]
    # cv.imshow('lines', img)
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
    # print(slope)
    return angle

if __name__=='__main__':
    old_angle = -np.radians(180)
    angles = []
    diffs = []
    for i in range(181):
        # i = 90
        # print('Aim angle:')
        # print(i)
        image = cv.imread('../100GOPRO/testfahrt_1006/kandidaten/%s_horizon/%s.png'%('3_2', i))
        angle = line_angle(image)
        angles.append(angle)
        diffs.append(np.abs(np.degrees(old_angle - angle)))
        old_angle = angle
        # cv.imshow(str(i), image)
        # quit()

    print('angles in deg')
    # print(angle)
    angles = np.array(angles)
    print(np.degrees(angles-np.pi/2))

    print('diffs:')
    # print(angle)
    print(diffs)
    plt.plot(np.arange(len(diffs)-2), diffs[1:-1], label= 'diffs', linewidth=1, color='red')
    # plt.plot(np.arange(angles.size), angles, label= 'angles', linewidth=1, color='black')
    plt.show()
