import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
from camera_K import K

def find_bv(img, color1, color2):
    points1 = np.where(np.all(img==color1), axis=-1)
    points2 = np.where(np.all(img==color1), axis=-1)

    diff_min = 100
    pitch_guess = 0

    for pitch in np.arange(30, 110, 1):
        H = get_homography(pitch)
        warped1 = warp(H, points1)
        warped2 = warp(H, points2)
        slope1, intersect1 = linregress(warped1)[:2]
        slope2, intersect2 = linregress(warped2)[:2]

        diff = np.abs(slope1-slope2)

        if diff<diff_min:
            diff_min = diff
            pitch_guess = pitch

    for pitch in np.arange(pitch_guess-1, pitch_guess+1, 0.01):
        H = get_homography(pitch)
        warped1 = warp(H, points1)
        warped2 = warp(H, points2)
        slope1, intersect1 = linregress(warped1)[:2]
        slope2, intersect2 = linregress(warped2)[:2]

        diff = np.abs(slope1-slope2)

        if diff<diff_min:
            diff_min = diff
            pitch_guess = pitch

    if diff_min < 0.0001:
        H = get_homography(pitch)
        bv_img = warp_img(H, img)
        warped1 = warp(H, points1)
        warped2 = warp(H, points2)
        slope1, intersect1 = linregress(warped1)[:2]
        slope2, intersect2 = linregress(warped2)[:2]

        return pitch, bv_img, warped1, slope1, intersect1, warped2, slope2, intersect2,

    else:
        print('No pitch angle found')
        return None

def warp_img(H, img):
    x = np.arange(img.shape[0], dtype=np.float64)
    y = np.arange(img.shape[1], dtype=np.float64)

    Y,X = np.meshgrid(y,x)

    warped = warp(H,(X,Y))
    scaled = scale(warped)

    return scaled

def warp(H, points):
    w_points = np.zeros(points[0], 2, dtype=np.float64)
    for i in np.arange(points[0].size):
        warped = np.dot(H, np.array([points[0][i], points[1][i], 1]))
        warped /= warped[3]
        w_points[i] = warped
    return w_points


def get_homography(pitch):
    rot = R.from_euler('xyz', [0, pitch, 0], degrees=True).as_matrix()
    H = np.dot(K, rot)
    H = np.dot(H, np.linalg.inv(K))

    return H
