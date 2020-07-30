import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
from camera_K import K
from scipy.stats import linregress
import matplotlib.pyplot as plt

def find_bv(img, color1, color2):
    points1 = np.where(np.all(img==color1, axis=-1))
    points2 = np.where(np.all(img==color2, axis=-1))

    # points1[0] sind y koordinaten

    points1 =(points1[1], points1[0])
    points2 =(points2[1], points2[0])
    # print(points1)
    # print(points2)
    diff_min = 100
    pitch_guess = 0
    height, width = img.shape[:2]

    for pitch in np.arange(0, -110, -10):
        H = get_homography(pitch)
        # print(H)
        # H = np.identity(3)
        # points1 = (np.array([1,2,3]),np.array([100,200,300]))
        # warped1 = warp(H, points1, height, width)
        # print(points1)
        # print(warped1)
        # quit()

        # warped1[:,0] are x coords
        warped1 = warp(H, points1, height, width)
        warped2 = warp(H, points2, height, width)

        # print(warped1)
        # print(points1)
        print(H)


        # points1 = np.stack((points1[1], points1[0]), axis=1)
        # cvH = cv.findHomography(points1, warped1)[0]
        # cvH = cv.findHomography(warped1, points1)[0]
        # print('cvH:')
        # print(cvH)
        # if cvH is not None:
        #     cvRot = R.from_matrix(get_rot(cvH))
        #     print(cvRot.as_matrix())
        #     print(cvRot.as_euler('xyz', degrees=True))
        # quit()


        slope1, intersect1 = linregress(warped1[:,0], warped1[:,1])[:2]
        slope2, intersect2 = linregress(warped2[:,0], warped2[:,1])[:2]

        warp_img(H,img)
        plt.scatter(warped1[:,1], warped1[:,0], color='orange')
        plt.scatter(warped2[:,1], warped2[:,0], color='green')
        plt.savefig('warp_plots/%s.png'%(pitch))
        plt.close()
        print(slope1,slope2)

        diff = np.abs(slope1-slope2)

        if diff<diff_min:
            diff_min = diff
            pitch_guess = pitch
    quit()

    for pitch in np.arange(pitch_guess-1, pitch_guess+1, 0.01):
        H = get_homography(pitch)
        warped1 = warp(H, points1, height, width)
        warped2 = warp(H, points2, height, width)
        slope1, intersect1 = linregress(warped1[:,1], warped1[:,0])[:2]
        slope2, intersect2 = linregress(warped2[:,1], warped2[:,0])[:2]

        diff = np.abs(slope1-slope2)
        if diff<diff_min:
            diff_min = diff
            pitch_guess = pitch
    if diff_min < 0.01:
        print(pitch_guess)
        H = get_homography(pitch)
        bv_img = warp_img(H, img)
        warped1 = warp(H, points1, height, width)
        warped2 = warp(H, points2, height, width)
        slope1, intersect1 = linregress(warped1)[:2]
        slope2, intersect2 = linregress(warped2)[:2]

        return pitch_guess, bv_img, warped1, slope1, intersect1, warped2, slope2, intersect2,

    else:
        print('No pitch angle found, (best guess, min diff):')
        print(pitch_guess, diff_min)
        return None

def warp_img(H, img):
    u = np.arange(img.shape[0], dtype=np.float64)
    v = np.arange(img.shape[1], dtype=np.float64)
    # print(x)
    # print(y)
    X, Y = np.meshgrid(u,v)
    # print(X.shape)
    # print(Y.shape)


    warped = np.zeros((X.shape[0], X.shape[1], 2))
    for i in np.arange(X.shape[0]):
        warped[i] = warp(H, (Y[i], X[i]), img.shape[0], img.shape[1])

    # print((X,Y))
    # print(warped[:,:,0].shape)
    # quit()
    plt.scatter(Y,X, color='blue')
    plt.scatter(warped[:,:,1], warped[:,:,0], color='red')
    # plt.savefig('warp_plots/%s.png'%(angle))

    # scaled = scale(warped)



    return None

def warp(H, points, height, width):
    w_points = np.zeros((points[0].size, 2), dtype=np.float64)
    points_y = points[0].astype(np.float64) - width/2
    points_x = points[1].astype(np.float64)# +  height/2
    for i in np.arange(points[0].size):
        warped = np.dot(H, np.array([points_x[i], points_y[i], 1]))
        # if np.abs(warped[2])<0.001:
        #     print(warped[2])
        warped /= warped[2]

        w_points[i] = warped[:2]
    return w_points

def scale(pos, width=4000, height=4000):
    print(pos.shape)
    print(pos[:,:,0].shape)
    print(pos[:,:,1].shape)
    bv = np.zeros((width, height))

    print(np.amin(pos[:,:,0]))
    print(np.amax(pos[:,:,0]))
    print(np.amin(pos[:,:,1]))
    print(np.amax(pos[:,:,1]))
    # quit()
    return bv


def get_homography(pitch):
    # rot = R.from_euler('xyz', [0, pitch, 0], degrees=True).as_matrix()
    rot = R.from_euler('xyz', [0, pitch, 0], degrees=True).as_matrix()
    print('rot')
    print(rot)
    # print(K)
    KR = np.dot(K, rot)
    # print(KR)
    KRK = np.dot(KR, np.linalg.inv(K))
    # print(KRK)
    #
    # KR = np.dot(KRK, K)
    # print(KR)
    # rot = np.dot(np.linalg.inv(K), KR)
    # print(rot)
    # quit()

    # print(np.dot(K, np.linalg.inv(K)))
    # quit()
    # return np.linalg.inv(KRK)
    return KRK# / KRK[2,2]
    # return rot

def get_rot(H):
    KR = np.dot(H, K)
    rot = np.dot(np.linalg.inv(K), KR)
    return rot
