import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.cluster.vq import kmeans,vq,whiten
from scipy.stats import linregress
from draw_curve import pu,pv,u0,v0,f
from skimage.draw import line

IMAGE_H = 562
IMAGE_W = 1920

# matrices for camera intrinsics
pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
focal_mat = np.array([[f,0,0],[0,f,0], [0,0,1]], dtype=np.float64)

K = np.dot(pixel_mat, focal_mat)

def birdview():
    img = cv.imread('./00-08-52-932_lines2.png') # Read the test img

    left = np.where(np.all(img == [0,0,255], axis=-1))
    middle = np.where(np.all(img == [0,255,0], axis=-1))
    right = np.where(np.all(img == [255,0,0], axis=-1))

    # left = np.column_stack((left[0], left[1]))#.astype(np.float64)
    # middle = np.column_stack((middle[0], middle[1]))#.astype(np.float64)
    # right = np.column_stack((right[0], right[1]))#.astype(np.float64)

    inter_l = linregress(left[1], left[0])
    inter_m = linregress(middle[1], middle[0])
    inter_r = linregress(right[1], right[0])

    test_reg = linregress([0,1], [1,2])
    print(inter_l)
    print(test_reg)

    tol = 1e-08
    slope_l, slope_m, slope_r = 1,2,3
    while not(np.isclose(slope_l, slope_m, tol, equal_nan=False)) or not(np.isclose(slope_l, slope_r, tol, equal_nan=False)):
        # print((slope_l,slope_m,slope_r))
        # slope_l, slope_m, slope_r = 1,1,1
        # print((slope_l,slope_m,slope_r))

        l_cand, m_cand, r_cand  = get_warp_pos(left, middle, right)#(irgenwie bekomm ich hier kandidaten fuer punkte, die die birdview linien zeigen)
        slope_l = linregress(l_cand[0], l_cand[1])[0]
        slope_m = linregress(m_cand[0], m_cand[1])[0]
        slope_r = linregress(r_cand[0], r_cand[1])[0]

    print('out of while')
    src = np.float32([[889, 577], [515, 902], [1018, 570], [1566, 1079]])
    dst = np.float32([[515, 577], [515, 902], [1566, 570], [1566, 1079]])
    # left = (np.array([0, 10,20,30]), np.array([0, 10,20,30]))
    # left = np.column_stack((left[0], left[1]))
    # l_cand = np.column_stack((l_cand[0], l_cand[1]))
    # print(left)
    # print(l_cand)
    homo, _ = cv.findHomography(src, dst)
    # homo, _ = cv.findHomography([left,middle,right], [l_cand, m_cand, r_cand])

    K  = np.identity(3)
    retval, rotations, translations, normals = cv.decomposeHomographyMat(homo, K)

    H = get_homography(np.linalg.inv(rotations[0]), -translations[0], -normals[0])
    # print((rotations[0], translations[0], normals[0]))
    print(-H)
    print(homo)
    # print(K)
    # print(np.linalg.inv(homo))
    # print(homo.T)

    quit()

def get_warp_pos(left, middle, right):
    l_cand = [[0, 1, 2, 3], [0, 1, 2, 3]]
    m_cand = [[1, 2], [1,2]]
    r_cand = [[2, 3], [2,3]]
    return l_cand, m_cand, r_cand
def get_homography(rot, t, n):
    # n = np.array([1,0,0])
    # t = np.array([1,0,1])
    # rot = R.from_euler('xyz', (0, rot, X)).as_matrix()

    H = rot + np.dot(t, n.T)

    return H


# whitened = whiten(src)
# code_book, distortion = kmeans(whitened, 3)
# clusters, dist = vq(whitened, code_book)
# print(clusters)
# test_im = np.zeros_like(img)
# print(np.max(src))
# print(src.shape)
# test_im[src[:,0],src[:,1]] = [255,255,255]
# test_im[src[clusters==0,0],src[clusters==0,1]] = [255,0,255]
# test_im[src[clusters==1,0],src[clusters==1,1]] = [255,0,0]
# test_im[src[clusters==2,0],src[clusters==2,1]] = [0,0,255]

# cv.imshow('test', test_im)
# cv.imwrite('./test.png', test_im)
#
# # src = np.float32([[0, 1080], [1920, 1080], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
# # dst = np.float32([[780, IMAGE_H], [1020, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# M = cv.getPerspectiveTransform(src, dst) # The transformation matrix
# Minv = cv.getPerspectiveTransform(dst, src) # Inverse transformation
# homo, _ = cv.findHomography(src,dst)
# retval, rotations, translations, normals = cv.decomposeHomographyMat(homo, K)
#
#
# # print(M)
# # print(np.dot(src,M))
# # print(Minv)
# # print(homo)
# for r, t, n in zip(rotations, translations, normals):
#     print(R.from_matrix(r).as_euler('xyz', degrees=True))
#     print(t)
#     print(n)
# # print(rotations)
# # print(translations)
# # print(normals)
#
#
#
# # img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
# warped_img = cv.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
# # cv.imwrite('./00-08-52-932_warp_lines.png', warped_img)
# plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)) # Show results
# plt.show()

if __name__ == '__main__':
    birdview()
