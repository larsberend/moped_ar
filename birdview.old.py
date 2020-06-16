import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.cluster.vq import kmeans,vq,whiten
from scipy.stats import linregress
from scipy.ndimage import maximum_filter
from draw_curve import pu,pv,u0,v0,f
from skimage.draw import line

IMAGE_H = 562
IMAGE_W = 1920

font = cv.FONT_HERSHEY_SIMPLEX
# matrices for camera intrinsics


def birdview():
    focal_mat = np.array([[f,0,0],[0,f,0], [0,0,1]], dtype=np.float64)
    pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
    K = np.dot(pixel_mat, focal_mat)
    img = cv.imread('./00-08-52-932_points_cut.png') # Read the test img

    cnt = 0
    for p in np.arange(0.0, np.pi, 0.1):
        # for o in np.arange(0.9, 1, 0.1):
        print(p)
        rot = R.from_euler('xyz', (0, 0.6, 0), degrees=False).as_matrix()
        # H = get_homography(rot = rot,
        #                t = np.array([0, 0, 0]),
        #                n = np.array([0, 1, 0])
        #                )

        H = get_homography2(rot, K)
        print(H)

        small_img = cv.resize(img, (np.int32(img.shape[1]/3), np.int32(img.shape[0]/3)))

        # warped_img = new_warp(small_img, H)

        # warped_img = my_warp3(img, H)

        # warped_img = my_warp3(img, H)
        # cv.imwrite('./my_warp/%s.png'%(cnt), warped_img)

        # warped_img = cv.resize(warped_img, (1920, 1080))
        # warped_img = cv.resize(warped_img, (np.int32(img.shape[1]/5), np.int32(img.shape[0]/5)))

        warped_img = my_warp4(img, H)
        warped_img = maximum_filter(warped_img, footprint=np.ones((5, 3, 3)))
        cv.imwrite('./my_warp/%s-3.png'%(cnt), warped_img)
        '''
        # left = np.where(np.all(warped_img == [0,0,255], axis=-1))
        middle = np.where(np.all(warped_img == [0,255,0], axis=-1))
        right = np.where(np.all(warped_img == [255,0,0], axis=-1))
        # slope_l = linregress(left[0], left[1])[0]
        slope_m = linregress(middle[0], middle[1])[0]
        slope_r = linregress(right[0], right[1])[0]

        tol = 1e-02
        if np.isclose(slope_r, slope_m, tol, equal_nan=False):
            print('ja nice!')
            print((slope_r,slope_m))
            print(x)
            cv.imwrite('./parallel.png'%(cnt), warped_img)

            quit()
            # slope_l, slope_m, slope_r = 1,1,1
            # print((slope_l,slope_m,slope_r))

        print((slope_r, slope_m))
        '''
        cnt += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            print(H)
            print(x)
            break
    # warped_img = cv.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    # cv.imwrite('./00-08-52-932_warp_lines_test.png', warped_img)
    # print(K)
    # print(np.linalg.inv(homo))
    # print(homo.T)

    # quit()


def my_warp3(src, H):
    # height, width = 1000, 1000
    width, height = 5000, 5000
    # height, width = src.shape[:2]
    dst_points = np.zeros((height, width, 3), dtype=np.float64)
    # print(H)

    cnt_tru=0

    x_vec = np.arange(-height/2, height/2)
    y_vec = np.arange(-width/2, width/2)

    Y, X = np.meshgrid(y_vec, x_vec)

    # print(X.shape)
    # print(Y.shape)

    a_vec = np.zeros((height, width))
    b_vec = np.zeros((height, width))

    # print(a_vec.shape)
    # print(b_vec.shape)

    # H = np.linalg.inv(H)

    a_vec = np.float32((H[0,0]*X + H[0,1]*Y + H[0,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]) + src.shape[0]/2)
    b_vec = np.float32((H[1,0]*X + H[1,1]*Y + H[1,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]) + src.shape[1]/2)



    dst_points = cv.remap(src, b_vec, a_vec, 0)
    # print(dst_points.shape)

    dst = cv.warpPerspective(src, H, (5000, 5000), flags=cv.WARP_INVERSE_MAP)
    cv.imwrite('./my_warp/cv.png', dst)

    return dst_points

def my_warp4(src, H):
    # height, width = 1000, 1000
    # width, height = 1000, 1000
    height, width = src.shape[:2]
    # dst_points = np.zeros((height, width, 3), dtype=np.float64)



    x_vec = np.arange(-height/2, height/2)
    y_vec = np.arange(-width/2, width/2)

    Y, X = np.meshgrid(y_vec, x_vec)

    # print(X.shape)
    # print(Y.shape)

    # a_vec = np.zeros((height, width))
    # b_vec = np.zeros((height, width))

    # print(a_vec.shape)
    # print(b_vec.shape)

    H = np.linalg.inv(H)
    print(H)
    a_vec = np.float32((H[0,0]*X + H[0,1]*Y + H[0,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]) + src.shape[0]/2)
    b_vec = np.float32((H[1,0]*X + H[1,1]*Y + H[1,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]) + src.shape[1]/2)

    # a_vec -= np.amin(a_vec)
    # b_vec -= np.amin(b_vec)
    amin = np.amin(a_vec)
    bmin = np.amin(b_vec)
    if amin < 0:
        a_vec -= amin
        amin = np.amin(a_vec)
    if bmin < 0:
        b_vec -= bmin
        bmin = np.amin(b_vec)


    print((np.amin(a_vec), np.amax(a_vec)))
    print((np.amin(b_vec), np.amax(b_vec)))

    # a_vec = ((a_vec - np.amin(a_vec)) / (np.amax(a_vec)-np.amin(a_vec))) * 1000
    # b_vec = ((b_vec - np.amin(b_vec)) / (np.amax(b_vec)-np.amin(b_vec))) * 1000
    # a_vec = ((a_vec - np.amin(a_vec)) / (np.amax(a_vec)-np.amin(a_vec))) * src.shape[0]
    # b_vec = ((b_vec - np.amin(b_vec)) / (np.amax(b_vec)-np.amin(b_vec))) * src.shape[1]
    print(a_vec.shape)
    print(b_vec.shape)
    print(np.amin(a_vec), np.amax(a_vec))
    print(np.amin(b_vec), np.amax(b_vec))
    #
    # a_vec = cv.resize(a_vec,(1000,1000))
    # b_vec = cv.resize(b_vec,(1000,1000))
    # dst_points = cv.remap(src, b_vec, a_vec, 0)
    # print(dst_points.shape)



    dst_points = np.zeros((np.int32((np.amax(a_vec)+1, np.amax(b_vec)+1, 3))))
    pos = np.stack((a_vec, b_vec), 2).astype(np.int32)
    # print(np.arange(dst_points.shape[1]))
    # print(pos[:,:,0].shape)

    # print(np.amax(pos[:,:,1]))
    # quit()
    print(pos.shape)
    print(dst_points.shape)
    print(src.shape)

    print(np.amax(pos[:,:,0]))
    print(np.amax(pos[:,:,1]))

    for i in np.arange(pos.shape[0]):
        for k in np.arange(pos.shape[1]):
            dst_points[pos[i,k,0], pos[i,k,1]] = src[i,k]

            # print(pos[i,k])
            # print(dst_points[pos[i,k,0], pos[i,k,1]])
            # print(dst_points[[735,1901]])
            # print(dst_points.shape)
            # quit()
            # print(src[i,k])
            # bool_arr = np.logical_and(pos[:,:,0] == i, pos[:,:,1]==k)
            # print(bool_arr.shape)
            # bool_arr = np.stack((bool_arr,bool_arr, bool_arr), 2)
            # print(np.where(bool_arr))
            # print(src.shape)
            # where = src[np.where(bool_arr)]
            # print(type(where))
            # if where.size > 0:
            #     dst_points[i,k] = where
            # # quit()
            # dst_points[bool_arr] = src[bool_arr]
            # np.where(bool_arr, src, dst_points+np.zeros_like(src))
            # dst_points[i,k] = src[pos[pos == [i,k]]]
        # print('%s of %s'%(i, pos.shape[0]))

    return dst_points






def my_warp_lanes(srcY, srcX, width, height, H):
    dstY =- width
    dstX =- height
    dstY, dstX = np.zeros_like(srcY), np.zeros_like(srcX)
    # print(H)
    # print((H[0,2], H[1,2], H[2,2]))
    # quit()
    for x,y in zip(dstX,dstY):
        a = (H[0,0]*x + H[0,1]*y + H[0,2])/(H[2,0]*x + H[2,1]*y + H[2,2])
        b = (H[1,0]*x + H[1,1]*y + H[1,2])/(H[2,0]*x + H[2,1]*y + H[2,2])

            # if a < height and a > 0 and b < width and b > 0:
            # if np.abs(a) < height/2 and np.abs(b) < width/2:
        dst_points[np.int32(x+height/2),np.int32(y+width/2)] = src[np.int32(a),np.int32(b)]
    return dst_points
    # from Gerhard Roth
def get_homography2(rot, K):
    print(rot)
    KR = np.dot(K, rot)
    print(KR)
    KRK = np.dot(KR, np.linalg.inv(K))
    # print(KRK)
    print('end of homography2')
    return KRK

def get_homography(rot, t, n):
    # print('here')
    # print(rot)
    # print(t)
    # print(n)
    # n = np.array([0,0,0])
    # t = np.array([0,0,0])
    # rot = R.from_euler('xyz', (0, 0, np.pi/4)).as_matrix()
    # print(rot)
    # print(np.dot(t, n.T))
    # print(rot)
    # print(R.from_matrix(rot).as_euler('xyz', degrees=False))
    print('rot')
    print(rot)


    n = n.reshape((3,1))
    t = t.reshape((3,1))

    print('tn')
    print(np.dot(t, n.T))

    H = rot + np.dot(t, n.T)
    H = H / H[2,2]
    print('H')
    print(H)
    return H


def get_warp_pos(left, middle, right):
    l_cand = [[0, 1, 2, 3], [0, 1, 2, 3]]
    m_cand = [[1, 2], [1,2]]
    r_cand = [[2, 3], [2,3]]
    return l_cand, m_cand, r_cand

if __name__ == '__main__':
    birdview()
