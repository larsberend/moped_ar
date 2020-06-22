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


def birdview(img, view):
    focal_mat = np.array([[f,0,0],[0,f,0], [0,0,1]], dtype=np.float64)
    pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
    K = np.dot(pixel_mat, focal_mat)

    cnt = 0
    for p in np.arange(0.1, np.pi, 0.1):
        # print(p)
        rot = R.from_euler('xyz', (0, p, 0), degrees=False).as_matrix()

        H = get_homography2(rot, K)
        # print(H)

        # small_img = cv.resize(img, (np.int32(img.shape[1]/3), np.int32(img.shape[0]/3)))

        middle = np.where(np.all(img == [0,255,0], axis=-1))
        right = np.where(np.all(img == [255,0,0], axis=-1))
        # left = np.where(np.all(warped_img == [0,0,255], axis=-1))

        new_height = np.amin([min(middle[0]),min(right[0])])
        new_width = np.amax([max(middle[1]),max(right[1])])


        if view:
            img = img[new_height:, 0:new_width+1]
            warped_img = warp_img(img, H)
            middle_warp = np.where(np.all(warped_img == [0,255,0], axis=-1))
            right_warp = np.where(np.all(warped_img == [255,0,0], axis=-1))
            # pass x as y values and vice versa to avoid infinite slope
            slope_m, intercept_m = linregress(middle_warp[0], middle_warp[1])[:2]
            slope_r, intercept_r = linregress(right_warp[0], right_warp[1])[:2]
            warped_img = cv.line(warped_img, pt1=(np.int32(intercept_m), 0), pt2=(np.int32(slope_m*warped_img.shape[0] + intercept_m), warped_img.shape[0]), color=(0,255,0))
            warped_img = cv.line(warped_img, pt1=(np.int32(intercept_r), 0), pt2=(np.int32(slope_r*warped_img.shape[0] + intercept_r), warped_img.shape[0]), color=(255,0,0))

            # cv.imwrite('./my_warp/%s-2.png'%(cnt), warped_img)




        else:
            middle = np.where(np.all(img == [0,255,0], axis=-1))
            right = np.where(np.all(img == [255,0,0], axis=-1))
            # print(middle)
            # print(right)
            middle_warp = point_warp(middle, H, img)
            right_warp = point_warp(right, H, img)
            # print(middle_warp)
            # print(right_warp)

            # slope_l = linregress(left[0], left[1])[0]
            # pass x as y values and vice versa to avoid infinite slope
            slope_m, intercept_m = linregress(middle_warp[0], middle_warp[1])[:2]
            slope_r, intercept_r = linregress(right_warp[0], right_warp[1])[:2]

            # print(((np.int32(intercept_m), 0), (np.int32(slope_m*warped_img.shape[0] + intercept_m), warped_img.shape[0])))
            # print(((np.int32(intercept_r), 0), (np.int32(slope_r*warped_img.shape[0] + intercept_r), warped_img.shape[0])))

        print((slope_r, slope_m))


        tol = 1e-03
        if np.isclose(slope_r, slope_m, tol, equal_nan=False):
            print('ja nice!')
            print((slope_r,slope_m))
            warped_img = warp_img(img,H)

            # print(np.unique(warped_img, return_counts=True))
            cv.imwrite('./parallel.png', warped_img)
            return p

        cnt += 1
    print('No fitting angle found.')
    return None

def point_warp(points, H, img):
    # X = np.array([0, points[0], img.shape[0]])
    # print(X)
    # quit()
    X, Y = np.float64(points)

    xmin, ymin = -img.shape[0]/2, -img.shape[1]/2
    xmax, ymax = xmin + img.shape[0], ymin + img.shape[1]

    X += xmin
    Y += ymin

    # to visualize: insert min/max to fit to image
    X = np.append(X, [xmin, xmax])
    Y = np.append(Y, [ymin, ymax])

    # print(X.shape)
    # print(Y.shape)
    # print((X,Y))

    a_vec = np.float64((H[0,0]*X + H[0,1]*Y + H[0,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))
    b_vec = np.float64((H[1,0]*X + H[1,1]*Y + H[1,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))

    # normalize to image
    # a_vec = ((a_vec - np.amin(a_vec)) / (np.amax(a_vec)-np.amin(a_vec))) * 1920
    # b_vec = ((b_vec - np.amin(b_vec)) / (np.amax(b_vec)-np.amin(b_vec))) * 1080

    # print(X)
    # print(a_vec[:-2])
    # quit()
    warped_points = a_vec[:-2], b_vec[:-2]


    return warped_points



def warp_img(src, H):
    # height, width = 1000, 1000
    # width, height = 1000, 1000
    dst_height, dst_width = 1920, 1080
    height, width = src.shape[:2]
    # print((height,width))

    xstart = -height/2
    xend = xstart + height

    ystart = -3*width/16
    yend = ystart + width

    x_vec = np.arange(xstart, xend)
    y_vec = np.arange(ystart, yend)

    Y, X = np.meshgrid(y_vec, x_vec)

    # print(X.shape)
    # print(Y.shape)
    # print(H)

    a_vec = np.float32((H[0,0]*X + H[0,1]*Y + H[0,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))# + height/2)
    b_vec = np.float32((H[1,0]*X + H[1,1]*Y + H[1,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))# + width/2)

    amin = np.amin(a_vec)
    bmin = np.amin(b_vec)

    # print((np.amin(a_vec), np.amax(a_vec)))
    # print((np.amin(b_vec), np.amax(b_vec)))

    a_vec = ((a_vec - np.amin(a_vec)) / (np.amax(a_vec)-np.amin(a_vec))) * dst_height
    b_vec = ((b_vec - np.amin(b_vec)) / (np.amax(b_vec)-np.amin(b_vec))) * dst_width

    # print(a_vec.shape)
    # print(b_vec.shape)
    # print(np.amin(a_vec), np.amax(a_vec))
    # print(np.amin(b_vec), np.amax(b_vec))
    # print(dst_points.shape)



    dst_points = np.zeros((np.int32((np.amax(a_vec)+1, np.amax(b_vec)+1, 3))), dtype=np.uint8)
    pos = np.stack((a_vec, b_vec), 2).astype(np.uint32)

    # print(pos.shape)
    # print(dst_points.shape)
    # print(src.shape)

    for i in np.arange(pos.shape[0]):
        for k in np.arange(pos.shape[1]):
            dst_points[pos[i,k,0], pos[i,k,1]] = src[i,k]

    return dst_points


# from Gerhard Roth
def get_homography2(rot, K):
    print(rot)
    KR = np.dot(K, rot)
    print(KR)
    KRK = np.dot(KR, np.linalg.inv(K))
    # print(KRK)
    print('end of homography2')

    return np.linalg.inv(KRK)

# old version of homography, does not yield satisfying results.
def get_homography(rot, t, n):
    n = n.reshape((3,1))
    t = t.reshape((3,1))

    H = rot + np.dot(t, n.T)
    H = H / H[2,2]
    return H

def mark_lanes(img):





if __name__ == '__main__':
    img = cv.imread('./00-08-52-932_points.png') # Read the test img
    mark_lanes(img)
    # birdview(img, False)
