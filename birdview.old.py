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
    img = cv.imread('./00-08-52-932_points.png') # Read the test img
    # img = np.rot90(img)
    # cv.imwrite('malwiedertest.png',img)
    cnt = 0
    for p in np.arange(0.1, np.pi, 0.1):
        # for o in np.arange(0.9, 1, 0.1):
        # print(p)
        # rot = R.from_euler('xyz', (0, p, 0), degrees=False).as_matrix()
        rot = R.from_euler('xyz', (0, p, 0), degrees=False).as_matrix()
        # H = get_homography(rot = rot,
        #                t = np.array([0, 0, 0]),
        #                n = np.array([0, 1, 0])
        #                )

        H = get_homography2(rot, K)
        # print(H)

        # small_img = cv.resize(img, (np.int32(img.shape[1]/3), np.int32(img.shape[0]/3)))

        # warped_img = new_warp(small_img, H)

        # warped_img = my_warp3(img, H)

        # warped_img = my_warp3(img, H)
        # cv.imwrite('./my_warp/%s.png'%(cnt), warped_img)

        # warped_img = cv.resize(warped_img, (1920, 1080))
        # warped_img = cv.resize(warped_img, (np.int32(img.shape[1]/5), np.int32(img.shape[0]/5)))

        middle = np.where(np.all(img == [0,255,0], axis=-1))
        right = np.where(np.all(img == [255,0,0], axis=-1))
        new_height = np.amin([min(middle[0]),min(right[0])])
        new_width = np.amax([max(middle[1]),max(right[1])])

        img = img[new_height:, 0:new_width+1]
        # middle = np.where(np.all(img == [0,255,0], axis=-1))
        # right = np.where(np.all(img == [255,0,0], axis=-1))
        # print(img.shape)
        # quit()
        # print(middle)

        warped_img = my_warp4(img, H)
        # warped_img = maximum_filter(warped_img, footprint=np.ones((5, 3, 3)))
        # cv.imwrite('./my_warp/%s-2.png'%(cnt), warped_img)

        # left = np.where(np.all(warped_img == [0,0,255], axis=-1))
        # quit()

        middle_warp = np.where(np.all(warped_img == [0,255,0], axis=-1))
        # print(middle_warp)
        right_warp = np.where(np.all(warped_img == [255,0,0], axis=-1))
        # slope_m, intercept_m = linregress(middle_warp[0], middle_warp[1])[:2]
        # slope_r, intercept_r = linregress(right_warp[0], right_warp[1])[:2]
        # print(middle_warp)
        # print(right_warp)
        # print((slope_r, slope_m))

        # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_m), 0), pt2=(np.int32(slope_m*warped_img.shape[0] + intercept_m), warped_img.shape[0]), color=(0,255,0))
        # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_r), 0), pt2=(np.int32(slope_r*warped_img.shape[0] + intercept_r), warped_img.shape[0]), color=(255,0,0))

        middle = np.where(np.all(img == [0,255,0], axis=-1))
        # print(middle)
        right = np.where(np.all(img == [255,0,0], axis=-1))
        middle_warp = point_warp(middle, H, img)
        # print(middle_warp)
        # quit()
        right_warp = point_warp(right, H, img)
        # slope_l = linregress(left[0], left[1])[0]
        # pass x as y values and vice versa to avoid infinite slope
        slope_m, intercept_m = linregress(middle_warp[0], middle_warp[1])[:2]
        slope_r, intercept_r = linregress(right_warp[0], right_warp[1])[:2]

        # print(((np.int32(intercept_m), 0), (np.int32(slope_m*warped_img.shape[0] + intercept_m), warped_img.shape[0])))
        # print(((np.int32(intercept_r), 0), (np.int32(slope_r*warped_img.shape[0] + intercept_r), warped_img.shape[0])))
        # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_m), 0), pt2=(np.int32(slope_m*warped_img.shape[0] + intercept_m), warped_img.shape[0]), color=(0,255,0))
        # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_r), 0), pt2=(np.int32(slope_r*warped_img.shape[0] + intercept_r), warped_img.shape[0]), color=(255,0,0))

        cv.imwrite('./my_warp/%s-2.png'%(cnt), warped_img)
        print((slope_r, slope_m))


        tol = 1e-02
        if np.isclose(slope_r, slope_m, tol, equal_nan=False):
            print('ja nice!')
            print((slope_r,slope_m))
            # print(x)
            warped_img = my_warp4(img,H)

            # print(np.unique(warped_img, return_counts=True))
            warped_img = maximum_filter(warped_img, footprint=np.ones((20, 20, 3)))
            cv.imwrite('./parallel.png', warped_img)

            quit()
            # slope_l, slope_m, slope_r = 1,1,1
            # print((slope_l,slope_m,slope_r))
        cnt += 1

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

'''
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
    # Y -= np.int64(1920/4)
    H = np.linalg.inv(H)

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



def my_warp4(src, H):
    # height, width = 1000, 1000
    # width, height = 1000, 1000
    dst_height, dst_width = 1920, 1080
    height, width = src.shape[:2]
    # dst_points = np.zeros((height, width, 3), dtype=np.float64)
    # print((height,width))

    xstart = -height/2
    xend = xstart + height

    ystart = -3*width/16
    yend = ystart + width

    x_vec = np.arange(xstart, xend)
    y_vec = np.arange(ystart, yend)

    # print((xend,yend))
    # x_vec = np.arange(height)
    # y_vec = np.arange(width)

    # y_vec = np.arange(-width/4, 3*width/4)
    # y_vec = np.arange(-width,0)
    # x_vec = np.arange(-height,0)

    # x_vec = np.arange(-height/2, height/2)
    # y_vec = np.arange(-width/2, width/2)

    Y, X = np.meshgrid(y_vec, x_vec)

    # print(X.shape)
    # print(Y.shape)

    # a_vec = np.zeros((height, width))
    # b_vec = np.zeros((height, width))

    # print(a_vec.shape)
    # print(b_vec.shape)

    H = np.linalg.inv(H)
    # print(H)
    a_vec = np.float32((H[0,0]*X + H[0,1]*Y + H[0,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))# + height/2)
    b_vec = np.float32((H[1,0]*X + H[1,1]*Y + H[1,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))# + width/2)

    # a_vec -= np.amin(a_vec)
    # b_vec -= np.amin(b_vec)
    amin = np.amin(a_vec)
    bmin = np.amin(b_vec)
    # if amin < 0:
    #     a_vec -= amin
    #     amin = np.amin(a_vec)
    # if bmin < 0:
    #     b_vec -= bmin
    #     bmin = np.amin(b_vec)


    # print((np.amin(a_vec), np.amax(a_vec)))
    # print((np.amin(b_vec), np.amax(b_vec)))

    a_vec = ((a_vec - np.amin(a_vec)) / (np.amax(a_vec)-np.amin(a_vec))) * dst_height
    b_vec = ((b_vec - np.amin(b_vec)) / (np.amax(b_vec)-np.amin(b_vec))) * dst_width
    # a_vec = ((a_vec - np.amin(a_vec)) / (np.amax(a_vec)-np.amin(a_vec))) * src.shape[0]
    # b_vec = ((b_vec - np.amin(b_vec)) / (np.amax(b_vec)-np.amin(b_vec))) * src.shape[1]
    # print(a_vec.shape)
    # print(b_vec.shape)
    # print(np.amin(a_vec), np.amax(a_vec))
    # print(np.amin(b_vec), np.amax(b_vec))
    #
    # a_vec = cv.resize(a_vec,(1000,1000))
    # b_vec = cv.resize(b_vec,(1000,1000))
    # dst_points = cv.remap(src, b_vec, a_vec, 0)
    # print(dst_points.shape)



    dst_points = np.zeros((np.int32((np.amax(a_vec)+1, np.amax(b_vec)+1, 3))), dtype=np.uint8)
    pos = np.stack((a_vec, b_vec), 2).astype(np.uint32)
    # print(np.arange(dst_points.shape[1]))
    # print(pos[:,:,0].shape)

    # print(np.amax(pos[:,:,1]))
    # quit()
    # print(pos.shape)
    # print(dst_points.shape)
    # print(src.shape)
    #
    # print(np.amax(pos[:,:,0]))
    # print(np.amax(pos[:,:,1]))

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
