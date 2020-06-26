import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.cluster.vq import kmeans,vq,whiten
from scipy.stats import linregress
from scipy.ndimage import maximum_filter
from draw_curve import pu,pv,u0,v0,f
from skimage.draw import line
import skimage.transform
from sklearn.cluster import KMeans
from sklearn import linear_model
from mark_lanes import mark_lanes

IMAGE_H = 562
IMAGE_W = 1920

font = cv.FONT_HERSHEY_SIMPLEX
focal_mat = np.array([[f,0,0],[0,f,0], [0,0,1]], dtype=np.float64)
pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
K = np.dot(pixel_mat, focal_mat)


def birdview(img, view, last_angle):

    yellow = np.where(np.all(img == [0,255,255], axis=-1))
    blue = np.where(np.all(img == [255,0,0], axis=-1))
    # left = np.where(np.all(warped_img == [0,0,255], axis=-1))
    # print(yellow)
    # print(blue)
    # quit()
    # for x,y in zip(yellow[0], yellow[1]):
    #     img = cv.circle(img, (y,x), 10,  (0,0,255))
    # cv.imwrite('cirles.png', img)
    new_height = np.amin([min(yellow[0]),min(blue[0])])
    new_width = np.amax([max(yellow[1]),max(blue[1])])

    img = img[new_height:, 0:new_width+1]

    yellow = np.where(np.all(img == [0,255,255], axis=-1))
    blue = np.where(np.all(img == [255,0,0], axis=-1))
    # yellow = np.where(np.all(img == [0,204,204], axis=-1))
    # blue = np.where(np.all(img == [204,204,0], axis=-1))
    warped_img, angle, found = iter_angle(last_angle, img, view, yellow, blue)

    if found:
        return warped_img, angle, True
    else:
        warped_img, angle, found = iter_angle(angle, img, view, yellow, blue)
        if found:
            return warped_img, angle, True
        else:
            print('No fitting angle found')
            return warped_img, None, False

def iter_angle(last_angle, img, view, yellow, blue):
    warped_img = np.zeros((img.shape[1], img.shape[0],3))
    cnt = 0
    # print(search_grid.shape)
    angle_guess = 0
    smallest_diff = 100

    if last_angle is None:
        search_grid = np.arange(0, np.pi/4, 0.001)
    else:
        search_grid = np.arange(last_angle-0.01, last_angle+0.01, 0.0001)


    for angle in search_grid:
        # print(p)
        # angle = np.radians(40)
        rot = R.from_euler('xyz', (0, angle, 0), degrees=False).as_matrix()
        H = get_homography2(rot, K)

        # if view:
        #     warped_img = warp_img(img, H)
        #     yellow_warp = np.where(np.all(warped_img == [0,255,255], axis=-1))
        #     blue_warp = np.where(np.all(warped_img == [255,0,0], axis=-1))
        #
        #     slope_y, intercept_y = my_ransac(yellow_warp)
        #     slope_b, intercept_b = my_ransac(blue_warp)
        #     # print(slope_y)
        #     # print(slope_b)
        #
        #     warped_img = cv.line(warped_img, pt1=(np.int32(intercept_y), 0), pt2=(np.int32(slope_y*warped_img.shape[0] + intercept_y), warped_img.shape[0]), color=(0,255,0))
        #     warped_img = cv.line(warped_img, pt1=(np.int32(intercept_b), 0), pt2=(np.int32(slope_b*warped_img.shape[0] + intercept_b), warped_img.shape[0]), color=(255,0,0))
        #
        #     cv.imwrite('./my_warp/%s-3.png'%(cnt), warped_img)


        # else:


        # print(yellow)
        # print(blue)
        yellow_warp = point_warp(yellow, H, img)
        blue_warp = point_warp(blue, H, img)

        slope_y, intercept_y = linregress(yellow_warp)[:2]
        slope_b, intercept_b = linregress(blue_warp)[:2]


            # print(yellow_warp)
            # print(blue_warp)

            # slope_l = linregress(left[0], left[1])[0]
            # pass x as y values and vice versa to avoid infinite slope
            # slope_y, intercept_y = linregress(yellow_warp[0], yellow_warp[1])[:2]
            # slope_b, intercept_b = linregress(blue_warp[0], blue_warp[1])[:2]
            # print(slope_y)

            # slope_y, intercept_y = my_ransac(yellow_warp, False)
            # slope_b, intercept_b = my_ransac(blue_warp, False)

            # slope_y, intercept_y, yellow_rans_X, yellow_rans_Y = my_ransac(yellow_warp, True)
            # slope_b, intercept_b, blue_rans_X, blue_rans_Y = my_ransac(blue_warp, True)
            # slope_y, intercept_y, yellow_rans = my_ransac(yellow_warp, True)
            # slope_b, intercept_b, blue_rans = my_ransac(blue_warp, True)


            # print(slope_b)
            # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_y), 0), pt2=(np.int32(slope_y*warped_img.shape[0] + intercept_y), warped_img.shape[0]), color=(0,255,0))
            # cv.imwrite('./guckstehier.png', warped_img)
            # quit()

            # print(((np.int32(intercept_m), 0), (np.int32(slope_m*warped_img.shape[0] + intercept_m), warped_img.shape[0])))
            # print(((np.int32(intercept_r), 0), (np.int32(slope_r*warped_img.shape[0] + intercept_r), warped_img.shape[0])))

        # print((slope_y, slope_b))
        diff = np.abs(slope_b-slope_y)
        if diff < smallest_diff:
            smallest_diff = diff
            angle_guess = angle
            # print((angle_guess,smallest_diff))
        tol = 1e-04
        # tol = 100
        if np.isclose(smallest_diff, 0, rtol=1, atol=tol, equal_nan=False):
            # print(smallest_diff)
            # print('ja nice!')
            # print((slope_y,slope_b))
            # warped_img = warp_img(img,H)
            #
            # # print(np.unique(warped_img, return_counts=True))
            # cv.imwrite('./parallel.png', warped_img)
            # print(yellow)
            # print(yellow_warp)
            # Hinv = np.linalg.inv(H)
            # print(yellow_rans)
            # ransac_rwarp_yellow = point_warp(yellow_rans, Hinv, img)
            # ransac_rwarp_blue = point_warp(blue_rans, Hinv, img)
            # print(ransac_rwarp_yellow)
            # quit()
            # print(yellow)
            # slope_y_rwarp, intercept_y_rwarp, ransac_rwarp_yellow = my_ransac(yellow, True)
            # slope_b_rwarp, intercept_b_rwarp, ransac_rwarp_blue = my_ransac(blue, True)
            warped_img = warp_img(img, H)
            '''
            yellow_warp = np.where(np.all(warped_img == [0,255,255], axis=-1))
            blue_warp = np.where(np.all(warped_img == [255,0,0], axis=-1))

            sy2, iy2 = linregress(yellow_warp[0], yellow_warp[1])[:2]
            sb2, ib2 = linregress(blue_warp[0], blue_warp[1])[:2]
            print((sy2,iy2))
            print(warped_img.shape)
            print((intercept_y,slope_y))
            print((intercept_b,slope_b))
            warped_img = cv.line(warped_img, pt1=(np.int32(iy2), 0), pt2=(np.int32(sy2*warped_img.shape[0] + iy2), warped_img.shape[0]), color=(0,255,0))
            warped_img = cv.line(warped_img, pt1=(np.int32(ib2), 0), pt2=(np.int32(sb2*warped_img.shape[0] + ib2), warped_img.shape[0]), color=(255,0,0))
            '''

            # quit()
            # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_y), 0), pt2=(np.int32(slope_y*warped_img.shape[0] + intercept_y), warped_img.shape[0]), color=(0,255,0))
            # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_b), 0), pt2=(np.int32(slope_b*warped_img.shape[0] + intercept_b), warped_img.shape[0]), color=(255,0,0))
            cv.imwrite('schaunwirmal6.png', warped_img)
            # quit()
            # _, _, ransac_rwarp_blue = my_ransac(blue, True)
            # print(img.shape)
            # print(intercept_y_rwarp)
            # img = cv.line(img, pt1=(np.int32(intercept_y_rwarp), 0), pt2=(np.int32(slope_y_rwarp*img.shape[0] + intercept_y_rwarp), img.shape[0]), color=(0,255,0), thickness=5)
            # img = cv.line(img, pt1=(np.int32(intercept_b_rwarp), 0), pt2=(np.int32(slope_b_rwarp*img.shape[0] + intercept_b_rwarp), img.shape[0]), color=(255,0,0), thickness=5)



            # ransac_rwarp_yellow = np.int32(ransac_rwarp_yellow)
            # ransac_rwarp_blue = np.int32(ransac_rwarp_blue)

            # print(ransac_rwarp_yellow)
            # print(ransac_rwarp_yellow.shape)
            # quit()
            # for x,y in zip(ransac_rwarp_yellow[0], ransac_rwarp_yellow[1]):
            #     # print((x,y))
            #     img = cv.circle(img, (y,x), 5, (0,255,0), -1)
            # cv.imwrite('wasisthierlos.png', img)
            # cv.line(img, (ransac_rwarp_yellow[1][0], ransac_rwarp_yellow[0][0]), (ransac_rwarp_yellow[1][-1], ransac_rwarp_yellow[0][-1]), (0,255,0), 5)
            # cv.line(img, (ransac_rwarp_blue[0][1], ransac_rwarp_blue[0][0]), (ransac_rwarp_blue[-1][0], ransac_rwarp_blue[-1][1]), (255,0,0), 5)

            cv.putText(warped_img, 'Pitch Angle: %s'%(np.degrees(angle)), (10, 1650), font, 0.5, (255, 255, 0), 2, cv.LINE_AA)
            cv.putText(warped_img, 'Smallest diff: %s'%(smallest_diff), (10, 1670), font, 0.5, (255, 255, 0), 2, cv.LINE_AA)
            return warped_img, angle_guess, True

        cnt += 1
    return warped_img, angle_guess, False


def my_ransac(warped_p, return_points):
    ransac = linear_model.RANSACRegressor()

    # pass x as y values and vice versa to avoid infinite slope

    ransac.fit(warped_p[0].reshape(len(warped_p[0]), 1), warped_p[1])

    line_ransac = ransac.predict(warped_p[0].reshape(len(warped_p[0]), 1))


    slope, intercept = linregress(warped_p[0], line_ransac)[:2]
    # print(slope)
    if return_points:
        return slope, intercept, (warped_p[0], line_ransac)
    else:
        return slope, intercept

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
    # print(rot)
    KR = np.dot(K, rot)
    # print(KR)
    KRK = np.dot(KR, np.linalg.inv(K))
    # print(KRK)
    # print('end of homography2')

    return np.linalg.inv(KRK)


if __name__ == '__main__':
    # img = cv.imread('./00-08-52-932_points.png') # Read the test img
    # img = cv.imread('./3_00-01-48.png') # Read the test img
    # img, roll_angle= cv.imread('./problematic.png'), 0.298919415637517
    # img, roll_angle= cv.imread('./problematic2.png'), 0.171368206765908
    img, roll_angle= cv.imread('./problematic3.png'), -0.340036930698958

    # rotate image by specified roll angle


    marked_img, retval = mark_lanes(img, -roll_angle)
    # print(retval)
    warped_img, angle, found = birdview(marked_img, False, None)
    # cv.imwrite('aha.png', warped_img)
