import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn import linear_model
from mark_lanes import mark_lanes, pu, pv, u0, v0, f
import skimage.transform
from scipy.ndimage import maximum_filter



font = cv.FONT_HERSHEY_SIMPLEX
focal_mat = np.array([[f,0,0],[0,f,0], [0,0,1]], dtype=np.float64)
pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
K = np.dot(pixel_mat, focal_mat)


def birdview(img, view, last_angle):

    # # find points on lane markings in image
    yellow = np.where(np.all(img == [0,255,255], axis=-1))
    blue = np.where(np.all(img == [255,0,0], axis=-1))
    # cut image to location of markings
    new_height = np.amin([min(yellow[0]),min(blue[0])])
    new_width = np.amax([max(yellow[1]),max(blue[1])])
    orig_img = img.copy()
    img = img[new_height:, 0:new_width+1]
    yellow = np.where(np.all(img == [0,255,255], axis=-1))
    blue = np.where(np.all(img == [255,0,0], axis=-1))

    cut_to_road = orig_img[int(630):]
    img = cut_to_road
    warped_img, angle, found, slope, inter1, inter2 = iter_angle(last_angle, cut_to_road, view, yellow, blue)

    # check if an angle was found
    if found:
        return warped_img, angle, True, slope, inter1, inter2
    # else search for it again with best guess
    else:
        warped_img, angle, found, slope, inter1, inter2 = iter_angle(angle, img, view, yellow, blue)

        # fin_rot = R.from_euler('xyz', (0, angle, 0), degrees=False).as_matrix()
        # H = get_homography2(fin_rot, K)
        # cut_to_road = orig_img[int(630):]
        # big_warp = warp_img(cut_to_road, H)
        # cv.imwrite('schaunwirmal8.png', cut_to_road)
        # print(cut_to_road.shape)
        # big_warp = warp_img(cut_to_road, H)
        # big_warp = warp_img(cut_to_road, H, False)
        # cv.imwrite('schaunwirmal9.png', big_warp)

        if found:
            return warped_img, angle, True, slope, inter1, inter2
            # return big_warp, angle, True, slope, inter1, inter2
            # return warped_img, angle, True

        else:
            print('No fitting angle found')
            return warped_img, angle, False, slope, inter1, inter2

# calculate birdviews around a certain angle or in range [0, pi/4] if input angle==None
def iter_angle(last_angle, img, view, yellow, blue):
    warped_img = np.zeros((img.shape[1], img.shape[0], 3))
    cnt = 0
    # print(search_grid.shape)
    angle_guess = 0
    smallest_diff = 100
    slope_y, slope_b, intercept_y, intercept_b = None, None, None, None
    if last_angle is None:
        search_grid = np.arange(np.radians(30), np.radians(50), np.radians(0.1))
    else:
        search_grid = np.arange(last_angle-0.1, last_angle+0.1, 0.00001)


    for angle in search_grid:
        # print(p)
        # angle = np.radians(40)

        # convert rad angle to rotation matrix
        # print(np.degrees(angle))
        # angle = np.radians(2)
        # angle *= -1
        # print(angle)



        angle = 0.6626151157573931
        # angle = 0



        rot = R.from_euler('xyz', (0, angle, 0), degrees=False).as_matrix()
        H = get_homography2(rot, K)

        if view:
            # angle = 0.2
            warped_img = warp_img(img, H, angle, inv=False)
            warped_img = warp_img(img, H, angle, inv=True)
            # quit()
            # warped_img = warp_img(img, H)
            # yellow_warp = np.where(np.all(warped_img == [0,255,255], axis=-1))
            # blue_warp = np.where(np.all(warped_img == [255,0,0], axis=-1))
            #
            # slope_y, intercept_y = my_ransac(yellow_warp, H)
            # slope_b, intercept_b = my_ransac(blue_warp, H)
            # # print(slope_y)
            # # print(slope_b)
            #
            # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_y), 0), pt2=(np.int32(slope_y*warped_img.shape[0] + intercept_y), warped_img.shape[0]), color=(0,255,0))
            # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_b), 0), pt2=(np.int32(slope_b*warped_img.shape[0] + intercept_b), warped_img.shape[0]), color=(255,0,0))
            # if cnt == 0:
                # cv.imwrite('./my_warp_40/2_grad.png', img)
                # quit()
            # warped_img = maximum_filter(warped_img, size=(30,1,1))
            cv.imwrite('./my_warp/%s.png'%(cnt), warped_img)
            # cv.imwrite('./my_warp_40/%s.png'%(cnt), warped_img)
            cnt += 1
            # quit()
        else:
            yellow_warp = point_warp(yellow, H, img)
            blue_warp = point_warp(blue, H, img)

            slope_y, intercept_y = linregress(yellow_warp)[:2]
            slope_b, intercept_b = linregress(blue_warp)[:2]

            # print((slope_y, slope_b))
            # if sopes are equal, lines parallel in birdview.
            # get best guess for angle
            diff = np.abs(slope_b-slope_y)
            if diff < smallest_diff:
                smallest_diff = diff
                angle_guess = angle
                # print((angle_guess,smallest_diff))
            # check, if diff is smaller than 0.0001
            tol = 1e-03



            if True:



            # if np.isclose(smallest_diff, 0, rtol=1, atol=tol, equal_nan=False):
                # make birdview image and save
                # print(img.shape)
                print('here')
                print(angle_guess)
                print(smallest_diff)
                # warped_img = warp_img(img, H, angle_guess, False)
                warped_img = warp_img(img, H, angle_guess, True)
                cv.imwrite('schaunwirmal6.png', warped_img)
                cv.putText(warped_img, 'Pitch Angle: %s'%(np.degrees(angle)), (10, 1650), font, 0.5, (255, 255, 0), 2, cv.LINE_AA)
                cv.putText(warped_img, 'Smallest diff: %s'%(smallest_diff), (10, 1670), font, 0.5, (255, 255, 0), 2, cv.LINE_AA)
                return warped_img, angle_guess, True, slope_b, intercept_y, intercept_b
        # if no good angle found, return False and best guess
    # quit()
    return warped_img, angle_guess, False, slope_b, intercept_y, intercept_b


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

# warp points with specified Homography
def point_warp(points, H, img):
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

    # remove min and max from output
    warped_points = a_vec[:-2], b_vec[:-2]

    return warped_points

# fucntion to visualize & verify
# warp all pixels of image-exerpt in birdview
def warp_img(src, H, angle=None, inv=True):
    print('warp_im')
    print(inv)
    # declare output image size
    # height, width = 1000, 1000
    dst_height, dst_width = 4000, 4000
    # dst_height, dst_width = 1920, 1080
    height, width = src.shape[:2]
    # print((height,width))

    if inv == True:
        # move rotation centre of image, such that output image is more centered
        xstart = -height/2
        # xstart = 0
        xend = xstart + height

        ystart = -width/2
        # ystart = 0
        yend = ystart + width

        x_vec = np.arange(xstart, xend)
        y_vec = np.arange(ystart, yend)

        Y, X = np.meshgrid(y_vec, x_vec)

        # print(X.shape)
        # print(Y.shape)
        # print(H)
        # https://docs.opencv.org/4.2.0/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
        a_vec = np.float64((H[0,0]*X + H[0,1]*Y + H[0,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))# + height/2)
        b_vec = np.float64((H[1,0]*X + H[1,1]*Y + H[1,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))# + width/2)

        amin = np.amin(a_vec)
        bmin = np.amin(b_vec)
        # print((np.amin(a_vec), np.amax(a_vec)))
        # print((np.amin(b_vec), np.amax(b_vec)))

        # if np.any(a_vec < 0):
        #     print(angle)
        #     quit()
        # print(a_vec.shape)
        # quit()
        # print(X.shape)
        # print(Y.shape)
        # quit()
        # print(X)
        # quit()
        ymin, ymax = np.unravel_index(np.argmin(a_vec), a_vec.shape), np.unravel_index(np.argmax(a_vec), a_vec.shape)
        xmin, xmax = np.unravel_index(np.argmin(b_vec), b_vec.shape), np.unravel_index(np.argmax(b_vec), b_vec.shape)
        # xmin, xmax = np.argmin(b_vec), np.argmax(b_vec)
        # print((xmin,xmax))

        print('Output:')
        print('minX, maxX')
        print((b_vec[xmin], b_vec[xmax]))
        print('minY, maxY')
        print((a_vec[ymin], a_vec[ymax]))

        print('Came from this input: ')
        print('minX, maxX')
        print('x1, y1, x2, y2')
        print(((Y[xmin],X[xmin]), (Y[xmax], X[xmax])))
        print('minY, maxY')
        print('x3, y3, x4, y4')
        print(((Y[ymin],X[ymin]), (Y[ymax], X[ymax])))

        print('\n')
        # print(np.amin(a_vec) / (np.amax(a_vec)-np.amin(a_vec)))
        # print((np.amax(b_vec)-np.amin(b_vec)))

        # scale new positions to output
        # a_vec = ((a_vec - np.amin(a_vec)) / (np.amax(a_vec)-np.amin(a_vec))) * dst_height
        # b_vec = ((b_vec - np.amin(b_vec)) / (np.amax(b_vec)-np.amin(b_vec))) * dst_width

        a_mean = np.mean(a_vec)
        b_mean = np.mean(b_vec)
        a_vec = np.int64((a_vec - a_mean + 2 * dst_height ) /2 )# + 2000
        b_vec = np.int64((b_vec - b_mean + dst_width ) / 2)
        a_mean = np.mean(a_vec)
        b_mean = np.mean(b_vec)

        print(a_mean)
        print(b_mean)

        a_val, a_counts = np.unique(np.int64(a_vec), return_counts=True)
        b_val, b_counts = np.unique(np.int64(b_vec), return_counts=True)
        plt.bar(a_val, a_counts, label= 'height', color='blue')
        plt.bar(b_val, b_counts, label= 'width', color='red')
        plt.xlabel('Position in px')
        plt.ylabel('Amount')

        a_vec[a_vec>dst_height] = -1
        b_vec[b_vec>dst_width] = -1

        # a_vec /= 3
        # b_vec /= 3
        #

        # bool_arr = a_vec >= dst_height
        # # bool_arr = np.logical_and(bool_arr, b_vec > 0)
        # bool_arr = np.logical_or(bool_arr, b_vec >= dst_width)
        #
        # a_vec[bool_arr] = -1
        # b_vec[bool_arr] = -1

        # print(a_vec)
        # print(b_vec)
        # print(bool_arr)



        # quit()
        # b_vec = b_vec[bool_arr]
        # print(bool_arr.shape)
        # print(a_vec.shape)
        # print(b_vec.shape)
        # quit()
        # # print(np.amin(a_vec), np.amax(a_vec))
        # print(np.amin(b_vec), np.amax(b_vec))
        # print(dst_points.shape)
        # dst_points = np.zeros((np.int32((np.amax(a_vec)+1, np.amax(b_vec)+1, 3))), dtype=np.uint8)
        dst_points = np.zeros((dst_height+1, dst_width+1, 3), dtype=np.uint8)
        pos = np.stack((a_vec, b_vec), 2).astype(np.int64)
        # print(pos.shape)
        # print(dst_points.shape)
        # print(src.shape)
        # quit()
        # warping, assign colour from original image to new positions
        for i in np.arange(pos.shape[0]):
            for k in np.arange(pos.shape[1]):
                if pos[i,k,1] >= 0 and pos[i,k,0] >= 0:
                    # print((pos[i,k,0], pos[i,k,1]))
                    dst_points[pos[i,k,0], pos[i,k,1]] = src[i,k]




        # dst_points = maximum_filter(dst_points, (50,3,1))
        cv.imwrite('schaunwirmal10.png', dst_points)
        plt.show()
        quit()
        return dst_points
    else:
        print('hereagain')
        dst_height, dst_width = 2000, 2000
        print(H)
        H = np.linalg.inv(H)
        print(H)

        xstart = -dst_height/2
        # xstart = 0
        xend = xstart + dst_height

        ystart = -dst_width/2
        # ystart = 0
        yend = ystart + dst_width

        x_vec = np.arange(xstart, xend)
        y_vec = np.arange(ystart, yend)

        Y, X = np.meshgrid(y_vec, x_vec)
        # print(X.shape)
        a_vec = np.float32((H[0,0]*X + H[0,1]*Y + H[0,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))# + height/2)
        b_vec = np.float32((H[1,0]*X + H[1,1]*Y + H[1,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))# + width/2)

        a_mean = np.median(a_vec)
        b_mean = np.median(b_vec)

        print(a_mean)
        print(b_mean)
        print(np.unique(a_vec>dst_height, return_counts=True))
        print(np.unique(b_vec>dst_width, return_counts=True))
        # a_vec = a_vec + dst_height / 2
        # b_vec = b_vec + dst_width / 2
        a_vec = (a_vec - a_mean) + height / 2
        b_vec = (b_vec - b_mean) + width / 2
        print(np.unique(a_vec>dst_height, return_counts=True))
        print(np.unique(b_vec>dst_width, return_counts=True))

        # a_mean = np.median(a_vec)
        # b_mean = np.median(b_vec)
        #
        # print(a_mean)
        # print(b_mean)
        a_vec_int = a_vec.astype(np.int32)
        b_vec_int = b_vec.astype(np.int32)
        print(np.unique(b_vec_int, return_counts=True))
        dst_points = np.zeros((dst_height+1, dst_width+1, 3), dtype=np.uint8)
        a_vec[a_vec>src.shape[0]] = 0
        b_vec[b_vec>src.shape[1]] = 0
        a_vec[a_vec<0] = 0
        b_vec[b_vec<0] = 0
        pos = np.stack((a_vec, b_vec), 2).astype(np.int64)

        for i in np.arange(pos.shape[0]):
            for k in np.arange(pos.shape[1]):
                # if pos[i,k,1] >= 0 and pos[i,k,0] >= 0:
                    # print((pos[i,k,0], pos[i,k,1]))
                    # if i < src.shape[0] and k < src.shape[1]:
                dst_points[i,k] = src[pos[i,k,0], pos[i,k,1]]
                        # dst_points[pos[i,k,0], pos[i,k,1]] = src[i,k]
        # dst_points = cv.remap(src, b_vec, a_vec, interpolation=1)
        # print(dst_points.shape)

        cv.imwrite('schaunwirmal11.png', dst_points)
        print('here2')
        return dst_points
        # quit()


# from Gerhard Roth
def get_homography2(rot, K):
    # print(rot.shape)
    # print(K.shape)
    KR = np.dot(K, rot)
    # print(KR)
    KRK = np.dot(KR, np.linalg.inv(K))
    # print(KRK)
    # print('end of homography2')

    # return inverse of homography to achieve src --> dst warping
    return np.linalg.inv(KRK)


if __name__ == '__main__':
    # img = cv.imread('./00-08-52-932_points.png') # Read the test img
    # img = cv.imread('./3_00-01-48.png') # Read the test img
    # img, roll_angle= cv.imread('./problematic.png'), 0.298919415637517
    # img, roll_angle= cv.imread('./problematic2.png'), 0.171368206765908
    img, roll_angle= cv.imread('./problematic3.png'), -0.340036930698958


    marked_img, houg_im, retval = mark_lanes(img, -roll_angle)
    # print(retval)
    warped_img, angle, found,_,_,_ = birdview(marked_img, False, None)
    # cv.imwrite('aha.png', warped_img)
