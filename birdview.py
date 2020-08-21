import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import skimage.transform
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn import linear_model
from mark_lanes import mark_lanes, pu, pv, u0, v0, f, region_of_interest
import skimage.transform
from scipy.ndimage import maximum_filter



font = cv.FONT_HERSHEY_SIMPLEX
focal_mat = np.array([[f,0,0],[0,f,0], [0,0,1]], dtype=np.float64)
pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
K = np.dot(pixel_mat, focal_mat)


def birdview(img, view, last_angle):

    # # find points on lane markings in image
    # cut image to location of markings
    orig_img = img.copy()
    cut_to_road = orig_img[int(540):]
    img = cut_to_road

    yellow = np.where(np.all(img == [0,255,255], axis=-1))
    blue = np.where(np.all(img == [255,0,0], axis=-1))

    warped_img, angle, found, slope, inter1, inter2, yellow_warp, blue_warp, cam_origin = iter_angle(last_angle, cut_to_road, view, yellow, blue)

    # check if an angle was found
    if found:
        return warped_img, angle, True, slope, inter1, inter2, yellow_warp, blue_warp, cam_origin

    # else search for it again with best guess
    else:
        warped_img, angle, found, slope, inter1, inter2, yellow_warp, blue_warp, cam_origin = iter_angle(angle, cut_to_road, view, yellow, blue)

        if found:
            return warped_img, angle, True, slope, inter1, inter2, yellow_warp, blue_warp, cam_origin
            # return big_warp, angle, True, slope, inter1, inter2
            # return warped_img, angle, True

        else:
            print('No fitting angle found')
            return warped_img, angle, False, slope, inter1, inter2, yellow_warp, blue_warp, cam_origin


# calculate birdviews around a certain angle or in range [30, 43] if input last_angle==None
def iter_angle(last_angle, img, view, yellow, blue):
    warped_img = np.zeros((img.shape[1], img.shape[0], 3))
    cnt = 0
    angle_guess = 0
    smallest_diff = 100
    slope_y, slope_b, intercept_y, intercept_b = None, None, None, None

    if last_angle is None:
        search_grid = np.arange(np.radians(30), np.radians(43), np.radians(0.1))
    else:
        search_grid = np.arange(last_angle-0.1, last_angle+0.1, 0.00001)


    for angle in search_grid:
        # angle = 0
        # angle = 0.6626151157573931
        # angle = 0.7
        # angle = 0.758468224617082
        rot = R.from_euler('xyz', (0, angle, 0), degrees=False).as_matrix()
        H = get_homography2(rot, K)
        # view==True save images with 30,31,32,...,42 degree pitch angle
        if view:
            warping = warp_img(img, H, angle, True, yellow, blue)
            cv.imwrite('./my_warp/%s.png'%(cnt), warping[0])
            # cv.imwrite('./my_warp_40/%s.png'%(cnt), warped_img)
            cnt += 1

        else:
            yellow_minus = np.float64((yellow[0] - img.shape[0]/2, yellow[1] - img.shape[1]/2))
            blue_minus = np.float64((blue[0] - img.shape[0]/2, blue[1] - img.shape[1]/2))

            yellow_warp = point_warp(yellow_minus, H, img)
            blue_warp = point_warp(blue_minus, H, img)

            slope_y, intercept_y = linregress(yellow_warp)[:2]
            slope_b, intercept_b = linregress(blue_warp)[:2]

            # get best guess for angle
            diff = np.abs(slope_b-slope_y)
            if diff < smallest_diff:
                smallest_diff = diff
                angle_guess = angle
                # print((angle_guess,smallest_diff))

        if smallest_diff < 0.001:
            # make birdview image and save
            print('Angle, smallest_diff:')
            print((angle_guess,smallest_diff))
            warped_img, yellow_warp, blue_warp, cam_origin = warp_img(img, H, angle_guess, True, yellow, blue)
            slope_y, intercept_y = linregress(yellow_warp)[:2]
            slope_b, intercept_b = linregress(blue_warp)[:2]
            # cv.imwrite('schaunwirmal6.png', warped_img)

            # rotate image back (see mark_lanes.py)
            rotate_angle = np.degrees(0 - np.arctan(slope_b))
            rotated_img = skimage.transform.rotate(warped_img, rotate_angle, clip=True, preserve_range=True)
            # slope_b = 0
            # cv.imwrite('bv_rotate_test.png', rotate_img)

            cv.putText(warped_img, 'Pitch Angle: %s'%(np.degrees(angle)), (10, 1650), font, 0.5, (255, 255, 0), 2, cv.LINE_AA)
            cv.putText(warped_img, 'Smallest diff: %s'%(smallest_diff), (10, 1670), font, 0.5, (255, 255, 0), 2, cv.LINE_AA)
            return rotated_img, angle_guess, True, slope_b, intercept_y, intercept_b, yellow_warp, blue_warp, cam_origin

    # make birdview image and save
    print('Angle, smallest_diff:')
    print((angle_guess,smallest_diff))
    warped_img, yellow_warp, blue_warp, cam_origin = warp_img(img, H, angle_guess, True, yellow, blue)
    slope_y, intercept_y = linregress(yellow_warp)[:2]
    slope_b, intercept_b = linregress(blue_warp)[:2]
    # cv.imwrite('schaunwirmal6.png', warped_img)

    # rotate image back (see mark_lanes.py)
    rotate_angle = np.degrees(0 - np.arctan(slope_b))
    rotated_img = skimage.transform.rotate(warped_img, rotate_angle, clip=True, preserve_range=True)
    # slope_b = 0
    # cv.imwrite('bv_rotate_test.png', rotate_img)

    cv.putText(warped_img, 'Pitch Angle: %s'%(np.degrees(angle)), (10, 1650), font, 0.5, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(warped_img, 'Smallest diff: %s'%(smallest_diff), (10, 1670), font, 0.5, (255, 255, 0), 2, cv.LINE_AA)
    return rotated_img, angle_guess, True, slope_b, intercept_y, intercept_b, yellow_warp, blue_warp, cam_origin


# warp points with specified Homography
def point_warp(points, H, img):
    X, Y = np.float64(points)
    a_vec = np.float64((H[0,0]*X + H[0,1]*Y + H[0,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))
    b_vec = np.float64((H[1,0]*X + H[1,1]*Y + H[1,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]))

    return a_vec, b_vec


# function to visualize & verify
# warp all pixels of image-exerpt in birdview
# also warp lines for road markings and camera origin and color in birdview
def warp_img(src, H, angle=None, inv=True, yellow=None, blue=None, dst_height=4000, dst_width=4000):
    height, width = src.shape[:2]
    if inv == True:
        # move rotation centre of image, such that output image is more centered
        xstart = -height/2
        xend = xstart + height
        ystart = -width/2
        yend = ystart + width
        x_vec = np.arange(xstart, xend)
        y_vec = np.arange(ystart, yend)
        Y, X = np.meshgrid(y_vec, x_vec)

        yellow = np.int64((yellow[0] + xstart, yellow[1] + ystart))
        blue = np.int64((blue[0] + xstart, blue[1] + ystart))

        cam_origin = ([height + xstart], [width/2 + ystart])


        # warp points of image, lines and camera origin

        a_vec, b_vec = point_warp((X,Y), H, src)

        yellow = point_warp(yellow, H, src)
        blue = point_warp(blue, H, src)

        cam_origin = point_warp(cam_origin, H, src)

        # plot lines in matplotlib if needed
        # linear regression for warped points on road markings
        # slope_y, intercept_y = linregress(yellow)[:2]
        # slope_b, intercept_b = linregress(blue)[:2]

        # abline(slope_y, intercept_y)
        # abline(slope_b, intercept_b)

        # scale all points to get somewhat common image dimensions
        a_vec = np.int64((a_vec / 30) + dst_height)
        b_vec = np.int64((b_vec / 30) + dst_width / 2)

        yellow_a = np.int64((yellow[0] / 30) + dst_height)
        yellow_b = np.int64((yellow[1] / 30) + dst_width / 2)
        blue_a = np.int64((blue[0] / 30) + dst_height)
        blue_b = np.int64((blue[1] / 30) + dst_width / 2)

        cam_origin_a = np.int64((cam_origin[0] / 30) + dst_height)
        cam_origin_b = np.int64((cam_origin[1] / 30) + dst_width / 2)

        a_vec[a_vec>dst_height] = -1
        b_vec[b_vec>dst_width] = -1

        # warping, assign colour from original image to new positions
        dst_points = np.zeros((dst_height+1, dst_width+1, 3), dtype=np.uint8)
        pos = np.stack((a_vec, b_vec), 2).astype(np.int64)
        for i in np.arange(pos.shape[0]):
            for k in np.arange(pos.shape[1]):
                if pos[i,k,0] >= 0 and pos[i,k,1] >= 0:
                    # print((pos[i,k,0], pos[i,k,1]))
                    dst_points[pos[i,k,0], pos[i,k,1]] = src[i,k]

        # interpolation to enhance visibility
        dst_points = my_interpol2(dst_points)

        yellow_a[yellow_a>dst_height] = 0
        yellow_b[yellow_b>dst_width] = 0
        blue_a[blue_a>dst_height] = 0
        blue_b[blue_b>dst_width] = 0

        yellow_a[yellow_a<0] = 0
        yellow_b[yellow_b<0] = 0
        blue_a[blue_a<0] = 0
        blue_b[blue_b<0] = 0

        # print(np.unique(yellow_a, return_counts=True))
        # print(np.unique(yellow_b, return_counts=True))

        # drwa lines and camera origin in birdview
        dst_points[yellow_a, yellow_b] = [0,255,255]
        dst_points[blue_a, blue_b] = [255,0,0]

        cv.circle(dst_points, (cam_origin_b, cam_origin_a), 5, (0,0,255), thickness=-1)
        return dst_points, (yellow_a, yellow_b), (blue_a, blue_b), (cam_origin_a, cam_origin_b)
    else:
        pass

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')



# from Gerhard Roth
def get_homography2(rot, K):
    KR = np.dot(K, rot)
    KRK = np.dot(KR, np.linalg.inv(K))
    # print(rot.shape)
    # print(K.shape)
    # print(KR)
    # print(KRK)
    # print('end of homography2')

    # return inverse of homography to achieve src --> dst warping
    return np.linalg.inv(KRK)

# interpolation to enhance viibility in birdview
def my_interpol2(img):
    zero = []
    for a in range(img.shape[0]):
        if np.all(img[a]==[0,0,0]):
            zero.append(a)


    interpolated = img.copy()
    zero = np.array(zero)

    # iterate through all black rows in image
    for i in np.arange(zero.size):
        # print((i, zero.size))
        dist1, dist2, nonzero1, nonzero2 = None,None,None,None

        # go from black row down until one pixel in row is in color
        # save it and distance
        j = zero[i]
        while(j<img.shape[1]):
            if np.any(img[j] != [0,0,0]):
                nonzero1 = img[j]
                dist1 = np.abs(j-zero[i])
                break
            else:
                j = j+1
        # print('out of while')

        # same, just go up
        j = zero[i]
        while(j>=0):
            if np.any(img[j] != [0,0,0]):
                nonzero2 = img[j]
                dist2 = np.abs(j-zero[i])
                break
            else:
                j = j-1

        # give row fractions of pixel values of colore rows above and below
        # according to distance
        if dist1 and dist2:
            interval = dist1 + dist2
            dist1_norm = dist1/interval
            dist2_norm = dist2/interval

            bgr = nonzero1 * dist1_norm + nonzero2 * dist2_norm

            interpolated[zero[i]] = np.int32(bgr)
    # cv.imwrite('interpol.png',interpolated)
    return interpolated


if __name__ == '__main__':
    # img = cv.imread('./bv_stripes2.png')
    # bv = my_interpol2(img)
    # cv.imwrite('./bv_no_stripes.png')
    # quit()


    # img = cv.imread('./00-08-52-932_points.png') # Read the test img
    # img = cv.imread('./3_00-01-48.png') # Read the test img
    # img, roll_angle= cv.imread('./problematic.png'), 0.298919415637517
    # img, roll_angle= cv.imread('./problematic2.png'), 0.171368206765908
    img, roll_angle= cv.imread('./problematic3.png'), -0.340036930698958
    # img, roll_angle= cv.imread('./problematic0.png'), -0.234190505239813

    marked_img, houg_im, retval = mark_lanes(img, -roll_angle)

    # print(retval)
    # marked_img, roll_angle= cv.imread('./calib_yb_rot.png'), 0
    warped_img, angle, found,_,_,_,_,_,_ = birdview(marked_img, False, None)
    print(angle)
    print(np.degrees(angle))
    cv.imwrite('aha.png', warped_img)
