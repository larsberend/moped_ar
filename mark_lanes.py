import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import linregress
import skimage.transform
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning
import warnings

# warnings.filterwarnings('error', category=ConvergenceWarning, module='sklearn')
# actual focal length = equivalent focal length / crop factor
f = np.float64(0.019/5.6)

# sensor size / video resolution
pu = np.float64(0.0062/1920)
pv = np.float64(0.0045/1080)
# central coordinates of video
u0 = 960
v0 = 540
# inspired by https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0
def mark_lanes(img, roll_angle):

    # rotate image around roll angle accessed by gyro data
    # and specify polygon in which lane markings are expected
    roll_angle = np.degrees(roll_angle)
    img = skimage.transform.rotate(img, -roll_angle, clip=True, preserve_range=True).astype(np.uint8)
    imshape = img.shape
    # if np.radians(roll_angle) > 0.2:
    #     top_left = [1*imshape[1]/2, 2*imshape[0]/3]
    # else:
    top_left = [962, 692]
        # top_left = [imshape[1]/3, 2*imshape[0]/3]
    top_right = [3*imshape[1]/6, 2*imshape[0]/3]
    lower_left = [0, imshape[0]-100]

    # lower_left = [imshape[1]/9,imshape[0]]
    lower_right = [imshape[1],imshape[0]]

    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    # print(vertices)
    roi_image = region_of_interest(img, vertices)
    # print(img.dtype)
    cv.imwrite('schaunwirmal.png', roi_image)

    gray = cv.cvtColor(roi_image, cv.COLOR_BGR2GRAY)
    kernel_size = 11
    gauss_gray = cv.GaussianBlur(gray, (kernel_size,kernel_size), 0)
    cv.imwrite('schaunwirmal0.png', gauss_gray)

    # find canny edges in blurred, gray image
    low_threshold = 50
    high_threshold = 150
    aperture_size = 500
    canny_edges = cv.Canny(gauss_gray,low_threshold,high_threshold, aperture_size)
    cv.imwrite('schaunwirmal1.png', canny_edges)

    # only show edges in image and discard of edges found along sides of polygon
    mask_white = cv.inRange(canny_edges, 230, 255)
    mask_w_image = cv.bitwise_and(gauss_gray, mask_white)
    mask_w_image[mask_w_image<160] = 0
    cv.imwrite('schaunwirmal2.png', mask_w_image)

    # rho and theta are the distance and angular resolution of the grid in Hough space
    # same values as quiz
    rho = 2
    theta = np.pi/180
    # threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 150
    min_line_len = 50
    max_line_gap = 70

    lines = cv.HoughLinesP(mask_w_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is not None:
        # print(lines)
        klines = lines.reshape((lines.shape[0], lines.shape[-1]))
        # print(klines)
        # print(lines)
        if lines.shape[0] > 2:
        # print(lines.shape)
        # print(lines)
        # lines[1] = lines[2]
        # print(lines)
            slope_arr = np.zeros(len(lines))

            for l in range(len(lines)):
                for x1,y1,x2,y2 in lines[l]:
                    if x2-x1 != 0:
                        slope_arr[l] = ((y2-y1)/(x2-x1))

            # cluster all hough lines in three clusters by slope
            # try:
            kmeans2 = KMeans(n_clusters = 3)
            # except RuntimeWarning:
            #     kmeans2 = KMeans(n_clusters = 2)
            #     print('in except')
            #     pass
            # print('out ecxcept')

            klines = kmeans2.fit_predict(slope_arr.reshape(-1,1))
            if np.unique(klines).size > 2:
                hough_img = img.copy()
                # print(klines)
                # if klines
                # draw all lines with their respective color of cluster in image
                for l in range(len(lines)):
                    for x1,y1,x2,y2 in lines[l]:
                        if klines[l] == 0:
                            hough_img = cv.line(hough_img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=3)
                        elif klines[l] == 1:
                            hough_img = cv.line(hough_img, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,255), thickness=3)
                        else:
                            hough_img = cv.line(hough_img, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=3)

                # cv.imwrite('schaunwirmal4.png', hough_img)

                lines = lines.reshape((lines.shape[0],2,2))

                # print(slope_arr)
                # print(klines)
                # print(lines[klines==0])
                lines_y = lines[klines==0]
                lines_b = lines[klines==1]
                # print(lines_b.shape)
                # print(np.unique(klines).shape)
                # if np.unique(klines).size > 2:
                lines_o = lines[klines==2]
                # else:
                #     lines_o = [[], []]


                # print(lines)
                # print(lines_y)
                # print(lines_b)
                # print(lines[0])
                line_points_y = connect2(lines_y[0])
                for ends in lines_y[1:]:
                    line_points_y = np.concatenate((line_points_y, connect2(ends)))

                line_points_b = connect2(lines_b[0])
                for ends in lines_b[1:]:
                    line_points_b = np.concatenate((line_points_b, connect2(ends)))

                line_points_o = connect2(lines_o[0])
                for ends in lines_o[1:]:
                    line_points_o = np.concatenate((line_points_o, connect2(ends)))

                lines_list = [line_points_y, line_points_b, lines_o]
                sort_lines = sorted(lines_list, key=np.size)
                lines_points_y = sort_lines[0]
                lines_points_b = sort_lines[1]

                slope_y, intercept_y, ransac_y = my_ransac(line_points_y, True)
                slope_b, intercept_b, ransac_b = my_ransac(line_points_b, True)
                print("Initial slope of the two clusters of lines selected:")
                print((slope_y,slope_b))
                # print(ransac_y)
                img[ransac_b[1], ransac_b[0]] = [255,0,0]
                img[ransac_y[1], ransac_y[0]] = [0,255,255]
                cv.imwrite('schaunwirmal5.png', img)




                return img, hough_img, True
    # if not at least 3 lines are found in image, return False

    return img, np.zeros_like(img), False

# calculate ONE line from cluster using ransac and regression
# return either slope and intercept or corresponding points in image
def my_ransac(warped_p, return_points):
    ransac = linear_model.RANSACRegressor()

    # print(warped_p[:,0])
    # pass x as y values and vice versa to avoid infinite slope

    ransac.fit(warped_p[:,0].reshape(-1, 1), warped_p[:,1])

    ransacX = np.arange(min(warped_p[:,0]), max(warped_p[:,0]), dtype=np.int64)
    line_ransac = ransac.predict(ransacX.reshape(-1, 1)).astype(np.int64)

    # remove edge cases from data, where points lie out of image
    ransacX = ransacX[line_ransac < 1080]
    line_ransac = line_ransac[line_ransac < 1080]
    ransacX = ransacX[line_ransac >= 0]
    line_ransac = line_ransac[line_ransac >= 0]
    line_ransac = line_ransac[ransacX >= 0]
    ransacX = ransacX[ransacX >= 0]
    line_ransac = line_ransac[ransacX < 1920]
    ransacX = ransacX[ransacX < 1920]

    slope, intercept = linregress(ransacX, line_ransac)[:2]
    # print(slope)
    if return_points:
        return slope, intercept, (ransacX, line_ransac)
    else:
        return slope, intercept


''' deprecated
def my_ransac_two(warped_p, return_points):
    ransac = linear_model.RANSACRegressor()
    # print(warped_p)
    # pass x as y values and vice versa to avoid infinite slope
    # warped_p = warped_p[:,0]
    # X = warped_p[:,0::2]

    X = warped_p[:, 1]
    y = warped_p[:, 0]

    X = X.reshape((X.size,1))
    # y = warped_p[:,1::2]
    # y = y.flatten()
    # print(y)
    ransac.fit(X, y)

    line_X_y = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac_y = ransac.predict(line_X_y)
    line_X_y = line_X_y.flatten()
    line_X_y = line_X_y[line_y_ransac_y < 1920]
    line_y_ransac_y = line_y_ransac_y[line_y_ransac_y < 1920]
    line_X_y = line_X_y[line_y_ransac_y >= 0]
    line_y_ransac_y = line_y_ransac_y[line_y_ransac_y >= 0]
    # line_ransac = ransac.predict(warped_p[0].reshape(len(warped_p[0]), 1))
    inlier_mask = ransac.inlier_mask_
    if np.all(inlier_mask) == True:
        return None, None
    else:
        # print(inlier_mask)
        outlier_mask = np.logical_not(inlier_mask)

        # print(y[inlier_mask])
        # print(X[inlier_mask])

        # slope1, intercept1 = linregress(y[inlier_mask], X.flatten()[inlier_mask])[:2]
        slope1, intercept1 = linregress(X.flatten()[inlier_mask], y[inlier_mask])[:2]

        ransac.fit(X[outlier_mask], y[outlier_mask])

        # slope2, intercept2 = linregress(y[outlier_mask], X.flatten()[outlier_mask])[:2]
        # slope2, intercept2 = linregress(X.flatten()[outlier_mask], y[outlier_mask])[:2]
        # print(slope1, slope2)


        # Predict data of estimated models
        line_X_b = np.arange(X.min(), X.max())[:, np.newaxis]
        # print(line_X_b)
        # quit()
        line_y_ransac_b = ransac.predict(line_X_b)
        line_X_b = line_X_b.flatten()
        # print(line_y_ransac_b.shape)
        # print(line_X_b.shape)

        line_X_b = line_X_b[line_y_ransac_b < 1920]
        line_y_ransac_b = line_y_ransac_b[line_y_ransac_b < 1920]
        line_X_b = line_X_b[line_y_ransac_b >= 0]
        line_y_ransac_b = line_y_ransac_b[line_y_ransac_b >= 0]


        return (line_X_b.astype(np.int64), line_y_ransac_b.astype(np.int64)), (line_X_y.astype(np.int64), line_y_ransac_y.astype(np.int64))
'''
# get pixels in between two points
# from Paul Panzer at: https://stackoverflow.com/questions/47704008/fastest-way-to-get-all-the-points-between-two-x-y-coordinates-in-python
def connect2(ends):
    d0, d1 = np.diff(ends, axis=0)[0]
    if np.abs(d0) > np.abs(d1):
        return np.c_[np.arange(ends[0, 0], ends[1,0] + np.sign(d0), np.sign(d0), dtype=np.int32),
        np.arange(ends[0, 1] * np.abs(d0) + np.abs(d0)//2,
        ends[0, 1] * np.abs(d0) + np.abs(d0)//2 + (np.abs(d0)+1) * d1, d1, dtype=np.int32) // np.abs(d0)]
    else:
        return np.c_[np.arange(ends[0, 0] * np.abs(d1) + np.abs(d1)//2,
        ends[0, 0] * np.abs(d1) + np.abs(d1)//2 + (np.abs(d1)+1) * d0, d0, dtype=np.int32) // np.abs(d1),
        np.arange(ends[0, 1], ends[1,1] + np.sign(d1), np.sign(d1), dtype=np.int32)]


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    mask = cv.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


if __name__ == '__main__':
    # img, roll_angle= cv.imread('./problematic.png'), 0.298919415637517
    # img, roll_angle= cv.imread('./problematic2.png'), 0.171368206765908
    # img, roll_angle= cv.imread('./problematic3.png'), -0.340036930698958
    # img, roll_angle= cv.imread('./problematic4.png'), 0.008977732261967
    # img, roll_angle= cv.imread('./problematic5.png'), 0.006774174538544
    img, roll_angle= cv.imread('./problematic6.png'), 0.164914878690494





    # rotate image by specified roll angle


    marked_img, hough_im, retval = mark_lanes(img, -roll_angle)
    print(retval)
