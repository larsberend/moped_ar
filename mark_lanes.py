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
# from birdview import my_ransac

# from https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0
def mark_lanes(img, roll_angle):
    # img = rotate_image(img, roll_angle)
    roll_angle = np.degrees(roll_angle)
    img = skimage.transform.rotate(img, -roll_angle, clip=True, preserve_range=True).astype(np.uint8)
    # print(img.dtype)

    cv.imwrite('schaunwirmal.png', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel_size = 11
    gauss_gray = cv.GaussianBlur(gray, (kernel_size,kernel_size), 0)
    # cv.imwrite('schaunwirmal1.png', gauss_gray)
    mask_white = cv.inRange(gauss_gray, 230, 255)
    mask_w_image = cv.bitwise_and(gauss_gray, mask_white)
    cv.imwrite('schaunwirmal2.png', gauss_gray)
    # kernel_size = 5
    # gauss_gray = cv.GaussianBlur(mask_w_image, (kernel_size,kernel_size), 0)

    # cv.imwrite('schaunwirmal2.png', mask_w_image)

    low_threshold = 50
    high_threshold = 150
    aperture_size = 1000
    canny_edges = cv.Canny(mask_w_image,low_threshold,high_threshold, aperture_size)

    imshape = img.shape
    lower_left = [0,imshape[0]]
    # lower_left = [imshape[1]/9,imshape[0]]
    lower_right = [imshape[1],imshape[0]]

    if roll_angle < 0:
        top_left = [imshape[1]/4, 2*imshape[0]/3]
        top_right = [5*imshape[1]/6, 2*imshape[0]/3]
    else:
        top_left = [imshape[1]/3, 2*imshape[0]/3]
        top_right = [3*imshape[1]/6, 2*imshape[0]/3]

    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    # print(vertices)
    roi_image = region_of_interest(canny_edges, vertices)
    # print(np.array_equal(roi_image, canny_edges))

    cv.imwrite('schaunwirmal3.png', roi_image)

    #rho and theta are the distance and angular resolution of the grid in Hough space
    #same values as quiz
    rho = 2
    theta = np.pi/180
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 150
    min_line_len = 50
    max_line_gap = 70

    lines = cv.HoughLinesP(roi_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is not None:

        # print(lines)
        klines = lines.reshape((lines.shape[0], lines.shape[-1]))
        # print(klines)
        # print(lines)

        if lines.shape[0] > 1:
            # kmeans2 = KMeans(n_clusters = 2)
            # kmeans3 = KMeans(n_clusters = 3)



            # klines2 = kmeans2.fit_predict(klines)


            # klines3 = kmeans3.fit_predict(klines)
            # print(kmeans2.inertia_)
            # print(lines)

            # klines3 = kmeans3.fit_predict(lines)
            # print(kmeans2.get_params())
            #
            # if kmeans3.inertia_ < kmeans2.inertia_:
            #     klines = klines3
            #     print('three lines')
            # else:
            #     klines = klines2
            #     print('two lines')
            #
            # print(lines)
            # print(klines)
            # quit()

            # klines = klines2

            # c1, c2 = kmeans2.cluster_centers_.astype(np.int32)
            # cv.line(img, (c1[0], c1[1]), (c1[2], c1[3]), (120,120,0), 50)
            # cv.line(img, (c2[0], c2[1]), (c2[2], c2[3]), (0,120,120), 50)

            # print(lines)
            # lines = np.roll(lines,1,2)
            # quit()
            # print(lines)
            slope_arr = np.zeros(len(lines))
            for l in range(len(lines)):
                for x1,y1,x2,y2 in lines[l]:
                    slope_arr[l] = ((y2-y1)/(x2-x1))
            kmeans2 = KMeans(n_clusters = 2)
            klines = kmeans2.fit_predict(slope_arr.reshape(-1,1))
            hough_img = img.copy()

            for l in range(len(lines)):
                for x1,y1,x2,y2 in lines[l]:
                    if klines[l] == 0:
                        hough_img = cv.line(hough_img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=3)
                    else:
                        hough_img = cv.line(hough_img, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,255), thickness=3)
            lines = lines.reshape((lines.shape[0],2,2))

            cv.imwrite('schaunwirmal4.png', hough_img)
            # print(slope_arr)
            # print(klines)
            # print(lines[klines==0])
            lines_y = lines[klines==0]
            lines_b = lines[klines==1]
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


            slope_y, intercept_y, ransac_y = my_ransac(line_points_y, True)
            slope_b, intercept_b, ransac_b = my_ransac(line_points_b, True)
            print((slope_y,slope_b))
            img[ransac_b[1], ransac_b[0]] = [255,0,0]
            img[ransac_y[1], ransac_y[0]] = [0,255,255]
            cv.imwrite('schaunwirmal5.png', img)
            return img, True
            # quit()
            # print(line_points)
            # img[line_points[:,1], line_points[:,0]] = [255,0,0]
            # for l in range(len(lines)):
            #     for x1,y1,x2,y2 in lines[l]:
            #         img = cv.line(img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0))


            # print(line_points[line_points>1080])
            '''
            line_b, line_y = my_ransac_two(line_points, False)
            if line_b is None:
                return img, False
            else:
                # print(line_b[line_b[0]>1920])
                # quit()
                # slope1, slope2, intercept1, intercept2 = my_ransac_two(line_points, False)

                # img = cv.line(img, pt1=(np.int32(intercept1), 0), pt2=(np.int32(slope1*img.shape[0] + intercept1), img.shape[0]), color=(0,255,0))
                # img = cv.line(img, pt1=(np.int32(intercept2), 0), pt2=(np.int32(slope2*img.shape[0] + intercept2), img.shape[0]), color=(0,0,255))
                # print(line_b[1])
                img[line_b[0], line_b[1]] = [255,0,0]
                img[line_y[0], line_y[1]] = [0,255,255]

                # print(img.shape)
                # line_img = np.zeros((roi_image.shape[0], roi_image.shape[1], 3), dtype=np.uint8)
                # for l in range(lines.shape[0]):
                #     for x1,y1,x2,y2 in lines[l]:
                #         slope = linregress((y1,x1), (y2,x2))[0]
                #         if slope < 0:
                #         # if klines[l]==0:
                #             cv.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
                #         # elif klines[l]==1:
                #         #     cv.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
                #         else:
                #             cv.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
                weighted_img = img
                # weighted_img = cv.addWeighted(img, 0.8, line_img, 1, 0)

                # cv.imwrite('schaunwirmal4.png', weighted_img)

                # for v in vertices:
                #     for x,y in v:
                #         cv.circle(weighted_img, (x,y), 5,  (255,0,0), -1)
                # weighted_img = skimage.transform.rotate(weighted_img, roll_angle/2, clip=True, preserve_range=True)
                # weighted_img = rotate_image(weighted_img, -roll_angle/2)
                cv.imwrite('schaunwirmal5.png', weighted_img)
                return weighted_img, True
                '''
    return img, False


def my_ransac(warped_p, return_points):
    ransac = linear_model.RANSACRegressor()

    # print(warped_p[:,0])
    # pass x as y values and vice versa to avoid infinite slope

    ransac.fit(warped_p[:,0].reshape(-1, 1), warped_p[:,1])

    ransacX = np.arange(min(warped_p[:,0]), max(warped_p[:,0]))
    line_ransac = ransac.predict(ransacX.reshape(-1, 1))

    ransacX = ransacX[line_ransac < 1920]
    line_ransac = line_ransac[line_ransac < 1920]
    ransacX = ransacX[line_ransac >= 0]
    line_ransac = line_ransac[line_ransac >= 0]

    slope, intercept = linregress(ransacX, line_ransac)[:2]
    # print(slope)
    if return_points:
        return slope, intercept, (ransacX.astype(np.int64), line_ransac.astype(np.int64))
    else:
        return slope, intercept



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

    # Compare estimated coefficients
    # print("Estimated coefficients (true, linear regression, RANSAC):")
    # print(ransac.estimator_.coef_)
    #
    # lw = 2
    # plt.scatter(y[inlier_mask], X[inlier_mask], color='yellowgreen', marker='.',
    #             label='Inliers')
    # plt.scatter(y[outlier_mask], X[outlier_mask], color='gold', marker='.',
    #             label='Outliers')
    # plt.plot(line_y_ransac_y, line_X_y, color='yellow', linewidth=lw,
    #          label='RANSAC regressor')
    # plt.plot(line_y_ransac_b, line_X_b, color='cornflowerblue', linewidth=lw,
    #          label='RANSAC regressor')
    # plt.ylim(1080,0, 50)
    # plt.xlim(0, 1920, 50)
    # plt.legend(loc='lower right')
    # plt.xlabel("Input")
    # plt.ylabel("Response")
    # plt.show()
    # quit()
    # if return_points:
    #     return slope1, slope2, intercept1, intercept2, (warped_p[0], line_ransac)
    # else:
    #     return slope1, slope2, intercept1, intercept2

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

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

if __name__ == '__main__':
    img, roll_angle= cv.imread('./problematic.png'), 0.298919415637517
    # img, roll_angle= cv.imread('./problematic2.png'), 0.171368206765908
    # img, roll_angle= cv.imread('./problematic3.png'), -0.340036930698958

    # rotate image by specified roll angle


    marked_img, retval = mark_lanes(img, -roll_angle)
    # print(retval)
