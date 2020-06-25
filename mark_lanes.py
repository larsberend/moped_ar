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


# from https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0
def mark_lanes(img, roll_angle):
    # img = rotate_image(img, roll_angle)
    roll_angle = np.degrees(roll_angle)
    img = skimage.transform.rotate(img, -roll_angle, clip=True, preserve_range=True).astype(np.uint8)
    # print(img.dtype)

    cv.imwrite('schaunwirmal.png', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel_size = 5
    gauss_gray = cv.GaussianBlur(gray, (kernel_size,kernel_size), 0)
    cv.imwrite('schaunwirmal1.png', gauss_gray)
    mask_white = cv.inRange(gauss_gray, 230, 255)
    mask_w_image = cv.bitwise_and(gauss_gray, mask_white)
    # cv.imwrite('schaunwirmal2.png', gauss_gray)
    # kernel_size = 5
    # gauss_gray = cv.GaussianBlur(mask_w_image, (kernel_size,kernel_size), 0)
    cv.imwrite('schaunwirmal2.png', mask_w_image)

    low_threshold = 50
    high_threshold = 150
    canny_edges = cv.Canny(mask_w_image,low_threshold,high_threshold)

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
            kmeans2 = KMeans(n_clusters = 2)
            # kmeans3 = KMeans(n_clusters = 3)



            klines2 = kmeans2.fit_predict(klines)
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

            klines = klines2

            # c1, c2 = kmeans2.cluster_centers_.astype(np.int32)
            # cv.line(img, (c1[0], c1[1]), (c1[2], c1[3]), (120,120,0), 50)
            # cv.line(img, (c2[0], c2[1]), (c2[2], c2[3]), (0,120,120), 50)

            line_img = np.zeros((roi_image.shape[0], roi_image.shape[1], 3), dtype=np.uint8)
            for l in range(lines.shape[0]):
                for x1,y1,x2,y2 in lines[l]:
                    if klines[l]==0:
                        cv.line(img, (x1, y1), (x2, y2), (255,255,0), 2)
                    # elif klines[l]==1:
                    #     cv.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    else:
                        cv.line(img, (x1, y1), (x2, y2), (0,255,255), 2)
            weighted_img = cv.addWeighted(img, 0.8, line_img, 1, 0)

            # cv.imwrite('schaunwirmal4.png', weighted_img)

            for v in vertices:
                for x,y in v:
                    cv.circle(weighted_img, (x,y), 5,  (255,0,0), -1)
            # weighted_img = skimage.transform.rotate(weighted_img, roll_angle/2, clip=True, preserve_range=True)
            # weighted_img = rotate_image(weighted_img, -roll_angle/2)
            cv.imwrite('schaunwirmal5.png', weighted_img)
            return weighted_img, True
    return img, False


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
