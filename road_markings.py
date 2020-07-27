import numpy as np
import cv2 as cv
import skimage.transform
from sklearn.cluster import KMeans
from sklearn import linear_model
from scipy.stats import linregress

def get_markings(img, roll, color1, color2):

    img_r = skimage.transform.rotate(img, np.degrees(roll), clip=True, preserve_range=True).astype(np.uint8)
    cv.imwrite('00_rotated.png', img_r)
    imshape = img_r.shape

    top_left = [962, 692]
    top_right = [3*imshape[1]/6, 2*imshape[0]/3]
    lower_left = [0, imshape[0]-100]
    lower_right = [imshape[1],imshape[0]]

    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    # print(vertices)
    roi_image = region_of_interest(img_r, vertices)
    # print(img.dtype)
    cv.imwrite('01_roi_image.png', roi_image)

    gray = cv.cvtColor(roi_image, cv.COLOR_BGR2GRAY)
    kernel_size = 11
    gauss_gray = cv.GaussianBlur(gray, (kernel_size,kernel_size), 0)
    cv.imwrite('02_gaussian.png', gauss_gray)

    # find canny edges in blurred, gray image
    low_threshold = 50
    high_threshold = 150
    aperture_size = 500
    canny_edges = cv.Canny(gauss_gray, low_threshold, high_threshold, aperture_size)
    cv.imwrite('03_canny.png', canny_edges)

    # only show edges in image and discard of edges found along sides of polygon
    mask_white = cv.inRange(canny_edges, 230, 255)
    mask_w_image = cv.bitwise_and(gauss_gray, mask_white)
    mask_w_image[mask_w_image<160] = 0
    cv.imwrite('04_masked.png', mask_w_image)

    # rho and theta are the distance and angular resolution of the grid in Hough space
    # same values as quiz
    rho = 2
    theta = np.pi/180
    # threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 150
    min_line_len = 50
    max_line_gap = 70
    lines = cv.HoughLinesP(mask_w_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    if lines is None or lines.shape[0] < 3:
        return img, np.zeros(img.shape), False

    line_points, hough_img = my_kmeans(lines, img_r)

    slope_y, intercept_y, ransac_y = my_ransac(line_points[0], True)
    slope_b, intercept_b, ransac_b = my_ransac(line_points[1], True)
    print("Initial slope of the two clusters of lines selected:")
    print((slope_y,slope_b))
    # print(ransac_y)
    img_r[ransac_b[1], ransac_b[0]] = color1
    img_r[ransac_y[1], ransac_y[0]] = color2
    cv.imwrite('06_ransac.png', img_r)

    return img_r, hough_img, True


def my_kmeans(lines, img_r):
    klines = lines.reshape((lines.shape[0], lines.shape[-1]))
    slope_arr = np.zeros(len(lines))

    for l in range(len(lines)):
        for x1,y1,x2,y2 in lines[l]:
            if x2-x1 != 0:
                slope_arr[l] = ((y2-y1)/(x2-x1))

    # cluster all hough lines in three clusters by slope
    kmeans2 = KMeans(n_clusters = 3)
    klines = kmeans2.fit_predict(slope_arr.reshape(-1,1))

    hough_img = img_r.copy()
    # draw all lines with their respective color of cluster in image
    for l in range(len(lines)):
        for x1,y1,x2,y2 in lines[l]:
            if klines[l] == 0:
                hough_img = cv.line(hough_img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=3)
            elif klines[l] == 1:
                hough_img = cv.line(hough_img, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,255), thickness=3)
            else:
                hough_img = cv.line(hough_img, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=3)
    cv.imwrite('05_hough_lines.png', hough_img)

    lines = lines.reshape((lines.shape[0],2,2))
    lines_y = lines[klines==0]
    lines_b = lines[klines==1]
    lines_o = lines[klines==2]
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

    # if np.unique(klines).size > 2:
    return (line_points_y, line_points_b), hough_img


# calculate ONE line from cluster using ransac and regression
# return either slope and intercept or corresponding points in image
def my_ransac(points, return_points=True):
    ransac = linear_model.RANSACRegressor()

    # print(points[:,0])
    # pass x as y values and vice versa to avoid infinite slope

    ransac.fit(points[:,0].reshape(-1, 1), points[:,1])

    ransacX = np.arange(np.amin(points[:,0]), np.amax(points[:,0]), dtype=np.int64)
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
