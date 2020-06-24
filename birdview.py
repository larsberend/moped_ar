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

IMAGE_H = 562
IMAGE_W = 1920

font = cv.FONT_HERSHEY_SIMPLEX


def birdview(img, view, last_angle):
    focal_mat = np.array([[f,0,0],[0,f,0], [0,0,1]], dtype=np.float64)
    pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
    K = np.dot(pixel_mat, focal_mat)
    warped_img = None

    yellow = np.where(np.all(img == [0,204,204], axis=-1))
    blue = np.where(np.all(img == [204,204,0], axis=-1))
    # left = np.where(np.all(warped_img == [0,0,255], axis=-1))
    # print(yellow)
    # for x,y in zip(yellow[0], yellow[1]):
    #     img = cv.circle(img, (y,x), 10,  (0,0,255))
    # cv.imwrite('cirles.png', img)
    new_height = np.amin([min(yellow[0]),min(blue[0])])
    new_width = np.amax([max(yellow[1]),max(blue[1])])

    img = img[new_height:, 0:new_width+1]

    # yellow = np.where(np.all(img == [0,204,204], axis=-1))
    # blue = np.where(np.all(img == [204,204,0], axis=-1))

    if last_angle is not None:
        search_grid = np.arange(last_angle-0.1, last_angle+0.1, 0.001)
    else:
        search_grid = np.arange(0, np.pi, 0.01)

    cnt = 0
    # print(search_grid.shape)
    for p in search_grid:
        # print(p)
        rot = R.from_euler('xyz', (0, p, 0), degrees=False).as_matrix()
        H = get_homography2(rot, K)

        if view:
            warped_img = warp_img(img, H)
            yellow_warp = np.where(np.all(warped_img == [0,204,204], axis=-1))
            blue_warp = np.where(np.all(warped_img == [204,204,0], axis=-1))
            # pass x as y values and vice versa to avoid infinite slope

            ransac_y = linear_model.RANSACRegressor()

            ransac_y.fit(yellow_warp[0].reshape(len(yellow_warp[0]), 1), yellow_warp[1])
            line_ransac_y = ransac_y.predict(yellow_warp[0].reshape(len(yellow_warp[0]), 1))

            slope_y, intercept_y = linregress(yellow_warp[0], line_ransac_y)[:2]
            # print(slope_y)

            ransac_b = linear_model.RANSACRegressor()
            ransac_b.fit(blue_warp[0].reshape(len(blue_warp[0]), 1), blue_warp[1])
            line_ransac_b = ransac_b.predict(blue_warp[0].reshape(len(blue_warp[0]), 1))

            slope_b, intercept_b = linregress(blue_warp[0], line_ransac_b)[:2]
            # print(slope_b)

            warped_img = cv.line(warped_img, pt1=(np.int32(intercept_y), 0), pt2=(np.int32(slope_y*warped_img.shape[0] + intercept_y), warped_img.shape[0]), color=(0,255,0))
            warped_img = cv.line(warped_img, pt1=(np.int32(intercept_b), 0), pt2=(np.int32(slope_b*warped_img.shape[0] + intercept_b), warped_img.shape[0]), color=(255,0,0))

            cv.imwrite('./my_warp/%s-3.png'%(cnt), warped_img)




        else:
            # print(yellow)
            # print(blue)
            yellow_warp = point_warp(yellow, H, img)
            blue_warp = point_warp(blue, H, img)
            # print(yellow_warp)
            # print(blue_warp)

            # slope_l = linregress(left[0], left[1])[0]
            # pass x as y values and vice versa to avoid infinite slope
            # slope_y, intercept_y = linregress(yellow_warp[0], yellow_warp[1])[:2]
            # slope_b, intercept_b = linregress(blue_warp[0], blue_warp[1])[:2]
            ransac_y = linear_model.RANSACRegressor()

            ransac_y.fit(yellow_warp[0].reshape(len(yellow_warp[0]), 1), yellow_warp[1])
            line_ransac_y = ransac_y.predict(yellow_warp[0].reshape(len(yellow_warp[0]), 1))

            slope_y, intercept_y = linregress(yellow_warp[0], line_ransac_y)[:2]
            # print(slope_y)

            ransac_b = linear_model.RANSACRegressor()
            ransac_b.fit(blue_warp[0].reshape(len(blue_warp[0]), 1), blue_warp[1])
            line_ransac_b = ransac_b.predict(blue_warp[0].reshape(len(blue_warp[0]), 1))

            slope_b, intercept_b = linregress(blue_warp[0], line_ransac_b)[:2]
            # print(slope_b)
            # warped_img = cv.line(warped_img, pt1=(np.int32(intercept_y), 0), pt2=(np.int32(slope_y*warped_img.shape[0] + intercept_y), warped_img.shape[0]), color=(0,255,0))
            # cv.imwrite('./guckstehier.png', warped_img)
            # quit()

            # print(((np.int32(intercept_m), 0), (np.int32(slope_m*warped_img.shape[0] + intercept_m), warped_img.shape[0])))
            # print(((np.int32(intercept_r), 0), (np.int32(slope_r*warped_img.shape[0] + intercept_r), warped_img.shape[0])))

        # print((slope_y, slope_b))


        tol = 1e-02
        if np.isclose(slope_y, slope_b, tol, equal_nan=False):
            # print('ja nice!')
            # print((slope_y,slope_b))
            # warped_img = warp_img(img,H)
            #
            # # print(np.unique(warped_img, return_counts=True))
            # cv.imwrite('./parallel.png', warped_img)

            return warped_img, p

        cnt += 1

    print('No fitting angle found.')
    return warped_img, last_angle

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

# old version of homography, does not yield satisfying results.
def get_homography(rot, t, n):
    n = n.reshape((3,1))
    t = t.reshape((3,1))

    H = rot + np.dot(t, n.T)
    H = H / H[2,2]
    return H

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

            c1, c2 = kmeans2.cluster_centers_.astype(np.int32)
            cv.line(img, (c1[0], c1[1]), (c1[2], c1[3]), (120,120,0), 50)
            cv.line(img, (c2[0], c2[1]), (c2[2], c2[3]), (0,120,120), 50)

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

if __name__ == '__main__':
    # img = cv.imread('./00-08-52-932_points.png') # Read the test img
    # img = cv.imread('./3_00-01-48.png') # Read the test img
    # img, roll_angle= cv.imread('./problematic.png'), 0.298919415637517
    # img, roll_angle= cv.imread('./problematic2.png'), 0.171368206765908
    img, roll_angle= cv.imread('./problematic3.png'), -0.340036930698958

    # rotate image by specified roll angle


    marked_img, retval = mark_lanes(img, -roll_angle*2)
    # print(retval)
    # warped_img, angle = birdview(marked_img, False, None)
    # cv.imwrite('aha.png', warped_img)
