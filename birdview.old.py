import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.cluster.vq import kmeans,vq,whiten
from scipy.stats import linregress
from draw_curve import pu,pv,u0,v0,f
from skimage.draw import line

IMAGE_H = 562
IMAGE_W = 1920

font = cv.FONT_HERSHEY_SIMPLEX
# matrices for camera intrinsics
pixel_mat = np.array([[1/pu,0,u0], [0,1/pv,v0],[0,0,1]], dtype=np.float64)
focal_mat = np.array([[f,0,0],[0,f,0], [0,0,1]], dtype=np.float64)

K = np.dot(pixel_mat, focal_mat)

def birdview():
    img = cv.imread('./00-08-52-932_points_cut.png') # Read the test img

    # left = np.where(np.all(img == [0,0,255], axis=-1))
    # middle = np.where(np.all(img == [0,255,0], axis=-1))
    # right = np.where(np.all(img == [255,0,0], axis=-1))


    # left = np.column_stack((left[0], left[1]))#.astype(np.float64)
    # middle = np.column_stack((middle[0], middle[1]))#.astype(np.float64)
    # right = np.column_stack((right[0], right[1]))#.astype(np.float64)

    # inter_l = linregress(left[1], left[0])
    # inter_m = linregress(middle[1], middle[0])
    # inter_r = linregress(right[1], right[0])
    #
    # test_reg = linregress([0,1], [1,2])
    # print(inter_l)
    # print(test_reg)

    # tol = 1e-08
    # slope_l, slope_m, slope_r = 1,2,3
    # while not(np.isclose(slope_l, slope_m, tol, equal_nan=False)) or not(np.isclose(slope_l, slope_r, tol, equal_nan=False)):
    #     # print((slope_l,slope_m,slope_r))
    #     # slope_l, slope_m, slope_r = 1,1,1
    #     # print((slope_l,slope_m,slope_r))
    #
    #     l_cand, m_cand, r_cand  = get_warp_pos(left, middle, right)#(irgenwie bekomm ich hier kandidaten fuer punkte, die die birdview linien zeigen)
    #     slope_l = linregress(l_cand[0], l_cand[1])[0]
    #     slope_m = linregress(m_cand[0], m_cand[1])[0]
    #     slope_r = linregress(r_cand[0], r_cand[1])[0]
    # print('out of while')

    # src = np.float32([[889, 577, 1], [515, 902, 1], [1018, 570, 1], [1566, 1079, 1]])
    # dst = np.float32([[889, 577, 1], [889, 902, 1], [1018, 570, 1], [1018, 1079, 1]])

    # dist_far = np.linalg.norm(dst[0]-dst[2])
    # dist_near = np.linalg.norm(dst[1]-dst[3])
    # print(dist_far)
    # print(dist_near)
    # print(abs(dist_far-dist_near))
    # print([dst[0,:2],dst[1,:2]])
    # slope_m = linregress([dst[0,:2],dst[1,:2]])[0]
    # slope_s = linregress([dst[2,:2],dst[3,:2]])[0]
    # slope_m = np.polyfit([889, 889], [577, 902], 1)[0]
    # slope_s = np.polyfit([1018, 1018],[570, 1079], 1)[0]
    # print(slope_m)
    # print(slope_s)
    # quit()
    # src = np.float32([[889, 577, 1], [515, 902, 1], [1018, 570, 1], [1566, 1079, 1]])
    # dst = np.float32([[515, 577, 1], [515, 902, 1], [1566, 570, 1], [1566, 1079, 1]])
    # left = (np.array([0, 10,20,30]), np.array([0, 10,20,30]))
    # left = np.column_stack((left[0], left[1]))
    # l_cand = np.column_stack((l_cand[0], l_cand[1]))
    # print(left)
    # print(l_cand)
    # homo, _ = cv.findHomography(src, dst)
    # homo, _ = cv.findHomography([left,middle,right], [l_cand, m_cand, r_cand])

    # K  = np.identity(3)
    # retval, rotations, translations, normals = cv.decomposeHomographyMat(homo, K)

    # H = get_homography(rotations[0], translations[0], normals[0])
    # print((rotations[0], translations[0], normals[0]))
    # print('hello rotation')
    # print(R.from_matrix(rotations[0]).as_euler('xyz'))
    # print(normals[0])
    # print(H)
    # print(homo)

    # print(dst)
    # print(dst2)
    # print(dst)
    # dst2 = [x for x in src]
    # print(dst2.shape)
    # print(dst2[:,2].shape)
    # M = cv.getPerspectiveTransform(src[:,:2].astype(np.float32), dst[:,:2].astype(np.float32)) # The transformation matrix
    # print(M)
    # quit()
    # for x in np.arange(0, np.pi, 0.000001):
    cnt = 0
    for p in np.arange(0.000, np.pi, 0.1):
        print(p)
        H = get_homography(rot = R.from_euler('xyz', (0, p, 0), degrees=False).as_matrix(),
                       t = np.array([0, 0, 0]),
                       n = np.array([0, 1, 0])
                       )
        # print(H)
        # large_im = np.zeros((10000, 10000, 3))
        # large_im[4000:5080, 4000:5920] = img
        # warped_img = cv.warpPerspective(large_im, H, (10000, 10000)) # Image warping
        small_img = cv.resize(img, (np.int32(img.shape[1]/2), np.int32(img.shape[0]/2)))
        # warped_img = my_warp_vec(small_img, H)
        warped_img = my_warp3(small_img, H)
        # warped_img2 = my_warp2(small_img, H)
        # warped_img = cv.rotate(warped_img, cv.ROTATE_90_COUNTERCLOCKWISE)
        # warped_img = cv.warpPerspective(img, H, (1920, 1080), (cv.INTER_LINEAR, cv.WARP_INVERSE_MAP)) # Image warping
        # warped_img = cv.resize(warped_img, (1920, 1080))
        # cv.imshow('warp', warped_img)
        # cv.putText(warped_img, 'x = %s'%(x), (50,20), font, 0.5, (0, 255, 0), 20, cv.LINE_AA)
        cv.imwrite('./my_warp/%s-3.png'%(cnt), warped_img)
        # cv.imwrite('./my_warp/%s-2.png'%(cnt), warped_img2)
        '''
        # left = np.where(np.all(warped_img == [0,0,255], axis=-1))
        middle = np.where(np.all(warped_img == [0,255,0], axis=-1))
        right = np.where(np.all(warped_img == [255,0,0], axis=-1))
        # slope_l = linregress(left[0], left[1])[0]
        slope_m = linregress(middle[0], middle[1])[0]
        slope_r = linregress(right[0], right[1])[0]

        tol = 1e-02
        if np.isclose(slope_r, slope_m, tol, equal_nan=False):
            print('ja nice!')
            print((slope_r,slope_m))
            print(x)
            cv.imwrite('./parallel.png'%(cnt), warped_img)

            quit()
            # slope_l, slope_m, slope_r = 1,1,1
            # print((slope_l,slope_m,slope_r))

        print((slope_r, slope_m))
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

    quit()
'''
def my_warp(src, H):
    height, width = src.shape[:2]
    dst_points = np.zeros_like(src)
    # print(H)
    # print((H[0,2], H[1,2], H[2,2]))
    # quit()
    for x in np.arange(height):
        for y in np.arange(width):
            a = (H[0,0]*x + H[0,1]*y + H[0,2])/(H[2,0]*x + H[2,1]*y + H[2,2])
            b = (H[1,0]*x + H[1,1]*y + H[1,2])/(H[2,0]*x + H[2,1]*y + H[2,2])
            if a < height and a > 0 and b < width and b > 0:
                dst_points[x,y] = src[np.int64(a),np.int64(b)]
    return dst_points
'''
def my_warp2(src, H):
    height, width = 1000, 1000
    # height, width = src.shape[:2]
    dst_points = np.zeros((height, width, 3), dtype=np.float64)
    # print(H)
    # print((H[0,2], H[1,2], H[2,2]))
    # quit()
    cnt_tru=0
    for x in np.arange(-height/2, height/2):
        for y in np.arange(-width/2, width/2):
            a = np.float64((H[0,0]*x + H[0,1]*y + H[0,2])/(H[2,0]*x + H[2,1]*y + H[2,2]) + src.shape[0]/2)
            b = np.float64((H[1,0]*x + H[1,1]*y + H[1,2])/(H[2,0]*x + H[2,1]*y + H[2,2]) + src.shape[1]/2)

            # if a < height and a > 0 and b < width and b > 0:
            if a < src.shape[0] and a > 0 and b < src.shape[1] and b > 0:
            # if np.abs(a) < height/2 and np.abs(b) < width/2:
                cnt_tru+=1
                dst_points[np.int64(x+height/2),np.int64(y+width/2)] = src[np.int64(a),np.int64(b)]
                dst_points[10,10] = [255,255,0]
        # print(y)
    return dst_points

def my_warp3(src, H):
    # height, width = 1000, 1000
    # width, height = 1000, 10000
    height, width = src.shape[:2]
    dst_points = np.zeros((height, width, 3), dtype=np.float64)
    # print(H)
    # print((H[0,2], H[1,2], H[2,2]))
    # quit()
    cnt_tru=0

    x_vec = np.arange(-height/2, height/2)
    y_vec = np.arange(-width/2, width/2)

    Y, X = np.meshgrid(y_vec, x_vec)

    # print(X.shape)
    # print(Y.shape)

    a_vec = np.zeros((height, width))
    b_vec = np.zeros((height, width))

    # print(a_vec.shape)
    # print(b_vec.shape)

    # for x in np.arange(height):
    #     for y in np.arange(width):

    a_vec = np.float32((H[0,0]*X + H[0,1]*Y + H[0,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]) + src.shape[0]/2)
    b_vec = np.float32((H[1,0]*X + H[1,1]*Y + H[1,2])/(H[2,0]*X + H[2,1]*Y + H[2,2]) + src.shape[1]/2)

    dst_points = cv.remap(src, b_vec, a_vec, 0)
    # print(dst_points.shape)
    # quit()
    return dst_points





    '''

    a_vec = np.zeros((height, width))
    b_vec = np.zeros((height, width))

    for x in np.arange(-height/2, height/2):
        for y in np.arange(-width/2, width/2):
            a = np.float64((H[0,0]*x + H[0,1]*y + H[0,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])+ src.shape[0]/2)
            b = np.float64((H[1,0]*x + H[1,1]*y + H[1,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])+ src.shape[1]/2)
            # a_vec[np.int64(x), np.int64(y)] = a
            # b_vec[np.int64(x), np.int64(y)] = b
            a_vec[np.int64(x+height/2), np.int64(y+width/2)] = a
            b_vec[np.int64(x+height/2), np.int64(y+width/2)] = b


    # a_vec[a_vec<0] = 0
    # b_vec[b_vec<0] = 0
    #
    # a_vec[a_vec>height] = 0
    # b_vec[b_vec>width] = 0
    #
    # print(np.amin(a_vec))
    # print(np.amax(a_vec))
    # print(np.amin(b_vec))
    # print(np.amax(b_vec))

    a_vec = np.float32(a_vec)
    b_vec = np.float32(b_vec)
    # src[0,0] = 255,255,0
    # src[:,0] = 255,255,0
    # src[0,:] = 255,255,0
    dst_points = cv.remap(src, b_vec, a_vec, 0)
                #
                # cnt_tru+=1
                # dst_points[np.int64(x+height/2),np.int64(y+width/2)] = src[np.int64(a),np.int64(b)]
                # dst_points[10,10] = [255,255,0]
        # print(y)
    print(dst_points.shape)
    print(a_vec.shape)
    print(b_vec.shape)

    quit()
    return dst_points

    '''


def my_warp_vec(src, H):
    height, width = 1000, 1000
    dst_points = np.zeros((height, width, 3), dtype=np.float64)

    x_vec = np.arange(-height/2, height/2)
    y_vec = np.arange(-width/2, width/2)

    a_vec = np.int64((H[0,0]*x_vec + H[0,1]*y_vec + H[0,2])/(H[2,0]*x_vec + H[2,1]*y_vec + H[2,2]) + src.shape[0]/2)
    b_vec = np.int64((H[1,0]*x_vec + H[1,1]*y_vec + H[1,2])/(H[2,0]*x_vec + H[2,1]*y_vec + H[2,2]) + src.shape[1]/2)

    print((a_vec.shape, b_vec.shape))
    cnt_tru = 0
    for x in np.int64(x_vec+height/2):
        for y in np.int64(y_vec+width/2):
            if a_vec[x] < src.shape[0] and a_vec[x] > 0 and b_vec[y] < src.shape[1] and b_vec[y] > 0:
                cnt_tru+=1
                dst_points[np.int64(x),np.int64(y)] = src[a_vec[x],b_vec[y]]
    print(cnt_tru)
    return dst_points
def qsda():
    print((a_vec.shape, b_vec.shape))

    truth_a = a_vec < src.shape[0]
    truth_a = np.logical_and(truth_a, a_vec > 0)
    truth_b = b_vec > 0
    truth_b = np.logical_and(truth_b, b_vec < src.shape[1])
    # print(truth.dtype)

    # a_vec = a_vec[a_vec<src.shape[0]]
    # a_vec = a_vec[a_vec>0]
    # b_vec = b_vec[b_vec<src.shape[1]]
    # b_vec = b_vec[b_vec>0]



    print((a_vec.shape, b_vec.shape))
    # print(dst_points[np.int64(x_vec[truth_a]+height/2), np.int64(y_vec[truth_b]+width/2)].shape)
    print(src[a_vec[truth_a],b_vec[truth_b]].shape)
    dst_points[np.int64(x_vec[truth_a]+height/2), np.int64(y_vec[truth_b]+width/2)] = src[a_vec[truth_a],b_vec[truth_b]]
    # dst_points[x_vec+height/2, y_vec+width/2] = src[a_vec,b_vec]

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


# whitened = whiten(src)
# code_book, distortion = kmeans(whitened, 3)
# clusters, dist = vq(whitened, code_book)
# print(clusters)
# test_im = np.zeros_like(img)
# print(np.max(src))
# print(src.shape)
# test_im[src[:,0],src[:,1]] = [255,255,255]
# test_im[src[clusters==0,0],src[clusters==0,1]] = [255,0,255]
# test_im[src[clusters==1,0],src[clusters==1,1]] = [255,0,0]
# test_im[src[clusters==2,0],src[clusters==2,1]] = [0,0,255]

# cv.imshow('test', test_im)
# cv.imwrite('./test.png', test_im)
#
# # src = np.float32([[0, 1080], [1920, 1080], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
# # dst = np.float32([[780, IMAGE_H], [1020, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# Minv = cv.getPerspectiveTransform(dst, src) # Inverse transformation
# homo, _ = cv.findHomography(src,dst)
# retval, rotations, translations, normals = cv.decomposeHomographyMat(homo, K)
#
#
# # print(M)
# # print(np.dot(src,M))
# # print(Minv)
# # print(homo)
# for r, t, n in zip(rotations, translations, normals):
#     print(R.from_matrix(r).as_euler('xyz', degrees=True))
#     print(t)
#     print(n)
# # print(rotations)
# # print(translations)
# # print(normals)
#
#
#
# # img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
# warped_img = cv.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
# # cv.imwrite('./00-08-52-932_warp_lines.png', warped_img)
# plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)) # Show results
# plt.show()
def get_warp_pos(left, middle, right):
    l_cand = [[0, 1, 2, 3], [0, 1, 2, 3]]
    m_cand = [[1, 2], [1,2]]
    r_cand = [[2, 3], [2,3]]
    return l_cand, m_cand, r_cand

if __name__ == '__main__':
    birdview()
