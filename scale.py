import numpy as np
import cv2 as cv
import skimage.transform as tf
from scipy.spatial.transform import Rotation as R
from birdview import get_homography2, K
from scipy.ndimage import rotate


def scale(img):
    scale = tf.AffineTransform(scale=(1, 1.739))
    scaled_img = tf.warp(img, inverse_map=scale)
    scaled_img *= 255


    return scaled_img

def affine(img, pitch):
    rot = R.from_euler('xyz', [-pitch, 0, 0], degrees=True).as_matrix()
    H = get_homography2(rot, K)
    # tform = tf.ProjectiveTransform(matrix=rot)
    tform = tf.ProjectiveTransform(matrix=H)
    tf_img = tf.warp(img, tform.inverse)
    tf_img *= 255
    return tf_img

def my_rotate(img, pitch):
    rotated = rotate(img, pitch, (1,2), reshape=False)
    return rotated
    
if __name__ == '__main__':
    # print(K)
    img = cv.imread('calib_rotated.png')
    img2 = cv.imread('calib_yb.png')
    scaled_img = scale(img)
    cv.imwrite('scaled_test.png', scaled_img)
    # cv.imwrite('scaled_test.png', img)
    affine_img = affine(img2, 50)
    cv.imwrite('warped_test.png', affine_img)


    img_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray_3d = np.stack((img_gray, np.ones_like(img_gray)), 2)
    print(gray_3d.shape)
    # quit()
    rotated_img = my_rotate(gray_3d, 20)
    print(rotated_img.shape)
    print(np.unique(rotated_img[2], return_counts=True))
    quit()


    cv.imwrite('rotated_test.png', rotated_img)
