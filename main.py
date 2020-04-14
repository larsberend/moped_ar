import numpy as np
import pandas as pd
from cordFrames import cordFrame, worldFrame, transform
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last
from skimage.draw import circle_perimeter
import cv2 as cv

f = np.float64(1.9/5.6)
AoV_hor = 1.476549
AoV_ver = 1.181588
z_far = 1000
z_near = 50

def main():
    # world coordinate frame, origin at ground level of world-plane, contact point of front-wheel and road
    # +X: left(!), +Y: Up, +Z TO the front (!) of motorcycle
    world_frame = worldFrame()
    # print(worldFrame.name)
    # plane coordinate frame
    # same as world
    plane_frame = cordFrame(data=None,
                            name='plane',
                            world_frame=world_frame,
                            parent=world_frame,
                            children=[],
                            transToParent=transform(np.array([0, 0, 0], dtype=np.float64),
                                          R.from_quat(np.array([0, 0, 0, 1], dtype=np.float64)))
                            )

    # imu coordinate frame
    # lies 121,37 cm above origin of world frame
    # +X: up (away from world origin), +Y: to the left(? TODO!!!) +Z: to the back of motorcycle

    imu_frame = cordFrame(data=None,
                          name='IMU',
                          world_frame=world_frame,
                          parent=world_frame, children=[],
                          transToParent=transform(np.array([0, -121.37, 0], dtype=np.float64),
                                                  R.from_quat(np.array([[0, 0, np.sin(np.pi/4),np.cos(np.pi/4)],
                                                              [np.sin(np.pi/2), 0, 0, np.cos(np.pi/2)]
                                                             ], dtype=np.float64))
                                                 )
                        )

    # blau rot vertauscht
    # camera coordinate frame
    # position same as IMU
    # orientation as a camera: +X: right (width), +Y: down (to worldframe, height), +Z in driving direction

    cam_frame = cordFrame(data=None,
                          name='Camera',
                          world_frame=world_frame,
                          parent=world_frame,
                          children=[],
                          transToParent=transform(np.array([0, -121.37, 0], dtype=np.float64),
                          R.from_quat(np.array([0, np.sin(np.pi/2), 0, np.cos(np.pi/2)], dtype=np.float64)))
                          )

    # img_frame = cordFrame(data=None,
    #                         name='Image',
    #                         world_frame=world_frame,
    #                         parent=cam_frame,
    #                         children=[],
    #                         transToParent=transform(np.array([0, 0, 0], dtype=np.float64),
    #                         R.from_quat(np.array([0, 0, np.sin(np.pi/2), np.cos(np.pi/2)], dtype=np.float64)))
    #
    #                         )

    # print(world_frame)
    # print(plane_frame)
    # print(imu_frame)
    # print(cam_frame)

    # get curve radius in m, convert to cm
    radius = 20
    radius *= 10
    # calculate circle perimeter in 2d, extend to 3d(4d) with y-values=0 (1 for 4th dim)
    curve2 = np.array(circle_perimeter(np.int(radius), 0, np.abs(radius)))
    # eliminate coords behind camera
    a = curve2[0].tolist()
    b = curve2[1].tolist()
    indexes_to_remove = []
    for x in range(len(a)-1, -1, -1):
        if a[x]<=0 or b[x]<=0:
            del(a[x])
            del(b[x])

    print(a)
    print(b)
    curve2 = (np.asarray(a), np.asarray(b))

    curve3d = (curve2[0], np.zeros(curve2[0].shape, dtype=np.int64),curve2[1], np.ones(curve2[0].shape, dtype=np.int64))
    curve3d = np.column_stack(curve3d).astype(np.float64)
    print(curve3d)
    # camera rotation inverse (from parent(world) to camera), homogeneous

    cam_mat = cam_frame.transToWorld.rotQuat.inv().as_matrix()
    homog_rot = np.zeros((cam_mat.shape[0]+1, cam_mat.shape[1]+1))
    # print(homog_rot)
    homog_rot[:-1,:-1] = cam_mat
    homog_rot[-1,-1] = 1

    # homogeneous camera translation
    cam_trans = cam_frame.transToWorld.trans
    homog_pos = np.identity(4)
    homog_pos[:-1, -1] =- cam_trans

    # combine trans and rot
    trans_rot = np.dot(homog_rot, homog_pos)
    print(trans_rot)
    # curve_homog = np.zeros((curve3d.shape[0], curve3d.shape[1]))
    curve_cam = np.zeros_like(curve3d)
    curve_proj = np.column_stack(np.zeros_like(curve2)).astype(np.float64)

    # perp_proj_mat = np.zeros((4,4), dtype=np.float64)
    # perp_proj_mat[0,0] = np.arctan(AoV_hor/2)
    # perp_proj_mat[1,1] = np.arctan(AoV_ver/2)
    # perp_proj_mat[2,2] = 2/(z_far-z_near)
    # perp_proj_mat[2,3] = (z_far+z_near)/(z_far-z_near)
    # perp_proj_mat[3,2] = -1
    # print(perp_proj_mat)


    for i in range(curve3d.shape[0]):
        # print(homog_rot[i])
        # print(homog_pos[i])

        # curve_homog[i] = np.dot(trans_rot, curve3d[i])

        curve_cam[i] = np.dot(trans_rot, curve3d[i])

        # view = np.dot(curve_cam[i], perp_proj_mat)
        # view = np.dot(perp_proj_mat, curve_cam[i])

        # print(view)

        # image space :
        # x = view[0]/view[3]
        # y = view[1]/view[3]
        # z = view[2]/view[3]
        x = f * (curve_cam[i][0]/-curve_cam[i][2]) + 0.617/2
        y = f * (curve_cam[i][1]/-curve_cam[i][2]) + 0.455/2

        x_norm = (x+1)/2
        y_norm = (y+1)/2

        x_rast = np.floor(x_norm * 1920)
        y_rast = np.floor(y_norm * 1080)
        curve_proj[i] = x_rast,y_rast
        # calc pos on 'film' of camera, discard everythin behind camera.
        # if np.greater(curve_cam[i,2],0):
        #     x_film = f * (curve_cam[i,0]/curve_cam[i,3]) + 0.617/2
        #     y_film = f * (curve_cam[i,1]/curve_cam[i,3]) + 0.455/2
        #     # curve_proj[i] = x_film*100, y_film*100
        #     curve_proj[i] = x*100, y*100

    print(np.amin(curve_proj[0, :]))
    print(np.amax(curve_proj[0, :]))
    print(np.mean(curve_proj[0, :]))

    print(np.amin(curve_proj[1, :]))
    print(np.amax(curve_proj[1, :]))
    print(np.mean(curve_proj[1, :]))

    # print(curve_proj)

    img = np.zeros((1080, 1920, 3),np.uint8)

    for j in curve_proj.astype(np.int32):
        cv.circle(img, (j[0], j[1]), 5, (0,255,0))

    cv.imwrite('curve.png', img)




    return curve_proj
















if __name__=='__main__':
    main()
