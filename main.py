import numpy as np
import pandas as pd
from cordFrames import cordFrame, worldFrame, transform, get_cordFrames
from scipy.spatial.transform import Rotation as R # quaternion in scalar-last
from skimage.draw import circle_perimeter, line
from draw_curve import draw_curve
from radius import turn
import cv2 as cv
from CMadgwick_wenigSelf import CMad

def main():

    world_frame, plane_frame, cam_frame = get_cordFrames()

    csv_path = '../100GOPRO/kandidaten/csv/'
    video_path = '../100GOPRO/kandidaten/'
    file = '3_2'
    start_msec = 4000.0
    test = turn()
    cmad = CMad()


    radius_madgwick = pd.read_csv(csv_path + file + '-gyroAcclGpsMadgwick.csv')[['Milliseconds','Radius']].to_numpy()[2:].swapaxes(1,0)
    # print(radius_madgwick)
    # print(video_path+'.mp4')
    radius_madgwick[0] -= radius_madgwick[0,0]
    cap = cv.VideoCapture(video_path + file + '.mp4')
    # print(cap.isOpened())
    out = cv.VideoWriter('3_2_curve.avi', cv.VideoWriter_fourcc('M','J','P','G'), 59.94, (1920,1080))
    font = cv.FONT_HERSHEY_SIMPLEX
    cap.set(0, start_msec)
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # print(ret)
        if ret:
            height,width = frame.shape[:2]
            nearest_rad = find_nearest(radius_madgwick[0], cap.get(0))
            # print(radius_madgwick[0])
            # print(cap.get(0))
            # print(nearest_rad)
            # print(radius_madgwick[0,nearest_rad])

            mil, radius = radius_madgwick[:,nearest_rad]
            cv.putText(frame, 'Frame Nr: %s'%(cap.get(0)), (1650,20), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, 'Radius: %s'%(radius), (1650,40), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, 'MIlliseconds(IMU): %s'%(mil), (1650,60), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)
            # print(radius)
            curve_proj, bird_view = draw_curve(radius, cam_frame)
            for j in curve_proj.astype(np.int32):
                if j[0]<1920 and j[1]<1080:
                    cv.circle(frame, (j[0], j[1]), 5, (0,255,0))

            # Display the resulting frame
            # print(frame.shape)
            # print(bird_view.shape)
            x_offset = width - bird_view.shape[1]
            y_offset = np.int(height/3) - bird_view.shape[0]
            frame[y_offset:y_offset+bird_view.shape[0], x_offset:x_offset+bird_view.shape[1]] = bird_view

            # out.write(frame)
            cv.imshow('frame', frame)
            cv.imwrite('problem2.png', frame)
            quit()
            # cv.imwrite('frameNr_' + str(cap.get(0)) + '.png', frame)
            # current += 4
            # print(radius)
            # cv.waitKey(0)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv.destroyAllWindows()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # print(idx)
    return idx





if __name__=='__main__':
    main()
