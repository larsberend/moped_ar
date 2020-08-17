import numpy as np
import cv2 as cv

def combine(file, folder):
    cap = cv.VideoCapture(file)
    frame_int = 0
    bv_shape = 500,500,3
    while(cap.isOpened):
        ret, frame = cap.read()
        if ret:
            print(frame_int)
            bv = cv.imread('%s%s.png'%(folder, frame_int))
            if np.unique(np.all(bv==[0,0,0], axis=-1)).shape[0] < 2:#, axis=-1):
                bv = np.zeros(bv_shape)
                # print(bv.shape)
                # quit()
            else:
                bv = cv.resize(bv, (np.int32(bv.shape[0]/2), np.int32(bv.shape[1]/2)))
            frame[:bv.shape[0], :bv.shape[1]] = bv
            cv.imwrite('../100GOPRO/testfahrt_1006/kandidaten/3_2_bv_proc/%s.png'%(frame_int), frame)
            frame_int += 1


if __name__=='__main__':
    combine('../100GOPRO/testfahrt_1006/kandidaten/3_2_processed/3_2_processed_2.mp4', '../100GOPRO/testfahrt_1006/kandidaten/3_2_birdview/')
