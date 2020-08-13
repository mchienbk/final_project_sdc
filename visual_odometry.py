import cv2
import re
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import os
import time
from datetime import datetime as dt
from camera_model import CameraModel
from image import load_image
import my_params


np.set_printoptions(suppress=True)
H = np.identity(4)

final_points=[]

if __name__ == '__main__':

    # Get image    
    image_dir = my_params.image_dir
    model_dir = my_params.model_dir
    reprocess_dir = my_params.reprocess_image_dir
    scale = 0.1

    width = int(1280 * scale)
    height = int(720 * scale)
    dim = (width, height)

    x_old=(0,0)
    z_old=(0,0)

    intrinsics_path = model_dir + "/stereo_narrow_left.txt"
    # lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"
    intrinsics = np.loadtxt(intrinsics_path)
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]


    fig = plt.figure()
    gs = plt.GridSpec(2,3)

    frames = 0
    start = time.time() 

    camera = re.search('(stereo|mono_(left|right|rear))', image_dir).group(0)

    timestamps_path = os.path.join(os.path.join(image_dir, os.pardir, os.pardir, camera + '.timestamps'))

    current_chunk = 0
    timestamps_file = open(timestamps_path)
    for line in timestamps_file:
        tokens = line.split()
        datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
        chunk = int(tokens[1])

        filename = os.path.join(reprocess_dir + '//' + tokens[0] + '.png')

        if not os.path.isfile(filename):
            if chunk != current_chunk:
                print("Chunk " + str(chunk) + " not found")
                current_chunk = chunk
            continue

        current_chunk = chunk
        
        # read and resize image
        img_next_frame = cv2.imread(filename)
        img_next_frame= cv2.rectangle(img_next_frame,(np.float32(50),np.float32(np.shape(img_next_frame)[0])),(np.float32(1250),np.float32(800)),(0,0,0),-1)
        img_next_frame = cv2.resize(img_next_frame, dim, interpolation = cv2.INTER_AREA)

        if (frames < 1):
            img_current_frame = img_next_frame
        else:
            sift = cv2.xfeatures2d.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img_current_frame,None)
            kp2, des2 = sift.detectAndCompute(img_next_frame,None)

            # FLANN parameters
            bf=cv2.BFMatcher()
            matches=bf.match(des1,des2)
            U = []
            V = []
            for m in matches:        
                pts_1 = kp1[m.queryIdx]
                x1,y1=pts_1.pt
                pts_2 = kp2[m.trainIdx]
                x2,y2=pts_2.pt
                U.append((x1,y1))
                V.append((x2,y2))
            U=np.array(U)
            V=np.array(V)
            # fix emty data
            
            # fix emty data
            E, _ = cv2.findEssentialMat(U, V, focal=fx, pp=(cx,cy), method=cv2.RANSAC, prob=0.999, threshold=0.5)
            _, cur_R, cur_t, mask = cv2.recoverPose(E, U, V, focal=fx, pp=(cx,cy))        	
            if np.linalg.det(cur_R)<0:
        	    cur_R = -cur_R
        	    cur_t = -cur_t
            new_pose = np.hstack((cur_R, cur_t))
            new_pose = np.vstack((new_pose,np.array([0,0,0,1])))
            x_old = (H[0][3])
            z_old = (H[2][3])
            H = H@new_pose
            x_new = (H[0][3])
            z_new = (H[2][3])
            
            print((x_old,z_old), " : ",(x_new,z_new))

            plt.plot(x_new,-z_new,'o',color='blue')
            plt.pause(0.01)
        
        # plt.show()

        cv2.imshow("img_current_frame",img_current_frame)
        cv2.imshow("img_next_frame",img_next_frame)

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break



        frames += 1
        img_current_frame = img_next_frame
        print(datetime)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
