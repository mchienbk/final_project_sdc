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
from transform import build_se3_transform

import my_params

np.set_printoptions(suppress=True)

DRAW_ON_IMAGE = True

output_image_dir = my_params.output_dir+'\\'+ my_params.dataset_no + '\\'
output_points_patch = my_params.output_dir+'\\'+ my_params.dataset_no + '_vo_points.csv'

if __name__ == '__main__':
    # Intial data
    # H = np.identity(4)      # intial pose
    xyzrpy = [620021.4778138875, 0, 5734795.685425566, 0.0128231,-0.0674645,-0.923368707] #2015
    H = build_se3_transform(xyzrpy)
    print(H)
    # H = np.identity(4)      # intial pose
    poses = []              # poses list
    final_points=[]         # positions list

    # First point
    poses.append(H)
    final_points.append((620021.4778138875, 5734795.685425566))
 
    x_0 = (H[0,3])
    z_0 = (H[2,3])
    print(x_0,z_0)
    plt.plot(x_0,z_0,'o',color='red')
    plt.show()
    # Get image    
    image_dir = my_params.image_dir
    model_dir = my_params.model_dir
    reprocess_dir = my_params.reprocess_image_dir
    scale = 0.5

    width = int(1280 * scale)
    height = int(720 * scale)
    dim = (width, height)

    # Get camera model
    intrinsics_path = model_dir + "\\stereo_narrow_left.txt"
    # lut_path = models_dir + "\\stereo_narrow_left_distortion_lut.bin"
    intrinsics = np.loadtxt(intrinsics_path)
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]


    fig = plt.figure()
    plt.gca().invert_yaxis()
    # gs = plt.GridSpec(2,3)

    frames = 0
    current_chunk = 0
    start = time.time() 

    camera = re.search('(stereo|mono_(left|right|rear))', image_dir).group(0)
    timestamps_path = os.path.join(os.path.join(image_dir, os.pardir, os.pardir, camera + '.timestamps'))
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
        
        # Read and resize image
        img_next_frame = cv2.imread(filename)
        img_next_frame= cv2.rectangle(img_next_frame,(np.float32(50),np.float32(np.shape(img_next_frame)[0])),(np.float32(1250),np.float32(800)),(0,0,0),-1)
        img_next_frame = cv2.resize(img_next_frame, dim, interpolation = cv2.INTER_AREA)

        if (frames < 1):
            img_current_frame = img_next_frame
        else:
            sift = cv2.xfeatures2d.SIFT_create()
            
            # Find the keypoints and descriptors with SIFT
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
            # fix emty data #
            # if (len(U) <= 0 or len(V) <= 0):

            # fix emty data #
            E, _ = cv2.findEssentialMat(U, V, focal=fx, pp=(cx,cy), method=cv2.RANSAC, prob=0.999, threshold=0.5)
            _, cur_R, cur_t, mask = cv2.recoverPose(E, U, V, focal=fx, pp=(cx,cy))        	
            if np.linalg.det(cur_R)<0:
        	    cur_R = -cur_R
        	    cur_t = -cur_t
            new_pose = np.hstack((cur_R, cur_t))
            new_pose = np.vstack((new_pose,np.array([0,0,0,1])))
            # x_old = (H[0][3])
            # z_old = (H[2][3])
            H = H@new_pose
            x_new = (H[0,3])
            z_new = (H[2,3])
            
            # Backup data
            poses.append(H)
            final_points.append((x_new,z_new))
            # print('from ', (x_old,z_old), " to ",(x_new,z_new))

            # Plot trajectory in plt
            plt.plot(x_new,z_new,'.',color='blue')
            plt.pause(0.01)
        
        # cv2.imshow("img_current_frame",img_current_frame)
        # cv2.imshow("Camera",img_next_frame)
        
        # Draw point to image
        if (DRAW_ON_IMAGE == True):
            output_frame = img_next_frame
            if (frames > 0):
                for (x,y) in V:
                    output_frame = cv2.circle(output_frame,(int(x),int(y)),1,(0,255,0),thickness=1)
            # cv2.imshow("Output",output_frame)
            cv2.imwrite(output_image_dir + tokens[0] + '.png',output_frame)


        frames += 1
        img_current_frame = img_next_frame

        # key = cv2.waitKey(5)
        # if key & 0xFF == ord('q'):
        #     break
        # if (frames > 50): break
        # print(datetime)
        # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    
    # Save and quit
    np.savetxt(output_points_patch, final_points, delimiter=",")
    print('done!')
    cv2.destroyAllWindows()

    