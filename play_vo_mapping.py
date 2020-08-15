import os
import cv2
import csv
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy.matlib as ml
from build_pointcloud import build_pointcloud


import my_params

from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from transform import build_se3_transform
from camera_model import CameraModel

# Load data

vo_directory = my_params.dataset_patch + 'vo//vo.csv'

lidar_folder_path = my_params.laser_dir
lidar_timestamps_path = my_params.dataset_patch + 'ldmrs.timestamps'

# Making Video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vw = cv2.VideoWriter('laser_scan.mp4', fourcc, 30, (600, 800))

C_origin=np.zeros((3,1))
R_origin=np.identity(3)
# xyzrpy = [0, 0, 0, -0.090749, -0.000226, 4.211563]
xyzrpy = [0, 0, 0, -0.083514, -0.009355, 1.087759]    # FOR PC

abs_pose = build_se3_transform(xyzrpy)
abs_poses = [abs_pose]
H_new = abs_pose

# fig = plt.figure()
# gs = plt.GridSpec(2,3)

vo_timestamps = []

# Blank image
reso = [600,800]
scale = 0.5
vo_map = np.zeros((reso[0],reso[1],3),np.uint8)          # 800*600 --> scale=1
start_map = (300,150)

index = 0
# vo_map = cv2.circle(vo_map,start_map,5,(0,0,255),thickness=3)

#$#$#$#$#$#$#$
TEST_data = [50,0,30,0,0,0]
TEST_pose = build_se3_transform(TEST_data)
#$#$#$#$#$#$#$

with open(vo_directory) as vo_file:
    vo_reader = csv.reader(vo_file)
    headers = next(vo_file)

    for row in vo_reader:
        timestamp = int(row[0])
        vo_timestamps.append(timestamp)
        datetime = dt.utcfromtimestamp(timestamp/1000000)
        xyzrpy = [float(v) for v in row[2:8]]
        rel_pose = build_se3_transform(xyzrpy)

        H_new=H_new@rel_pose    # wolrd-cord transform
        x_test=H_new[1,3]
        y_test=H_new[0,3]

        abs_poses.append(H_new)

        # plot map
        x_map = int(x_test + float(start_map[0]))
        y_map =  int(-y_test +  float(start_map[1]))

        # print(x_test,z_test,":",x_map,y_map)
        vo_map = cv2.circle(vo_map,(x_map,y_map),1,(255,0,0),thickness=1)

        #$#$#$#$#$#$#$
        # DRAW_pose=H_new@TEST_pose    # wolrd-cord transform
        # x_TEST_pose=int(float(start_map[0]) + DRAW_pose[1,3])
        # y_TEST_pose=int(float(start_map[1]) - DRAW_pose[0,3])
        # vo_map = cv2.circle(vo_map,(x_TEST_pose,y_TEST_pose),1,(0,255,0),thickness=1)
        #$#$#$#$#$#$#$
        # ----> OK
        
        abs_poses.append(H_new)

        # Draw pointcloud --------------------------------------
        draw_flag = 0
        with open(lidar_timestamps_path) as lidar_timestamps_file:
            for _,row in enumerate(lidar_timestamps_file):
                lidar_timestamps = int(row.split(' ')[0])
                if(lidar_timestamps < timestamp):
                    continue
                
                if((lidar_timestamps >= timestamp and draw_flag == 0)):
                    start_time = timestamp
                    end_time = timestamp + 5e6
                    draw_flag = 1

                if(lidar_timestamps >= end_time and draw_flag == 1):
                    print("Cann't find lidar file")
                    break

                if(draw_flag == 1):
                    file_path = os.path.join(lidar_folder_path + '\\' + str(lidar_timestamps) + '.bin')
                    scan_file = open(file_path)
                    objs_data = np.fromfile(scan_file, np.double)
                    scan_file.close()
                    objs_data = objs_data.reshape((len(objs_data) // 3, 3)).transpose()

                    # Change one by one pixel  # CV2 image
                    for i in range(500):
                    # for i in range(objs_data.shape[1]):
                        obj_data = [objs_data[0][i],objs_data[1][i],objs_data[2][i],0,0,0]        #xyzrpy
        #                 # print(obj_data)
                        obj_pose = build_se3_transform(obj_data)

                        obj_pose=H_new@obj_pose    # wolrd-cord transform
                        x_obj=int(float(start_map[0]) + float(obj_pose[1,3]))
                        y_obj=int(float(start_map[1]) - float(obj_pose[0,3]))
                        vo_map = cv2.circle(vo_map,(x_obj,y_obj),1,(0,255,0),thickness=1)
                        
                    # DRAW_pose=H_new@TEST_pose    # wolrd-cord transform
                    # x_TEST_pose=int(float(start_map[0]) + DRAW_pose[1,3])
                    # y_TEST_pose=int(float(start_map[1]) - DRAW_pose[0,3])
                    # vo_map = cv2.circle(vo_map,(x_TEST_pose,y_TEST_pose),1,(0,255,0),thickness=1)
                    break

        lidar_timestamps_file.close()

        #     # # Draw angle of laser
        #     # # cv2.line(point_map, (200, 299), (399, 125), (0, 255, 0), thickness=1)
        #     # # cv2.line(point_map, (199, 299), (0, 125), (0, 255, 0), thickness=1)
        #     # # Save
        #     # cv2.imshow("point_map",point_map)
        #     # # vw.write(img)  #Save video
        #     # # cv2.waitKey(0)

        #     # kernel = np.ones((2,2), np.uint8)
        #     # key_image = cv2.dilate(point_map, kernel, iterations=5)
        #     # key_image = cv2.erode(key_image, kernel, iterations=5)

        #     # cv2.imshow("key_image",key_image)
        #     # cv2.waitKey(0)

        index = index + 1

        if (index > 1000):
            break
        
cv2.imshow('vo_map',vo_map)
cv2.waitKey(0)
cv2.imwrite('vo_map.jpg',vo_map)
print('done!')
cv2.destroyAllWindows()

vo_timestamps = vo_timestamps[1:]
abs_poses = abs_poses[1:]

# print('vo_timestamps',len(vo_timestamps))   #2892
# print('abs_poses',len(abs_poses[1:]))       #2892


################# PLOT POINTCLOUD ##############################


# # # Load LiDAR data
# timestamps_path = my_params.dataset_patch + 'ldmrs.timestamps'
# # lidar_folder_path = my_params.laser_dir
# # # fig = plt.figure()

# timestamp = 0
# with open(timestamps_path) as timestamps_file:
#     for i, line in enumerate(timestamps_file):
#         if (i > 0 and i < 10):       
#             timestamp = int(line.split(' ')[0])
#             datetime = dt.utcfromtimestamp(timestamp/1000000)
#             # print(datetime)

#             file_path = os.path.join(lidar_folder_path + '\\' + str(timestamp) + '.bin')
#             scan_file = open(file_path)

#             data = np.fromfile(scan_file, np.double)
#             scan_file.close()
#             data = data.reshape((len(data) // 3, 3)).transpose()
        
#             # Blank image
#             reso = [300,400]
#             scale = reso[0]/50
#             point_map = np.zeros((reso[0],reso[1],3),np.uint8)          # 800*600 --> scale=12

#             # Change one by one pixel  # CV2 image
#             for i in range(data.shape[1]):
#                 x =  scale*data[1][i] + reso[1]/2    # width of the frame
#                 y = -scale*data[0][i] + reso[0]      # distance in front view
                
#                 # if (x < 0) : x = max(0,x)
#                 # if (x >= reso[1]) : x = min((reso[1]-1),x)
#                 # y = max(min((reso[0]-1),y),0)
#                 if (x < 0 or x >= reso[1] or y < 0 or y >= reso[0]):
#                     continue

#                 point_map[int(y),int(x)]=(255,255,255)

#     # Draw angle of laser
#     # cv2.line(point_map, (200, 299), (399, 125), (0, 255, 0), thickness=1)
#     # cv2.line(point_map, (199, 299), (0, 125), (0, 255, 0), thickness=1)
#     # Save
#     cv2.imshow("point_map",point_map)
#     # vw.write(img)  #Save video
#     cv2.waitKey(0)
#     cv2.imwrite('point_map.jpg',point_map)
#     kernel = np.ones((2,2), np.uint8)
#     key_image = cv2.dilate(point_map, kernel, iterations=5)
#     key_image = cv2.erode(key_image, kernel, iterations=5)

#     cv2.imshow("key_image",key_image)
#     cv2.waitKey(0)




################# PLOT POINTCLOUD ##############################





