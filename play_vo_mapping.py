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
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# vw = cv2.VideoWriter('laser_scan.mp4', fourcc, 30, (600, 800))

C_origin=np.zeros((3,1))
R_origin=np.identity(3)
xyzrpy = [0, 0, 0, -0.083514, -0.009355, 1.087759]

abs_pose = build_se3_transform(xyzrpy)
abs_poses = [abs_pose]
H_new = abs_pose
vo_timestamps = []

# Blank image
reso = [600,800]
scale = 0.5
vo_map = np.zeros((reso[0],reso[1],3),np.uint8)          # 800*600 --> scale=1
start_map = (300,150)

# vo_map = cv2.circle(vo_map,start_map,5,(0,0,255),thickness=3)
index = 0

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
        vo_map = cv2.circle(vo_map,(x_map,y_map),1,(255,0,0),thickness=1)
       
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

                if(lidar_timestamps >= end_time):
                    print("Can't find lidar file")
                    break

                if(draw_flag == 1):
                    file_path = os.path.join(lidar_folder_path + '\\' + str(lidar_timestamps) + '.bin')
                    scan_file = open(file_path)
                    objs_data = np.fromfile(scan_file, np.double)
                    scan_file.close()
                    objs_data = objs_data.reshape((len(objs_data) // 3, 3)).transpose()

                    # Change one by one pixel
                    for i in range(objs_data.shape[1]):
                        obj_data = [objs_data[0][i],objs_data[1][i],objs_data[2][i],0,0,0]        #xyzrpy
                        obj_pose = build_se3_transform(obj_data)
                        obj_pose=H_new@obj_pose    # wolrd-cord transform
                        x_obj=int(float(start_map[0]) + float(obj_pose[1,3]))
                        y_obj=int(float(start_map[1]) - float(obj_pose[0,3]))
                        vo_map = cv2.circle(vo_map,(x_obj,y_obj),1,(0,255,0),thickness=1)
                    break

        lidar_timestamps_file.close()

        index = index + 1
        if(draw_flag == 0):
            print("vo_error: ", index)
        if (index > 100):
            break
        
cv2.imshow('vo_map',vo_map)
# cv2.imwrite('output\\vo_map.jpg',vo_map)
cv2.waitKey(0)

print('done!')
cv2.destroyAllWindows()

vo_timestamps = vo_timestamps[1:]
abs_poses = abs_poses[1:]

# print('vo_timestamps',len(vo_timestamps))   #2892
# print('abs_poses',len(abs_poses[1:]))       #2892






