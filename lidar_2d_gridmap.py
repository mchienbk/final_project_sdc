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
from itertools import islice

import my_params

from interpolate_poses import interpolate_poses
from transform import build_se3_transform
from camera_model import CameraModel


# Data location
vo_directory = my_params.dataset_patch + 'vo//vo.csv'
lidar_folder_path = my_params.laser_dir
lidar_timestamps_path = my_params.dataset_patch + 'ldmrs.timestamps'


# Making Video
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# vw = cv2.VideoWriter('laser_scan.mp4', fourcc, 30, (600, 800))

# Blank image
# reso = [600,800]
scale = 5
# vo_map = np.zeros((reso[0],reso[1],3),np.uint8)          # 800*600 --> scale=1
start_map = (300,150)

# Intial data
xyzrpy = [0, 0, 0, -0.090749, -0.000226, 4.211563]
traj_map = []
abs_pose = build_se3_transform(xyzrpy)
abs_poses = []
H_new = abs_pose
index = 0
vo_timestamps = []
# vo_map = cv2.circle(vo_map,start_map,5,(0,0,255),thickness=3)
obj_map = []

index = 0
# Reading vo
print('Loading visual odometry...')
with open(vo_directory) as vo_file:
    vo_reader = csv.reader(vo_file)
    headers = next(vo_file)

    for row in vo_reader:
        timestamp = int(row[0])
        vo_timestamps.append(timestamp)
        # datetime = dt.utcfromtimestamp(timestamp/1000000)
        xyzrpy = [float(v) for v in row[2:8]]
        rel_pose = build_se3_transform(xyzrpy)

        H_new=H_new@rel_pose    # wolrd-cord transform
        x_test=H_new[1,3]
        y_test=H_new[0,3]

        abs_poses.append(H_new)

        # plot trajectory
        x_map = (x_test + float(start_map[0]))
        y_map = (-y_test +  float(start_map[1]))

        traj_map.append((x_map,y_map))

        index = index + 1
        if index > 1000 :
            upper_timestamp = timestamp
            break
index = 0        
vo_file.close()

print("timestamp", len(vo_timestamps))
print("abs_poses", len(abs_poses))
print("traj_map", len(traj_map))

interpolate_timestamp = vo_timestamps

traj_dict = dict(zip(vo_timestamps, traj_map))
pose_dict = dict(zip(vo_timestamps, abs_poses))
traj_map = np.array(traj_map)
vo_timestamps = np.array(vo_timestamps)
n_traj = traj_map.shape[0]
# Read lidar extrnsics

with open(os.path.join(my_params.extrinsics_dir, 'ldmrs.txt')) as extrinsics_file:
    extrinsics = next(extrinsics_file)
G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])


# Read pointcloud 
print('Matching lidar pointcloud...')
with open(lidar_timestamps_path) as lidar_timestamps_file:
    for _,row in enumerate(lidar_timestamps_file):
        lidar_timestamps = int(row.split(' ')[0])
        requested_timestamps = []
        requested_timestamps.append(lidar_timestamps)
        for relative_timestamp in vo_timestamps:
            if lidar_timestamps > relative_timestamp:
                origin_timestamp = relative_timestamp

        poses = interpolate_poses(interpolate_timestamp, abs_poses, requested_timestamps, origin_timestamp)

        #Find closet pose of car
        # closet_timestamp = get_closest(pose_dict, lidar_timestamps)
        # print('Given', lidar_timestamps, ', closest:', closet_timestamp)

        file_path = os.path.join(lidar_folder_path + '\\' + str(lidar_timestamps) + '.bin')
        scan_file = open(file_path)
        objs_data = np.fromfile(scan_file, np.double)
        scan_file.close()
        objs_data = objs_data.reshape((len(objs_data) // 3, 3)).transpose()

        # Change one by one pixel  # CV2 image
        # for i in range(500):
        for i in range(objs_data.shape[1]):
            obj_data = [objs_data[0][i],objs_data[1][i],objs_data[2][i],0,0,0]        #xyzrpy
            # print(obj_data)
            obj_pose = build_se3_transform(obj_data)

            obj_pose=poses@obj_pose    # wolrd-cord transform
            x_obj=(float(start_map[0]) + float(obj_pose[1,3]))
            y_obj=(float(start_map[1]) - float(obj_pose[0,3]))

            obj_map.append((x_obj,y_obj))
        index = index + 1
        # if (index > 100): break
        if (lidar_timestamps > upper_timestamp): 
            print("stop at: ",index)
            break

lidar_timestamps_file.close()

obj_map = np.array(obj_map)
n_points = obj_map.shape[0]

traj_min_x = np.min(traj_map[:, 0])
traj_min_y = np.min(traj_map[:, 1])
traj_max_x = np.max(traj_map[:, 0])
traj_max_y = np.max(traj_map[:, 1])

obj_min_x = np.min(obj_map[:, 0])
obj_min_y = np.min(obj_map[:, 1])
obj_max_x = np.max(obj_map[:, 0])
obj_max_y = np.max(obj_map[:, 1])

grid_min_x = scale*min(traj_min_x, obj_min_x)
grid_min_y = scale*min(traj_min_y, obj_min_y)
grid_max_x = scale*max(traj_max_x, obj_max_x)
grid_max_y = scale*max(traj_max_y, obj_max_y)

grid_res = [int(grid_max_x - grid_min_x), int(grid_max_y - grid_min_y)]

print ('amplitude_x: ', grid_min_x, grid_max_x )
print ('amplitude_y: ', grid_min_y, grid_max_y)
print ('grid_res: ', grid_res)

print(traj_map.shape)
print(obj_map.shape)

visit_counter = np.zeros(grid_res, dtype=np.int32)
occupied_counter = np.zeros(grid_res, dtype=np.int32)

grid_cell_size_x = (grid_max_x - grid_min_x) / float(grid_res[0])
grid_cell_size_y = (grid_max_y - grid_min_y) / float(grid_res[1])

norm_factor_x = float(grid_res[0] - 1) / float(grid_max_x - grid_min_x)
norm_factor_y = float(grid_res[1] - 1) / float(grid_max_y - grid_min_y)
print ('norm_factor_x: ', norm_factor_x)
print ('norm_factor_z: ', norm_factor_y)


for point_id in range(n_points):
    point_x = int((scale*obj_map[point_id,0]- grid_min_x) * norm_factor_x)
    point_y = int((scale*obj_map[point_id,1]- grid_min_y) * norm_factor_y)

    visit_counter[point_x, point_y] = 255

grid_map = np.zeros(grid_res, dtype=np.float32)
grid_map_thresh = np.zeros(grid_res, dtype=np.uint8)

for x_p in range(grid_res[0]):
    for y_p in range(grid_res[1]):
        if (visit_counter[x_p, y_p] == 0 ):
            grid_map_thresh[x_p, y_p] = 0
        if (visit_counter[x_p, y_p] > 0 ):
            grid_map_thresh[x_p, y_p] = 255

plt.xticks(size=12,color = "black")
plt.yticks(size=12,color = "black")
plt.imshow(grid_map_thresh,cmap="plasma")       # True = yellow
plt.show()


print("Done!!")




