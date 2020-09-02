import os
import re
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import my_params
import numpy.matlib as ml

from datetime import datetime as dt
from transform import build_se3_transform, so3_to_quaternion
from interpolate_poses import interpolate_poses




if __name__ == '__main__':
    image_dir = my_params.image_dir
    vo_directory = my_params.dataset_patch + 'vo//vo.csv'

    map_ldrms_dir = my_params.output_dir2 + 'map_ldrms_' + my_params.dataset_no + '\\'
    keyframe_image_dir = my_params.output_dir2 +'keyframe_'+ my_params.dataset_no + '\\'

    camera = re.search('(stereo|mono_(left|right|rear))', image_dir).group(0)

    timestamps_path = os.path.join(os.path.join(image_dir, os.pardir, os.pardir, camera + '.timestamps'))
    timestamps_file = open(timestamps_path)

##############################
    abs_poses = [ml.identity(4)]
    abs_pose = build_se3_transform(my_params.xyzrpy)
    abs_poses.append(abs_pose)
    vo_timestamps = [0]
    with open(vo_directory) as vo_file:
        vo_reader = csv.reader(vo_file)
        headers = next(vo_file)
        index = 0
        for row in vo_reader:
            timestamp = int(row[0])
            vo_timestamps.append(timestamp)
            # datetime = dt.utcfromtimestamp(timestamp/1000000)
        
            index += 1
            if index > 3000 : 
                end_time = timestamp
                break
    
            xyzrpy = [float(v) for v in row[2:8]]
            rel_pose = build_se3_transform(xyzrpy)
            abs_pose = abs_poses[-1] * rel_pose
            abs_poses.append(abs_pose)
    end_time = timestamp
    vo_file.close()
    vo_timestamps = np.array(vo_timestamps)
######################################

    # # Read lidar extrnsics
    with open(os.path.join(my_params.extrinsics_dir, 'ldmrs.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

###################################3
    current_chunk = 0 
    idx = 0
    requested_timestamps = []
    for line in timestamps_file:
        tokens = line.split()
        timestamp = int(tokens[0])
        idx +=1
        if(idx == 1):
            start_time = timestamp
            continue

        if(timestamp > end_time):
            break
        requested_timestamps.append(timestamp)
    timestamps_file.close()

    print(len(requested_timestamps))


    # poses = interpolate_poses(vo_timestamps, abs_poses, requested_timestamps, vo_timestamps[0])
    # print('pose',len(poses))
        
    timestamps_file = open(timestamps_path)
    idx = 0
    plt.figure()
    for line in timestamps_file:

        tokens = line.split()
        timestamp = int(tokens[0])
        idx +=1

        if(idx <= 15):
            continue
        if(timestamp > end_time):
            break
        
        ldrms_filename = os.path.join(map_ldrms_dir + tokens[0] + '.csv')
        print(ldrms_filename)
        ldrms_file = cv2.imread(ldrms_filename)

        lidar_points=np.genfromtxt(ldrms_filename,delimiter = ',')
        lidar_points = np.array(lidar_points)
        # print(lidar_points.shape)
        pose = np.dot(abs_pose[idx],G_posesource_laser)
        xyz1 = np.vstack((lidar_points,np.ones(lidar_points.shape[1])))
        new_xyz1 = pose*xyz1
        print(new_xyz1.shape)
        x_obj = np.array(new_xyz1[0,:])
        y_obj = np.array(new_xyz1[1,:])
        plt.scatter(-x_obj,y_obj,c='r',marker='.', zorder=0)

        if (idx > 1000): break
    plt.show()
'''

for i in range(num_of_laser):
    # if i > 500: break
    if lidar_timestamps[i] >= end_time: break

    # pose = poses[i]
    pose = np.dot(poses[i],G_posesource_laser)
    # print('pose',pose)
    # abs_quaternions = so3_to_quaternion(pose[0:3, 0:3])

    file_path = os.path.join(lidar_folder_path + '\\' + str(lidar_timestamps[i]) + '.bin')
    scan_file = open(file_path)
    objs_data = np.fromfile(scan_file, np.double)
    scan_file.close()

    xyz = objs_data.reshape((len(objs_data) // 3, 3)).transpose()
    xyz = np.array(xyz)

    new_xyz1 = pose*xyz1
    x_obj = np.array(new_xyz1[0,:])
    y_obj = np.array(new_xyz1[1,:])
    plt.scatter(-x_obj,y_obj,c='r',marker='.', zorder=0)

    obj_on_map.append((new_xyz1[0:1,:]))

    # save_to_csv = []
    save_csv = np.vstack((x_obj,y_obj))
    # np.savetxt((output_dir + str(lidar_timestamps[i]) + '.csv'), save_csv.transpose(), delimiter=",")
    
    
    plt.show()

        #         
        # if timestamp[idx] >= end_time: break

        # # pose = poses[i]
        # pose = np.dot(pose[idx],G_posesource_laser)
        
        # new_xyz1 = pose*xyz1
        # x_obj = np.array(new_xyz1[0,:])
        # y_obj = np.array(new_xyz1[1,:])
'''


