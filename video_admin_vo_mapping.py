import os
import re
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import my_params
from datetime import datetime as dt

DRAW_LASER  = 1
SAVE_VIDEO  = 0
USER_VO     = 0
RESIZE  = 1


if (RESIZE == 1):
    dim = (320,240)
if (RESIZE == 2):
    dim = (640,480)

output_points_patch = my_params.output_vo_points_admin
lidar_timestamps_path = my_params.dataset_patch + 'ldmrs.timestamps'
lidar_dir = my_params.backup_dir+'\\lidar_'+ my_params.dataset_no + '\\'

# pointcloud_img_dir = my_params.backup_dir+'\\pointcloud_img_'+ my_params.dataset_no + '\\'
keyframe_image_dir = my_params.output_dir2 +'keyframe_'+ my_params.dataset_no + '\\'

pointcloud_img_dir = my_params.output_dir2 + 'pl_img_'   + my_params.dataset_no + '\\'
yolo_img_dir = my_params.output_dir2 + 'yolo_img_'   + my_params.dataset_no + '\\'
yolo_obj_dir = my_params.output_dir2 + 'yolo_obj_'   + my_params.dataset_no + '\\'


if __name__ == '__main__':
    # Making save
    if (SAVE_VIDEO):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vw = cv2.VideoWriter('video\\visual_odometry.mp4', fourcc, 15, (640, 360))
    
    # Read trajectory data
    final_points=[]         # positions list
    points=np.genfromtxt(output_points_patch,delimiter = ',')
    for point in points:
        final_points.append((point[0],point[1]))

    # Read image data
    image_dir = my_params.image_dir
    camera = re.search('(stereo|mono_(left|right|rear))', image_dir).group(0)
    
    timestamps_path = os.path.join(os.path.join(image_dir, os.pardir, os.pardir, camera + '.timestamps'))
    timestamps_file = open(timestamps_path)
    
    current_chunk = 0
    index = 0
    lidar_index = 0; draw_lidar = 4
    lidar_timestamps = []

    plt.figure('Maps')
    # plt.title('Plot maps')
    with open(lidar_timestamps_path) as lidar_timestamps_file:
        num_of_laser = 0
        for _,row in enumerate(lidar_timestamps_file):
            lidar_timestamp = int(row.split(' ')[0])
            # if (lidar_timestamp > end_time): 
            #     print("num_of_laser_scan: ",num_of_laser)
            #     break
            # requested_timestamps.append(lidar_timestamp)
            lidar_timestamps.append(lidar_timestamp)
            num_of_laser += 1
    lidar_timestamps_file.close()
    lidar_timestamps = np.array(lidar_timestamps)

    for line in timestamps_file:
        tokens = line.split()
        datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
        chunk = int(tokens[1])

    # keyframe image
        keyframe = os.path.join(keyframe_image_dir + '//' + tokens[0] + '.png')
        if not os.path.isfile(keyframe):
            if chunk != current_chunk:
                print("Chunk " + str(chunk) + " not found")
                current_chunk = chunk
            continue

        frame = cv2.imread(keyframe)
        frame = cv2.resize(frame,dim)
        cv2.imshow('Capture',frame)
    
    # pointcloud image   
        # pl_filename = os.path.join(pointcloud_img_dir + '//' + tokens[0] + '.png')
        # pl_img = cv2.imread(pl_filename)
        # pl_img = cv2.resize(pl_img,dim)
        # cv2.imshow('pointcloud',pl_img)
    # yolo image 
        yolo_filename = os.path.join(yolo_img_dir + '//' + tokens[0] + '.png')
        yolo_img = cv2.imread(yolo_filename)
        yolo_img = cv2.resize(yolo_img,dim)
        cv2.imshow('Object tracking',yolo_img)

    # yolo object   
        object_filename = os.path.join(yolo_obj_dir + '//' + tokens[0] + '.png')
        object_img = cv2.imread(object_filename)
        # object_img = cv2.resize(object_img,dim)
        cv2.imshow('Object position',object_img)


        index = index + 1
        current_chunk = chunk

        # DRAW MAPS
        # Plot trajectory in plt
        plt.plot(final_points[index][0],final_points[index][1],'.',color='red')
        plt.pause(0.01)

        draw_lidar -=1
        if (draw_lidar == 0):
            draw_lidar = 4
            lidar_index +=1
            print(lidar_index)
            # print('lidar_index',lidar_index)
            lidar_path = os.path.join(lidar_dir + str(lidar_timestamps[lidar_index]) + '.csv')
            print('lidar_path',lidar_path)
            obj_points=[]         # positions list
            lidar_points=np.genfromtxt(lidar_path,delimiter = ',')
            for point in lidar_points:
                obj_points.append((point[0],point[1]))
            obj_points = np.array(obj_points)
            print("ponts",obj_points.shape)
            plt.plot(-obj_points[:,0],obj_points[:,1],'.',color='blue')
            # plt.pause(0.01)

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break

    print('done!')
    cv2.destroyAllWindows()