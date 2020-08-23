import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import my_params

# import random
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
# from math import *

# Play all reprocessed Image
def play_processed_img():

    # Get image    
    image_dir = my_params.image_dir
    model_dir = my_params.model_dir
    reprocess_dir = my_params.reprocess_image_dir
    camera_name = my_params.camera_name
    scale = 0.5

    frames = 0
    start = time.time() 
    timestamps_path = os.path.join(os.path.join(image_dir, os.pardir, os.pardir, camera_name + '.timestamps'))

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
        img = cv2.imread(filename)
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        cv2.imshow("img",img)
        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break

        frames += 1
        print(datetime)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
##############################

# Function Plot INS/GPS data into map
def play_gps():
    # Read data 
    input_INS = my_params.dataset_patch + 'gps//ins.csv'
    input_GPS = my_params.dataset_patch + 'gps//gps.csv'

    ins = pd.read_csv(input_INS) 
    ins.head()

    gps = pd.read_csv(input_GPS) 
    gps.head()

    # Get size
    # BBox = (gps.longitude.min(), gps.longitude.max(), 
    #         gps.latitude.min(),  gps.latitude.max())
    BBox = (gps.easting.min(), gps.easting.max(), 
            gps.northing.min(),  gps.northing.max())


    # image from opestreetmap.org
    ruh_m = plt.imread(my_params.project_patch + 'rtk\\' + my_params.dataset_no + '.png')

    # Plot INS 'red'
    fig, ax = plt.subplots()
    # ax.scatter(ins.longitude, ins.latitude, zorder=1, alpha= 0.2, c='r',marker='.', s=10)
    ax.scatter(ins.easting, ins.northing, zorder=1, alpha= 0.2, c='r',marker='.')
    ax.set_title('Plotting INS Map')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')

    # Plot GPS 'blue'
    fig2, ax2 = plt.subplots()
    # ax2.scatter(gps.longitude, gps.latitude, zorder=1, alpha= 0.2, c='b',marker='.', s=10)
    ax2.scatter(gps.easting, gps.northing, zorder=1, alpha= 0.2, c='b',marker='.')
    ax2.set_title('Plotting GPS Map')
    ax2.set_xlim(BBox[0],BBox[1])
    ax2.set_ylim(BBox[2],BBox[3])
    ax2.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')

    plt.show()
##############################


# Play Lidar Pointcloud 
def play_lidar():
    # Make save
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # video = cv2.VideoWriter('laser_scan.mp4', fourcc, 30, (800, 600))

    # Load data
    timestamps_path = my_params.dataset_patch + 'ldmrs.timestamps'
    lidar_folder_path = my_params.laser_dir

    timestamp = 0
    with open(timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            # if (i > 100): break

            timestamp = int(line.split(' ')[0])

            datetime = dt.utcfromtimestamp(timestamp/1000000)
            print(datetime)

            file_path = os.path.join(lidar_folder_path + '\\' + str(timestamp) + '.bin')
            scan_file = open(file_path)

            data = np.fromfile(scan_file, np.double)
            scan_file.close()
            data = data.reshape((len(data) // 3, 3)).transpose()

            # Create a color
            color = [255,0,0]

            # Blank image
            img = np.zeros((600,800,3),np.uint8)

            # Change one by one pixel
            for i in range(data.shape[1]):
                x = 5*data[0][i]
                y = 5*data[1][i]
                if (x > 599) : x = 599
                if (x < 1) : x = 1
                if (y < -400) : y = -399 
                if (y > 400) : y = 399 
                img[600-(int(x)),(int(y)) + 400]=color
            # Save
            cv2.imshow("result",img)
            # video.write(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # video.release()
    cv2.destroyAllWindows()
#######################


# Play Visual Odometry data
def play_vo():
    import numpy.matlib as ml
    import csv
    from interpolate_poses import interpolate_poses
    from transform import build_se3_transform

    # Read data
    vo_directory = my_params.dataset_patch + 'vo\\vo.csv'


    # fix start point   
    # abs_pose = [ml.identity(4)]

    abs_pose = build_se3_transform(my_params.xyzrpy)
    abs_poses = [abs_pose]
    with open(vo_directory) as vo_file:
        vo_reader = csv.reader(vo_file)
        headers = next(vo_file)

        index = 0
        for row in vo_reader:
            timestamp = int(row[0])
            datetime = dt.utcfromtimestamp(timestamp/1000000)
            
            xyzrpy = [float(v) for v in row[2:8]]
            rel_pose = build_se3_transform(xyzrpy)
            abs_pose = abs_poses[-1] * rel_pose
            abs_poses.append(abs_pose)

            index += 1
            # if index > 3000 : break
    vo_file.close()

    # abs_quaternions = np.zeros((4, len(abs_poses)))
    abs_positions = np.zeros((3, len(abs_poses)))

    print('num of point:', len(abs_poses))

    # plt.figure()
    # plt.gca().set_aspect('equal', adjustable='box')
    fig, ax = plt.subplots()

    # image from opestreetmap.org
    ruh_m = plt.imread(my_params.project_patch + 'rtk\\' + my_params.dataset_no + '.png')

    points = []
    for i, pose in enumerate(abs_poses):
        # abs_quaternions[:, i] = so3_to_quaternion(pose[0:3, 0:3])
        abs_positions[:, i] = np.ravel(pose[0:3, 3])
        point = (-abs_positions[0,i],abs_positions[1,i])        # -x, y of point
        # plt.plot(point[0],point[1],'.',color='red')
        # plt.scatter(point[0],point[1],c='b',marker='.', zorder=1)
        ax.scatter(point[0],point[1], zorder=1, alpha= 0.2, c='b',marker='.')
        points.append(point)

    points = np.array(points)
    # print(points.shape)

    BBox = (np.min(points[:,0]), np.max(points[:,0]), 
            np.min(points[:,1]), np.max(points[:,1]))
    
    ax.set_title('Plotting VO Map')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
    plt.show()


##################################

def map_extract_from_ins():
    print('pose_extract_from_ins')

    import numpy.matlib as ml
    import csv
    import re
    import cv2
    import time
    from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
    from transform import build_se3_transform
    from camera_model import CameraModel

    # Read data 
    input_INS = my_params.dataset_patch + 'gps//ins.csv'
    input_GPS = my_params.dataset_patch + 'gps//gps.csv'

    H_new=np.identity(4)
    abs_poses = [ml.identity(4)]
    ins_timestamps = [0]
    use_rtk=False

    x_old=(0,0)
    z_old=(0,0)
    fig = plt.figure()
    gs = plt.GridSpec(2,3)

    # first =  ['5736385.159098', '620163.607591', '-112.107004', '-0.090749', '-0.000226', '4.211563']
    startpoint = [620163.607591,5736385.159098]

    with open(input_INS) as ins_file:
        ins_reader = csv.reader(ins_file)
        headers = next(ins_file)

        # upper_timestamp = max(max(pose_timestamps), origin_timestamp)
        index = 0
        for row in ins_reader:
            timestamp = int(row[0])
            ins_timestamps.append(timestamp)

            # utm = row[5:8]
            # rpy = row[12:15] 
            # xyzrpy = [float(v) for v in utm] + [float(v) for v in rpy]
            # for i in range(len(xyzrpy)):
            #     xyzrpy[i] =  xyzrpy[i] - float(first[i])
            # plt.plot(xyzrpy[1], xyzrpy[0],'o',color='blue')
            # plt.pause(0.01)  

            x = float(row[6]) - float(startpoint[0])
            y = float(row[5]) - float(startpoint[1])
            if (index<1000):
                plt.plot(x,y,'.',color='red')
            else:
                plt.plot(x,y,'.',color='blue')
            index = index + 1
    
    plt.show()
    
    cv2.waitKey(0)
    ins_timestamps = ins_timestamps[1:]
    abs_poses = abs_poses[1:]
           

if __name__ == "__main__":
    print("Start")

    play_vo()

    print("Done!")
