import os
import re
import cv2
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import my_params
from datetime import datetime as dt

output_image_dir = my_params.backup_dir+'\\'+ my_params.dataset_no + '\\'
output_points_patch = my_params.backup_dir+'\\'+ my_params.dataset_no + '_vo_points.csv'

# if __name__ == '__main__':
    
#     # Round truth
#     # rtk_directory = my_params.project_patch + 'gt\\gt.csv'
#     rtk_directory = my_params.dataset_patch + 'rtk\\rtk.csv'
#     rtk = pd.read_csv(rtk_directory) 
#     rtk.head()

#     # Get size
#     BBox = (rtk.easting.min(), rtk.easting.max(), 
#             rtk.northing.min(),  rtk.northing.max())

#     print(BBox)

#     # image from opestreetmap.org
#     ruh_m = plt.imread(my_params.project_patch + 'rtk\\rtk.png')

#     fig, ax = plt.subplots()
#     ax.scatter(rtk.easting, rtk.northing, zorder=1, alpha= 0.2, c='b',marker='.', s=10)
#     ax.set_title('Mapping 2015-10-30-11-56-36')
#     ax.set_xlim(BBox[0],BBox[1])
#     ax.set_ylim(BBox[2],BBox[3])
#     ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')

#     plt.show()

# 5734795.685425566, 620021.4778138875

if __name__ == '__main__':
    # # Making save
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # vw = cv2.VideoWriter('output\\visual_odometry.mp4', fourcc, 15, (640, 360))
    
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

# ############################3    # Round truth
#     rtk_directory = my_params.dataset_patch + 'rtk\\rtk.csv'
#     rtk = pd.read_csv(rtk_directory) 
#     rtk.head()

#     # Get size
#     BBox = (rtk.easting.min(), rtk.easting.max(), 
#             rtk.northing.min(),  rtk.northing.max())

#     print(BBox)

#     # image from opestreetmap.org
#     ruh_m = plt.imread(my_params.project_patch + 'rtk\\rtk.png')

#     fig, ax = plt.subplots()
#     ax.scatter(rtk.easting, rtk.northing, zorder=1, alpha= 0.2, c='b',marker='.', s=10)
#     ax.set_title('Mapping 2015-10-30-11-56-36')
#     ax.set_xlim(BBox[0],BBox[1])
#     ax.set_ylim(BBox[2],BBox[3])
#     ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')

#     y_0 = 5734795.685425566
#     x_0 = 620021.4778138875
# ##################################

    for line in timestamps_file:
        tokens = line.split()
        datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
        chunk = int(tokens[1])

        filename = os.path.join(output_image_dir + '//' + tokens[0] + '.png')
        if not os.path.isfile(filename):
            if chunk != current_chunk:
                print("Chunk " + str(chunk) + " not found")
                current_chunk = chunk
            continue
        current_chunk = chunk

        frame = cv2.imread(filename)
        cv2.imshow('camera',frame)
        # vw.write(frame)

        # Plot trajectory in plt
        plt.plot(final_points[index][0],-final_points[index][1],'.',color='red')
        plt.pause(0.01)
        index = index + 1

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break

    print('done!')
    cv2.destroyAllWindows()