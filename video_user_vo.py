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

if __name__ == '__main__':
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

        # Plot trajectory in plt
        plt.plot(final_points[index][0],-final_points[index][1],'.',color='red')
        plt.pause(0.01)
        index = index + 1

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break

    print('done!')
    cv2.destroyAllWindows()