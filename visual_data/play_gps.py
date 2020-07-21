# import trackanimation
# from trackanimation.animation import AnimationTrack

# input_directory = 'D:/GoogleDrive/Data/20140514/gps/gps.csv'
# ibiza_trk = trackanimation.read_track(input_directory)
# ibiza_trk = ibiza_trk.time_video_normalize(time=10, framerate=10)
# ibiza_trk = ibiza_trk.set_colors('Speed', individual_tracks=True)

# fig = AnimationTrack(df_points=ibiza_trk, dpi=300, bg_map=True, map_transparency=0.5)
# fig.make_video(output_file='coloring-map-by-speed', framerate=10, linewidth=1.0)

# # Variable 'bg_map' must be to False in order to create an interactive map
# fig = AnimationTrack(df_points=ibiza_trk, dpi=300, bg_map=False, map_transparency=0.5)
# fig.make_map(output_file='coloring-map-by-speed')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data 
input_directory = 'D:/GoogleDrive/Data/20140514/gps/ins.csv'

data = pd.read_csv(input_directory) 

# Preview the first 5 lines of the loaded data 
data.head()

# Get size
BBox = (data.longitude.min(), data.longitude.max(), 
        data.latitude.min(), data.latitude.max())

# image from opestreetmap.org
ruh_m = plt.imread('D:/Workspace/Github/robotcar-dataset/img/maps.png')

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(data.longitude, data.latitude, zorder=1, alpha= 0.2, c='b', s=10)
ax.set_title('Plotting Spatial Data on Riyadh Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')

plt.show()