

project_patch = 'D:\\Untitled\\final_project_sdc-master\\'
dataset_patch = 'D:\\data\\'

# project_patch = "D:\\Github\\final_project_sdc\\"
# dataset_patch = "D:\\Dataset\\20140514\\"


poses_file = dataset_patch + 'vo\\vo.csv'
extrinsics_dir = project_patch + 'extrinsics'
laser_dir = dataset_patch + 'ldmrs'

camera = 'stereo\\centre'
camera_name = 'stereo'

image_dir = dataset_patch + camera
model_dir = project_patch + 'models'

model = model_dir + camera_name

reprocess_image_dir = 'D:\\data\\reprocessed\\img\\' + camera_name
