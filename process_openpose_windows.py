# please download the openpose windows portable demo and place the openpose folder in the same folder as this file. 
# go into the directory that contains this file and run this script, making# sure to change the paths where the videos are located to the correct disc. 
# the program will go into the openpose folder for you.

# the root directory should look like the following:
# openpose
# videos
# pose_data
# WLASL_v0.3.json

import json
import os
import subprocess

content = json.load(open('WLASL_v0.3.json'))

count = 0
total = 17295

for entry in content:
    instances = entry['instances']

    for inst in instances:
        video_id = inst['video_id']

        src_path = "videos/" + str(video_id) + '.mp4' #the filepath of the original video
        save_path = "pose_data/" + str(video_id) #the filepath of the openpose video

        #videos can only be saved as .avi files on windows
        if os.path.exists(save_path + ".mp4") or os.path.exists(save_path + ".avi"):
          count += 1
        elif os.path.exists(src_path):
          count += 1
          #change the paths to the correct port and disc
          command = 'cd openpose && bin\OpenPoseDemo.exe --video "D:/Projects/openpose-processing/' + src_path + '" --display 0 --hand --face --disable_blending --write_video "D:/Projects/openpose-processing/' + save_path + '.avi"'
          os.system(command)

        print("Processed " + src_path + "(" + str(count) + "/" + str(total) + " or " + str(count/total) + "%)")
        print("")
