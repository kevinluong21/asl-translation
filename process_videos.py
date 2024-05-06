#Source code from https://github.com/nicknochnack/Full-Body-Estimation-using-Media-Pipe-Holistic
#The code was modified for this project.

#NOTE: something is causing a context leak???

import os
import mediapipe as mp
import cv2
import numpy as np
import json
from pathlib import Path

def landmarks_to_dict(landmarks):
    '''
    Return a dictionary of landmarks where the index of each landmark is a number from 0 to n where n is the number of landmarks.
    Each index contains another dictionary with x, y, and z keys.

    Parameters:
        landmarks (): A list of landmarks pertaining to a specific body part.
    Returns:
        landmarks_dict (dict): A dictionary of landmarks.
    '''
    landmarks_dict = {}
    #landmarks may be None if the body part does not appear in the video
    #if landmarks is None, then it will return an empty dict
    if landmarks is not None:
        i = 0
        for landmark in landmarks.landmark:
            landmarks_dict[i] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            }
            i += 1

    return landmarks_dict

src_dir = "/Volumes/ESD-USB/Projects/openpose-processing/videos"
save_dir = "pose_json"

src_path = Path(src_dir)
save_path = Path(save_dir)

if not os.path.exists(save_path):
    os.mkdir(save_path)

videos = list(src_path.glob('*.mp4')) #make a list of all mp4 videos in the directory
saved_videos = list(save_path.glob('*.json')) #make a list of all JSON files in the directory

#the total number of videos to process
num_videos = len(videos)
#the number of videos that were already processed and saved
processed = len(saved_videos)

# mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)

#Print the number of videos already processed before processing resumes
print("Processed " + str(processed) + "/" + str(num_videos) + " videos (" + str(round((processed/num_videos) * 100, 3)) + "%)")

for video in videos:
    video_id = Path(video).stem

    #If the video was already processed and saved in the save folder, then skip to the next file
    if os.path.exists(save_dir + "/" + video_id + ".json"):
        continue
    
    cap = cv2.VideoCapture(video.absolute().as_posix())

    #TODO: Check first that the connection was successful

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter(filename="pose_json/" + file, fourcc=fourcc, fps=30.0, frameSize=(width, height))

    frame_number = 0
    frames = {}

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()

            #If the video is not read or the end of the video is reached, ret will be False
            if ret is False:
                break

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)

            frames[frame_number] = {
                "right_hand": landmarks_to_dict(results.right_hand_landmarks),
                "left_hand": landmarks_to_dict(results.left_hand_landmarks),
                "pose": landmarks_to_dict(results.pose_landmarks)
            }
            
            frame_number += 1

            # # Recolor image back to BGR for rendering
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # #Create an empty frame so that the video is not displayed in the background
            # landmarks = np.zeros((height, width, 3), dtype=np.uint8)
            
            # # Draw face landmarks
            # # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            
            # # Right hand
            # mp_drawing.draw_landmarks(landmarks, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # # Left Hand
            # mp_drawing.draw_landmarks(landmarks, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # # Pose Detections
            # mp_drawing.draw_landmarks(landmarks, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # # #Export the frames to a video
            # # out.write(landmarks)

            # cv2.imshow("Test", landmarks)

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

    #Write all landmarks to JSON file
    out = open(file=save_dir + "/" + video_id + ".json", mode="w")
    json.dump(frames, out, indent=6)

    cap.release()
    # cv2.destroyAllWindows()

    #After a video is done processing, increment the number of processed videos
    processed += 1
    #Print the number of videos that were processed after this iteration if it was successful
    print("Processed " + str(processed) + "/" + str(num_videos) + " videos (" + str(round((processed/num_videos) * 100, 3)) + "%)")