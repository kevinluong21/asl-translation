import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization, Reshape
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import cv2
import json
import random
from pathlib import Path

#imports to visualize videos as gifs
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed

def format_frames(frame, output_size):
    #format a single frame to the specified output size with padding
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video(video_path, output_size=(224, 224)):
    result = [] #returns an array of number of frames, height, width, and number of channels 
    src = cv2.VideoCapture(str(video_path)) #open video from path
    num_frames = src.get(cv2.CAP_PROP_FRAME_COUNT) #get the number of frames from the video
    
    for _ in range(int(num_frames)):
        ret, frame = src.read() #ret is a boolean that indicates whether the read was a success

        if ret:
            result.append(format_frames(frame, output_size)) #if read succeeds, format the frame, and append to result
        else:
            result.append(np.zeros_like(result[0])) #if read fails, leave the frame as a numpy array of 0s

    src.release()
    result = np.array(result)[..., [2, 1, 0]] #opencv flips colour channels to BGR, so flip them back to RGB

    return result

class FrameGenerator: #the generator that feeds the input videos into tf
    #process class names from json file
    file = open("WLASL_v0.3.json") #returns json file as a dict
    classes = json.load(file)

    def __init__(self, path):
        #path defines the paths of all the videos that will be input
        path = str(path)

        self.path = Path(path)
        self.class_names = []
        self.class_ids = {} #stores the class name and its id

        for entry in self.classes:
            self.class_names.append(entry['gloss'])

        #create a dictionary where each class name is assigned a corresponding id
        for class_id, class_name in enumerate(self.class_names):
            self.class_ids[class_name] = class_id

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('**/*')) #make a list of all files in the directory
        classes = [] #stores a parallel list that contains the class of the corresponding video path

        for path in video_paths:
            #isolate the video id
            path = str(path)
            video_id = path.replace("pose_data/", "")
            video_id = video_id.replace(".mp4", "") #assuming that all videos are mp4 videos

            for i in range(len(self.classes)):
                for j in range(len(self.classes[i]["instances"])):
                    if self.classes[i]["instances"][j]["video_id"] == video_id:
                        classes.append(self.classes[i]['gloss'])

        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes)) #iterates through video_paths and classes and creates a tuple each time before creating a list of tuples

        random.shuffle(pairs) #shuffle the pairs for training

        for path, name in pairs:
            frames = frames_from_video(path)
            label = self.class_ids[name] #the label is the corresponding id of the class name
            yield frames, label #yield keyword used in generator functions


