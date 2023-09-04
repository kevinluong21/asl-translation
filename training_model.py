import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization, Reshape
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import cv2

def format_frames(frame, output_size) {
    #format a single frame to the specified output size with padding
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame
}

def frames_from_video(video_path, num_frames, output_size=(224, 224), frame_step = 15) {
    
}