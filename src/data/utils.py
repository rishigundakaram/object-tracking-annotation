import cv2
import numpy as np
from PIL import Image
import streamlit as st

def resize_with_pad(image: np.array, 
                    new_shape, 
                    padding_color=(255, 255, 255)):
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

@st.cache(allow_output_mutation=True)

def preprocess_video(video): 
    """Turns cv2 VideoCapture object into array of processed PIL images

    Args:
        video (cv2.VideoCapture): video that needs to be processed
    Returns: 
        frames (array of PIL.Image): processed frames
    """
    frames = []
    success, frame = video.read() 
    while success: 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_with_pad(frame, (600, 400))
        frames.append(Image.fromarray(frame))
        success, frame = video.read() 
    return frames

def combine_bb_frames_into_video(frames, boxes, fps, filename): 
    color = (0, 255, 0)
    thickness = 3
    width, height, _ = np.shape(frames[0])
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    video = cv2.VideoWriter(filename, fourcc, fps, (height,width))
    for idx in range(len(frames)): 
        cv_frame = cv2.cvtColor(np.array(frames[idx]), cv2.COLOR_RGB2BGR)
        start = (boxes[idx][0], boxes[idx][1])
        end = (boxes[idx][0] + boxes[idx][2], boxes[idx][1] + boxes[idx][3])
        cv_frame = cv2.rectangle(cv_frame, start, end, color, thickness)
        video.write(cv_frame)
    video.release()