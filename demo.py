import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tempfile import NamedTemporaryFile

from models.tracker import SiameseTracker, run_tracker
from src.data.utils import preprocess_video, combine_bb_frames_into_video

st.title('Object Tracking Anotation App')
tracker = SiameseTracker()
uploaded_video = st.file_uploader('Upload a Video')

if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    vidcap = cv2.VideoCapture(vid) # load video from disk
    print('preprocessing')
    frames = preprocess_video(vidcap)
    print('done')
    canvas_result = st_canvas(
        fill_color="#21FF0000",
        stroke_color="#21FF00",
        stroke_width=3,
        background_image=frames[0],
        drawing_mode='rect',
        key="color_annotation_app",
    )
    
    if len(canvas_result.json_data["objects"]) > 0: 
        init_box = [canvas_result.json_data["objects"][0][i] for i in ["left", "top", "width", "height"]]
        bboxes = run_tracker(tracker, frames, init_box)
        file = NamedTemporaryFile(suffix='.webm')
        combine_bb_frames_into_video(frames, bboxes, 24, file.name)
        st.video(file.read(), "video/webm")
        
    
