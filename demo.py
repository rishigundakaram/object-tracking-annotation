import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tempfile import NamedTemporaryFile

from models.tracker import SiameseTracker, run_tracker
from src.data.utils import preprocess_video, combine_bb_frames_into_video
from models.siamfc import TrackerSiamFC
from models.siamrpn import TrackerSiamRPN

st.title('Object Tracking Anotation App')
# tracker = SiameseTracker()
# tracker = TrackerSiamFC(net_path='./weights/siamFC.pth')
tracker = TrackerSiamRPN(net_path='./weights/siamrpn.pth')
uploaded_video = st.file_uploader('Upload a Video')

if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    vidcap = cv2.VideoCapture(vid) # load video from disk
    
    frames = preprocess_video(vidcap)
    canvas_result = st_canvas(
        fill_color="#21FF0000",
        stroke_color="#21FF00",
        stroke_width=3,
        background_image=frames[0],
        drawing_mode='rect',
        key="color_annotation_app",
    )
    st.write("Draw a bounding box around the object of interest!")
    if len(canvas_result.json_data["objects"]) > 0:  
        init_box = [canvas_result.json_data["objects"][0][i] for i in ["left", "top", "width", "height"]]
        bboxes = run_tracker(tracker, frames, init_box)
        # file = 'video.webm'
        file = NamedTemporaryFile(suffix='.webm')
        combine_bb_frames_into_video(frames, bboxes, 24, file.name)
        print('here')
        st.video(file.read(), "video/webm")
        
    
