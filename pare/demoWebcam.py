#Modified by Augmented Startups 2021
#Face Landmark User Interface with StreamLit
#Watch Computer Vision Tutorials at www.augmentedstartups.info/YouTube
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image, ImageSequence
import os
import cv2
import copy
import torch
import joblib
import colorsys
from tqdm import tqdm
from loguru import logger
from yolov3.yolo import YOLOv3
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from ..models import PARE, HMR
from .config import update_hparams
from ..utils.kp_utils import convert_kps
from ..smplify.run import smplify_runner
from ..utils.vibe_renderer import Renderer
from ..utils.pose_tracker import run_posetracker
from ..utils.train_utils import load_pretrained_model
from ..dataset.inference import Inference, ImageFolder
from ..utils.smooth_pose import smooth_pose
from ..utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    convert_crop_coords_to_orig_img,
    prepare_rendering_results,
)
from ..utils.vibe_image_utils import get_single_image_crop_demo
from ..utils.geometry import convert_weak_perspective_to_perspective

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

TEACHER_DEMO_VIDEO = r'.\video\Yoga 2.mp4'
STUDENT_DEMO_VIDEO = r'.\video\Yoga 1.mp4'
DEMO_IMAGE = 'demo.jpg'

st.title('3D Human Pose Comparison in SMPL format using PARE')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('3D Human Pose Comparison in SMPL format using PARE')
st.sidebar.subheader('Parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

st.markdown('In this application, we are using **PART - Part Attention Regressor for 3D Human Body Estimation [ICCV 2021]** for creating Body Mesh and **Dynamic Time Warping** for comparing poses. **StreamLit** is to create the Web Graphical User Interface (GUI). ')
st.markdown(
"""
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 400px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    width: 400px;
    margin-left: -400px;
}
</style>
""",
unsafe_allow_html=True,
)

st.set_option('deprecation.showfileUploaderEncoding', False)

use_webcam = st.sidebar.button('Use Webcam')
record = st.sidebar.checkbox("Record Video")
if record:
    st.checkbox("Recording", value=True)

st.sidebar.markdown('---')
st.markdown(
"""
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 400px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    width: 400px;
    margin-left: -400px;
}
</style>
""",
unsafe_allow_html=True,
    )
# max faces
fps = st.sidebar.number_input('Number of frames used for comparison', value = 30, min_value=2, max_value=50)
i = 0

st.sidebar.markdown('---')

st.markdown(' ## Output')

stframe = st.empty() # Working with one element at a time
teacher_video_file_buffer = st.sidebar.file_uploader("Please upload teacher video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
teacher_tfflie = tempfile.NamedTemporaryFile(delete=False)
if not teacher_video_file_buffer:
    teacher_vid = cv2.VideoCapture(TEACHER_DEMO_VIDEO)
    teacher_tfflie.name = TEACHER_DEMO_VIDEO
else:
    teacher_tfflie.write(teacher_video_file_buffer.read())
    teacher_vid = cv2.VideoCapture(teacher_tfflie.name)

st.sidebar.text('Teacher Video')
st.sidebar.video(teacher_tfflie.name)

teacherFrames = []
while teacher_vid.isOpened():
    i +=1
    ret, frame = teacher_vid.read()
    if not ret:
        continue
    teacherFrames.append(frame) # ndarry
    
print(len(teacherFrames))

st.sidebar.markdown('---')
student_video_file_buffer = st.sidebar.file_uploader("Please upload student video (or Using Webcam)", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
student_tfflie = tempfile.NamedTemporaryFile(delete=False)

if not student_video_file_buffer:
    if use_webcam:
        student_vid = cv2.VideoCapture(0)
    else:
        student_vid = cv2.VideoCapture(STUDENT_DEMO_VIDEO)
        student_tfflie.name = STUDENT_DEMO_VIDEO

else:
    student_tfflie.write(student_video_file_buffer.read())
    student_vid = cv2.VideoCapture(student_tfflie.name)

st.sidebar.text('Student Video')
st.sidebar.video(student_tfflie.name)

width = int(student_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(student_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(student_vid.get(cv2.CAP_PROP_FPS))

#codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
codec = cv2.VideoWriter_fourcc('V','P','0','9')
out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))
fps = 0
i = 0
# drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

kpi1, kpi2, kpi3 = st.beta_columns(3)

with kpi1:
    st.markdown("**FrameRate**")
    kpi1_text = st.markdown("0")

with kpi2:
    st.markdown("**Detected Poses**")
    kpi2_text = st.markdown("0")

with kpi3:
    st.markdown("**Score**")
    kpi3_text = st.markdown("0")

st.markdown("<hr/>", unsafe_allow_html=True)

# with mp_face_mesh.FaceMesh(
# min_detection_confidence=detection_confidence,
# min_tracking_confidence=tracking_confidence , 
# max_num_faces = max_faces) as face_mesh:
#     prevTime = 0

#     while vid.isOpened():
#         i +=1
#         ret, frame = vid.read()
#         if not ret:
#             continue

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(frame)

#         frame.flags.writeable = True
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         face_count = 0
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 face_count += 1
#                 mp_drawing.draw_landmarks(
#                 image = frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACE_CONNECTIONS,
#                 landmark_drawing_spec=drawing_spec,
#                 connection_drawing_spec=drawing_spec)
#         currTime = time.time()
#         fps = 1 / (currTime - prevTime)
#         prevTime = currTime
#         if record:
#             #st.checkbox("Recording", value=True)
#             out.write(frame)
#         #Dashboard
#         kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
#         kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
#         kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

#         frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
#         frame = image_resize(image = frame, width = 640)
#         stframe.image(frame,channels = 'BGR',use_column_width=True)

# st.text('Video Processed')

# output_video = open('output1.mp4','rb')
# out_bytes = output_video.read()
# st.video(out_bytes)

# vid.release()
# out. release()
