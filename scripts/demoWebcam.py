import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import cv2
import time
import joblib
import argparse
import gradio as gr
from loguru import logger

sys.path.append('.')
from pare.core.tester import PARETester
from pare.utils.demo_utils import (
    download_youtube_clip,
    video_to_images,
    images_to_video,
)

CFG = 'data/pare/checkpoints/pare_w_3dpw_config.yaml'
CKPT = 'data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt'
MIN_NUM_FRAMES = 0

def SMPLPose(img):
  
def main(args):
    demo = gr.Interface(
        SMPLPose, 
        gr.Image(source="webcam", streaming=True), 
        "image",
        live=True
    )
    demo.launch()
    
    output_path = args.output_folder
    os.makedirs(output_path, exist_ok=True)
    logger.add(
        os.path.join(args.output_folder, 'demo.log'),
        level='INFO',
        colorize=False,
    )
    logger.info(f'Demo options: \n {args}')

    tester = PARETester(args)

    total_time = time.time()
    detections = tester.run_detector(input_image_folder) # Modify
    pare_time = time.time()
    tester.run_on_image_folder(input_image_folder, detections, output_path, output_img_folder,
                               run_smplify=args.smplify) # Modify
    end = time.time()
    
#     fps = num_frames / (end - pare_time)

    del tester.model

#     logger.info(f'PARE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    logger.info(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    logger.info(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')


    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default=CFG,
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default=CKPT,
                        help='checkpoint path')

    parser.add_argument('--exp', type=str, default='',
                        help='short description of the experiment')

    parser.add_argument('--output_folder', type=str, default='logs/demo/demo_results',
                        help='output folder to write results')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size of PARE')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--no_save', action='store_true',
                        help='disable final save of output results.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--draw_keypoints', action='store_true',
                        help='draw 2d keypoints on rendered image.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--smplify', action='store_true',
                        help='run MMPose and smplify to refine poses further')

    args = parser.parse_args()

    main(args)
  
