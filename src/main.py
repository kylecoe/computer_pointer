#!/usr/bin/env python3

import cv2
import numpy as np 
import os
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from head_pose_estimation import HeadPose
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
from visualize_output import visualize
import logging

FD = '/home/kyle/Desktop/starter/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.'
FLD = '/home/kyle/Desktop/starter/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.'
HP = '/home/kyle/Desktop/starter/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.'
GE = '/home/kyle/Desktop/starter/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.'

def argparser():

    parser = ArgumentParser()

    parser.add_argument("-f", "--face_detection_model", required=True, type=str, 
                        help="Specify the full path to the file as follows \
                            Ex: /home/usrX/Desktop/starter/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.")
    parser.add_argument("-ldm", "--landmark_detection_model", required=True, type=str, 
                        help="Specify the full path to the file as follows \
                            Ex: /home/usrX/Desktop/starter/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.")
    parser.add_argument("-p", "--pose_estimation_model", required=True, type=str, 
                        help="Specify the full path to the file as follows \
                            Ex: /home/usrX/Desktop/starter/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.")
    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str, 
                        help="Specify the full path to the file as follows \
                            Ex: /home/usrX/Desktop/starter/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.")
    parser.add_argument("-d", "--device", required=False, default='CPU',
                        help="Specify what kind of hardware to run on")
    parser.add_argument("-view", "--view_intermediate", required=False, default=False,
                        help="View the intermediate images with pose, gaze, face \
                            Input either 1 or 0")
    parser.add_argument("-i", "--input", required=True, type=str, default='cam',
                        help="Input the full path to the video file or enter 'cam' if using webcam")
    parser.add_argument("-e", "--extensions", required=False, type=str, default=None,
                        help="Add an extension if unsupprted layers exist")
    parser.add_argument("-as", "--async_mode", required=False, type=bool, default=False,
                        help="Determines whether to run inference in sync or async mode")

    return parser



def main():

    args = argparser().parse_args()
    device = args.device
    input_feed = args.input

    log = logging.getLogger()

    model_paths = {
        'facedet': args.face_detection_model + 'xml',
        'faceldmdet': args.landmark_detection_model + 'xml',
        'headpose': args.pose_estimation_model + 'xml',
        'gaze': args.gaze_estimation_model + 'xml'
    }

    for mp in model_paths.keys():
        if not os.path.isfile(model_paths[mp]):
            print(model_paths[mp])
            print('Recheck file path and try again')
            log.error("Not a file")
            raise FileNotFoundError 
    
    if input_feed == 'cam':

        feed = InputFeeder(input_type='cam')

    elif not os.path.isfile(input_feed):

        print('Recheck file path and try again')
        log.error("Unable to find specified video file")
        raise FileNotFoundError

    else:
        feed = InputFeeder(input_type='video', input_file=input_feed)

    facedet = FaceDetection(args.face_detection_model, args.device, args.extensions, args.async_mode)
    faceldmdet = FacialLandmarksDetection(args.landmark_detection_model, args.device, args.extensions, args.async_mode)
    headpose = HeadPose(args.pose_estimation_model, args.device, args.extensions, args.async_mode)
    gaze = GazeEstimation(args.gaze_estimation_model, args.device, args.extensions, args.async_mode)

    try:
        log.info('Loading models...')
        facedet.load_model()
        faceldmdet.load_model()
        headpose.load_model()   
        gaze.load_model()
        feed.load_data()
        log.info('Models loaded successfully!')
    except:
        log.error('One or more of the models failed to load..')
        exit(1)

    log.info('Initializing mouse controller')
    mouse = MouseController(precision='medium', speed='fast')

    for batch in feed.next_batch():
        face = facedet.predict(batch)
        eyes, eye_coords = faceldmdet.predict(face)
        pose = headpose.predict(face)

        point = gaze.predict(pose, eyes)
        #print('Gaze values = ', point[0], point[1])

        log.info('All inference complete')

        #print('view_inter = ', args.view_intermediate)
        if args.input == 'cam':
            point[0] = -point[0]
            
        mouse.move(point[0], point[1])
        if args.view_intermediate == True:
                visualize(pose, face, eye_coords, point)


if __name__ == "__main__":
    main()
