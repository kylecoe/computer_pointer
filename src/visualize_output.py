import numpy as np 
import cv2
import math
import matplotlib as plt 

# Implemented using the help of forum posts and www.learnopencv.com
# https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

def visualize(pose, face, eyes, point):

    yaw = pose[0]
    pitch = pose[1]
    roll = pose[2]

    cx = int(face.shape[1] / 2)
    cy = int(face.shape[0] / 2)

    focal_length = 950.0
    scale = 50

    camera_matrix = np.array(
                         [[focal_length, 0, cx],
                         [0, focal_length, cy],
                         [0, 0, 1]], dtype = "double"
                         )

    #print ("Camera Matrix :\n {0}".format(camera_matrix))

    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                   [0, 1, 0],
                   [math.sin(yaw), 0, math.cos(yaw)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
     
    R = np.dot(Rz, Ry, Rx)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]

    xaxis = np.dot(R, xaxis) + o
    yaxis = np.dot(R, yaxis) + o
    zaxis = np.dot(R, zaxis) + o
    zaxis1 = np.dot(R, zaxis1) + o

    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(face, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(face, (cx, cy), p2, (0, 255, 0), 2)

    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(face, p1, p2, (255, 0, 0), 2)
    cv2.circle(face, p2, 3, (255, 0, 0), 2)

    #gv1 = cv2.line(face, (int(eyes[0].shape[1] / 2), int(eyes[0].shape[0] / 2)), 
    #               (int((eyes[0].shape[1]) / 2)+50, int((eyes[0].shape[0]) / 2)+50), (0,0,255, 1))
    #gv2 = cv2.line(face, (int((eyes[1].shape[1]) / 2), int((eyes[1].shape[0]) / 2)), 
    #               (int((eyes[1].shape[1]) / 2)+50, int((eyes[1].shape[0]) / 2)+50), (0,0,255, 1))

    x, y, w = int(point[0]), int(point[1]), 6000
    x = int(x-w)
    y = int(y-w)
    le_cent = (int(eyes[0][0] + 20), int(eyes[0][1]+20))
    re_cent = (int(eyes[1][0] + 20), int(eyes[1][1]+20))
    #le =cv2.line(face, le_cent, (x, y), (255,0,0), 2)
    #cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
    #re = cv2.line(face, re_cent, (x, y), (0,0,255), 2)
    #cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
    #cv2.imshow('le', eyes[0])
    #cv2.imshow('re', eyes[1])
    
    left_eye = cv2.rectangle(face, (eyes[0][0], eyes[0][1]), (eyes[0][2], eyes[0][3]), (0,0,255))
    right_eye = cv2.rectangle(face, (eyes[1][0], eyes[1][1]), (eyes[1][2], eyes[1][3]), (0,0,255))
    cv2.imshow('Frame', face)
    cv2.waitKey()