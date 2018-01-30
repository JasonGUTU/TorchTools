import os

import numpy as np
import skimage.io as io

import torch


LEFT_EYE = [36, 37, 38, 39, 40, 41]
LEFT_EYEBROW = [17, 18, 19, 20, 21]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
RIGHT_EYEBROW = [22, 23, 24, 25, 26]
MOUTH = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
LEFT_MOUTH = [48, 60]
RIGHT_MOUTH = [54, 64]
LEFT_MOST = [0, 1, 2]
RIGHT_MOST = [14, 15, 16]
TOP_MOST = [18, 19, 20, 23, 24, 25]
DOWN_MOST = [7, 8, 9]
NOSE_TIP = [31, 32, 33, 34, 35]


def generat_detection_label(point_set, size_w, size_h):
    """
    This function is done at PIL.Image!!
    :param point_set: the point set on PIL.Image
    :param image_size: Of type PIL.Image!!
    :return: labels with m channels, m is the number of points
    """
    number = len(point_set)
    label = np.zeros((number, size_h, size_w), dtype=np.float32)
    for i, point in enumerate(point_set):
        x, y = point
        label[i, int(y), int(x)] = 1.
    return torch.FloatTensor(label)


def high_point_to_low_point(point_set, size_h, size_l):
    """
    The three parameters will be (w, h)
    :param point: the point location
    :param size_h: high resolution image size
    :param size_l: low resolution image size
    :return:
    """
    h_1, h_2 = size_h
    l_1, l_2 = size_l
    lr_points = list()
    for point in point_set:
        x, y = point
        lr_points.append([round(x * (l_1 / h_1)), round(y * (l_2 / h_2))])
    return lr_points


def get_landmarks(img, detector, predictor):
    """
    Return landmark martix
    :param img: img read by skimage.io.imread
    :param detector: dlib.get_frontal_face_detector() instance
    :param predictor: dlib.shape_predictor('..?./shape_predictor_68_face_landmarks.dat')
    :return: landmark matrix
    """
    rects = detector(img, 1)
    return np.array([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def _centroid(landmarks, point_list):
    """
    Return the centroid of given points
    :param point_list: point list
    :return: array(centroid) (x, y)
    """
    x = np.zeros((len(point_list),))
    y = np.zeros((len(point_list),))
    for i, p in enumerate(point_list):
        x[i] = landmarks[p][0]
        y[i] = landmarks[p][1]
    x_mean = int(x.mean())
    y_mean = int(y.mean())
    return np.array([x_mean, y_mean])


def landmark_6(landmarks):
    """
    Return 6 landmarks : left_eye, right_eye, mouth, left, right, down
    :param landmarks: landmarks matrix
    :return: the points of 6 landmarks, ndarray
    """
    left_eye = _centroid(landmarks, LEFT_EYE)
    right_eye = _centroid(landmarks, RIGHT_EYE)
    mouth = _centroid(landmarks, MOUTH)
    left = _centroid(landmarks, LEFT_MOST)
    right = _centroid(landmarks, RIGHT_MOST)
    down = _centroid(landmarks, DOWN_MOST)
    return np.array([left_eye, right_eye, mouth, left, right, down])


def landmark_5(landmarks):
    """
    Return 5 landmarks : left_eye, right_eye, left_mouth, right_mouth, nose
    :param landmarks: landmarks matrix
    :return: the points of 6 landmarks, ndarray
    """
    left_eye = _centroid(landmarks, LEFT_EYE)
    right_eye = _centroid(landmarks, RIGHT_EYE)
    nose = _centroid(landmarks, NOSE_TIP)
    left_mouth = _centroid(landmarks, LEFT_MOUTH)
    right_mouth = _centroid(landmarks, RIGHT_MOUTH)
    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth])


def face_detection_6marks(path, detector, predictor):
    """
    :param path: image path
    :param detector: detector
    :param predictor: predictor
    :return:
    """
    img = io.imread(path)
    return landmark_6(get_landmarks(img, detector, predictor))


def face_detection_5marks(path, detector, predictor):
    """
    :param path: image path
    :param detector: detector
    :param predictor: predictor
    :return:
    """
    img = io.imread(path)
    return landmark_5(get_landmarks(img, detector, predictor))

