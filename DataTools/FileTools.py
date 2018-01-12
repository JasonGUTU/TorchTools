import argparse
import os
import sys
import time
from math import log10

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def _video_image_file(path):
    """
    Data Store Format:

    Data Folder
        |
        |-Video Folder
        |   |-Video Frames (images)
        |
        |-Video Folder
            |-Video Frames (images)

        ...

        |-Video Folder
            |-Video Frames (images)

    :param path: path to Data Folder, absolute path
    :return: 2D list of str, the path is absolute path
            [[Video Frames], [Video Frames], ... , [Video Frames]]
    """
    abs_path = os.path.abspath(path)
    video_list = os.listdir(abs_path)
    frame_list = [None] * len(video_list)
    for i in range(len(video_list)):
        video_list[i] = os.path.join(path, video_list[i])
        frame_list[i] = os.listdir(video_list[i])
        for j in range(len(os.listdir(video_list[i]))):
            frame_list[i][j] = os.path.join(video_list[i], frame_list[i][j])
        frame_list[i].sort()
    return frame_list


def _sample_from_videos_frames(path, time_window, time_stride):
    """
    Sample from video frames files
    :param path: path to Data Folder, absolute path
    :param time_window: number of frames in one sample
    :param time_stride: strides when sample frames
    :return: 2D list of str, absolute path to each frames
            [[Sample Frames], [Sample Frames], ... , [Sample Frames]]
    """
    video_frame_list = _video_image_file(path)
    sample_list = list()
    for video in video_frame_list:
        assert isinstance(video, list), "Plz check video_frame_list = _video_image_file(path) should be 2D list"
        for i in range(0, len(video), time_stride):
            sample = video[i:i + time_window]
            if len(sample) != time_window:
                break
            sample.append(video[i + (time_window // 2)])
            sample_list.append(sample)
    return sample_list


# TODO: large sample number function
def _sample_from_videos_frames_large(path, time_window, time_stride):
    """
    write to a file, return one sample once. use pointer
    :param path:
    :param time_window:
    :param time_stride:
    :return:
    """
    pass


def _image_file(path):
    """
    return list of images in the path
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    abs_path = os.path.abspath(path)
    image_files = os.listdir(abs_path)
    for i in range(len(image_files)):
        if (not os.path.isdir(image_files[i])) and (_is_image_file(image_files[i])):
            image_files[i] = os.path.join(abs_path, image_files[i])
    return image_files


def _all_images(path):
    """
    return all images in the folder
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    # TODO: Tail Call Elimination
    abs_path = os.path.abspath(path)
    image_files = list()
    for subpath in os.listdir(abs_path):
        if os.path.isdir(os.path.join(abs_path, subpath)):
            image_files = image_files + _all_images(os.path.join(abs_path, subpath))
        else:
            if _is_image_file(subpath):
                image_files.append(os.path.join(abs_path, subpath))
    return image_files



