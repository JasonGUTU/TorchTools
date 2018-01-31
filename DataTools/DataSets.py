import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os, random
import os.path
import numpy as np
import skimage.io as io
import dlib

from .FileTools import _image_file, _all_images, _video_image_file, _sample_from_videos_frames
from .Loaders import pil_loader, load_to_tensor
from .Prepro import _id, random_pre_process

from ..Functions import functional as Func
from ..Functions.TestTools import face_detection_5marks, high_point_to_low_point, generat_detection_label


class SRDataSet(data.Dataset):
    """
    DataSet for small images, easy to read
    do not need buffer
    random crop.
    all the image are same size
    """
    def __init__(self,
                 data_path,
                 lr_patch_size,
                 image_size,
                 scala=4,
                 interp=Image.BICUBIC,
                 mode='Y',
                 sub_dir=False,
                 prepro=random_pre_process):
        """
            :param data_path: Path to data root
            :param lr_patch_size: the Low resolution size, by default, the patch is square
            :param scala: SR scala, default is 4
            :param interp: interpolation for resize, default is Image.BICUBIC, optional [Image.BILINEAR, Image.BICUBIC]
            :param mode: 'RGB' or 'Y'
            :param sub_dir: if True, then all the images in the `data_path` directory AND child directory will be use
            :parem prepro: function fo to ``PIL.Image``!, will run this function before crop and resize
        """
        data_path = os.path.abspath(data_path)
        print('Initializing DataSet, data root: %s ...' % data_path)
        if sub_dir:
            self.image_file_list = _all_images(data_path)
        else:
            self.image_file_list = _image_file(data_path)
        print('Found %d Images...' % len(self.image_file_list))
        assert lr_patch_size * scala <= image_size, "Wrong size."
        self.lr_size = lr_patch_size
        self.image_size = image_size
        self.scala = scala
        self.interp = interp
        self.mode = mode
        self.crop_size = lr_patch_size * scala
        self.prepro = prepro

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        if self.mode == 'Y':
            image = pil_loader(self.image_file_list[index], mode='YCbCr')
        else:
            image = pil_loader(self.image_file_list[index], mode=self.mode)
        hr_img = Func.random_crop(self.prepro(image), self.crop_size)
        lr_img = Func.resize(hr_img, self.lr_size, interpolation=self.interp)
        if self.mode == 'Y':
            return Func.to_tensor(lr_img)[:1], Func.to_tensor(hr_img)[:1]
        else:
            return Func.to_tensor(lr_img), Func.to_tensor(hr_img)

    def __len__(self):
        return len(self.image_file_list)


class SRDataLarge(data.Dataset):
    """
    DataSet for Large images, hard to read once
    need buffer
    need to random crop
    all the image are Big size (DIV2K for example)
    """
    def __init__(self,
                 data_path,
                 lr_patch_size,
                 scala=4,
                 interp=Image.BICUBIC,
                 mode='RGB',
                 sub_dir=False,
                 prepro=random_pre_process,
                 buffer=4):
        """
        :param data_path: Path to data root
        :param lr_patch_size: the Low resolution size
        :param scala: SR scala
        :param interp: interp to resize
        :param mode: 'RGB' or 'Y'
        :param sub_dir: if True, then use _all_image
        :param prepro: function fo to ``PIL.Image``!!, will do before crop and resize
        :param buffer: how many patches cut from one image
        """
        data_path = os.path.abspath(data_path)
        print('Initializing DataSet, data root: %s ...' % data_path)
        if sub_dir:
            self.image_file_list = _all_images(data_path)
        else:
            self.image_file_list = _image_file(data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.lr_size = lr_patch_size
        self.scala = scala
        self.interp = interp
        self.mode = mode
        self.crop_size = lr_patch_size * scala
        self.prepro = prepro
        self.buffer = buffer
        self.current_image_index = -1
        self.current_patch_index = -1
        self.current_image = None

    def __len__(self):
        return len(self.image_file_list) * self.buffer

    def __getitem__(self, index):
        image_index = index // self.buffer
        if self.current_image_index != image_index:
            if self.mode == 'Y':
                image = pil_loader(self.image_file_list[image_index], mode='YCbCr')
            else:
                image = pil_loader(self.image_file_list[image_index], mode=self.mode)
            self.current_image = self.prepro(image)
            self.current_image_index = image_index
        w, h = self.current_image.size
        cropable = (w - self.lr_size * self.scala, h - self.lr_size * self.scala)
        cw = random.randrange(0, cropable[0])
        ch = random.randrange(0, cropable[1])
        hr_img = Func.crop(self.current_image, cw, ch, self.lr_size * self.scala, self.lr_size * self.scala)
        lr_img = Func.resize(hr_img, self.lr_size, interpolation=self.interp)
        if self.mode == 'Y':
            return Func.to_tensor(lr_img)[:1], Func.to_tensor(hr_img)[:1]
        else:
            return Func.to_tensor(lr_img), Func.to_tensor(hr_img)


# TODO: SRDataLargeGrid(data.Dataset)
# class SRDataLargeGrid(data.Dataset):
#     """
#     DataSet for Large images, hard to read once
#     need buffer
#     need to crop, but crop by grid
#     all the image are Big size (DIV2K for example)
#     """


class VideoData(data.Dataset):
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
    """
    def __init__(self, data_folder_root, scala, image_size_w, image_size_h, time_window=5, time_stride=2, loader=pil_loader, interp=Image.BICUBIC):
        super(VideoData, self). __init__()
        self.time_window = time_window
        self.time_stride = time_stride
        self.loader = loader

        video_list = os.listdir(data_folder_root)
        # number of videos
        self.n_videos = len(video_list)
        # number of frames for every video, list
        self.n_video_frames = [0] * self.n_videos
        # samples number for every video, list
        # sample number times videos number is the sample of the data set
        self.n_samples = [0] * self.n_videos
        # start sample of the video
        self.area_summed = [0] * self.n_videos
        # the path to every video folder
        self.video_folders = [None] * self.n_videos
        # 2-D list, path for every frame for every video of the data set
        self.frame_files = [None] * self.n_videos
        # Initial above by for loop
        for i in range(self.n_videos):
            video_folder = os.path.join(data_folder_root, video_list[i])
            self.video_folders[i] = video_folder
            frame_file_list = os.listdir(video_folder)
            frame_file_list.sort()
            self.n_video_frames[i] = len(frame_file_list)
            self.frame_files[i] = [None] * len(frame_file_list)
            for j in range(self.n_video_frames[i]):
                self.frame_files[i][j] = os.path.join(video_folder, frame_file_list[j])
            self.n_samples[i] = (self.n_video_frames[i] - self.time_window) // self.time_stride
            if i != 0:
                self.area_summed[i] = sum(self.n_samples[:i])

        self.toTensor = transforms.ToTensor()
        self.scala = transforms.Resize((image_size_h // scala, image_size_w // scala), interpolation=interp)
        self.crop = transforms.CenterCrop((image_size_h, image_size_w))

        self.buffer = [None] * self.time_window
        self.return_buffer = [None] * (self.time_window + 1)
        self.current_sample_index = None
        self.current_video_index = None

    def _index_parser(self, index):
        global_sample_index = index
        for i in range(self.n_videos):
            if self.area_summed[i] > global_sample_index:
                video_index = i - 1
                frame_index = global_sample_index - self.area_summed[video_index]
                return video_index, frame_index
        video_index = self.n_videos - 1
        frame_index = global_sample_index - self.area_summed[video_index]
        return video_index, frame_index

    def _load_reuse(self, video_index, sample_index):
        self.buffer[: self.time_window - self.time_stride] = self.buffer[self.time_stride:]
        frame_file_index_start = sample_index * self.time_stride
        frame_files = self.frame_files[video_index][frame_file_index_start + self.time_window - self.time_stride: frame_file_index_start + self.time_window]
        for i, file_path in enumerate(frame_files):
            self.buffer[self.time_window - self.time_stride + i] = self.loader(file_path)

    def _load_new(self, video_index, sample_index):
        frame_file_index_start = sample_index * self.time_stride
        frame_files = self.frame_files[video_index][frame_file_index_start: frame_file_index_start + self.time_window]
        for i, file_path in enumerate(frame_files):
            self.buffer[i] = self.loader(file_path)

    def _load_frames(self, video_index, sample_index):
        if (self.current_video_index == video_index) and (self.current_sample_index == (sample_index - 1)):
            self._load_reuse(video_index, sample_index)
        elif (self.current_video_index == video_index) and (self.current_sample_index == sample_index):
            pass
        else:
            self._load_new(video_index, sample_index)
        self.current_video_index = video_index
        self.current_sample_index = sample_index

    def __getitem__(self, index):
        video_index, sample_index = self._index_parser(index)
        self._load_frames(video_index, sample_index)
        for i in range(self.time_window):
            if i != (self.time_window // 2):
                self.return_buffer[i] = self.toTensor(self.scala(self.crop(self.buffer[i])))[:1]
            else:
                HR_center = self.crop(self.buffer[i])
                self.return_buffer[i] = self.toTensor(self.scala(HR_center))[:1]
                self.return_buffer[self.time_window] = self.toTensor(HR_center)[:1]
        return tuple(self.return_buffer)

    def __len__(self):
        return sum(self.n_samples)


class FaceDetectorData(data.Dataset):
    """
    This data set is for training face detector
    """
    def __init__(self, image_folder, image_size=(960, 540), lr_image_size=(120, 67), dlib_predictor='/home/sensetime/Documents/shape_predictor_68_face_landmarks.dat'):
        """
        :param image_folder: the hr image folder
        :param image_size: w, h
        :param lr_image_size: w, h
        :param dlib_predictor:
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_predictor)
        self.image_file_list = _all_images(image_folder)
        self.l_size = lr_image_size
        self.l_w, self.l_h = lr_image_size
        self.h_size = image_size

    def __getitem__(self, index):
        file_path = self.image_file_list[index]
        points_hr = face_detection_5marks(file_path, self.detector, self.predictor)
        points_on_lr = high_point_to_low_point(points_hr, self.h_size, self.l_size)
        lr_image = Func.to_tensor(Func.resize(pil_loader(file_path, mode='YCbCr'), (self.l_h, self.l_w)))[0]
        return lr_image, generat_detection_label(points_on_lr, self.l_w, self.l_h)

    def __len__(self):
        return len(self.image_file_list)


class OpticalFlowData(data.Dataset):
    """
    This Dataset is for training optical flow
    """
    def __init__(self, path, stride=2, mode='YCbCr'):
        """
        :param path:
        :param stride: 1 or 2
        """
        self.stride = stride
        self.mode = mode
        self.video_frame_list = _video_image_file(path)
        self.num_videos = len(self.video_frame_list)
        self.num_frames = [0] * self.num_videos
        for i, video in enumerate(self.video_frame_list):
            self.num_frames[i] = len(video)
        self.num_samples = [0] * self.num_videos
        for i, frames in enumerate(self.num_frames):
            self.num_samples[i] = frames // stride
        self.area_summed = [0] * self.num_videos
        for i, frames in enumerate(self.num_samples):
            if i != 0:
                self.area_summed[i] = sum(self.num_samples[:i])

    def _index_parser(self, index):
        global_sample_index = index
        for i in range(self.num_videos):
            if self.area_summed[i] > global_sample_index:
                video_index = i - 1
                frame_index = global_sample_index - self.area_summed[video_index]
                return video_index, frame_index
        video_index = self.num_videos - 1
        frame_index = global_sample_index - self.area_summed[video_index]
        return video_index, frame_index

    def _load_frames(self, video_index, sample_index):
        frame_t = pil_loader(self.video_frame_list[video_index][sample_index * self.stride], mode=self.mode)
        frame_tp1 = pil_loader(self.video_frame_list[video_index][sample_index * self.stride + 1], mode=self.mode)
        return Func.to_tensor(frame_t)[:1], Func.to_tensor(frame_tp1)[:1]

    def __getitem__(self, index):
        video_index, frame_index = self._index_parser(index)
        return self._load_frames(video_index, frame_index)

    def __len__(self):
        return sum(self.num_samples)




