import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import os, random
import os.path
import numpy as np

from .FileTools import _image_file, _all_images, _video_image_file, _sample_from_videos_frames
from .Loaders import pil_loader, load_to_tensor
from .Prepro import _id, random_pre_process

from ..Functions import functional as Func


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








