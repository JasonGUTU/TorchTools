import os
import pickle
import dominate
from dominate.tags import *
from collections import OrderedDict
import torch


class _FileLogger(object):
    """Logger for losses files"""
    def __init__(self, logger, log_name, title_list):
        """
        Init a new log term
        :param log_name: The name of the log
        :param title_list: list, titles to store
        :return:
        """
        assert isinstance(logger, Logger), "logger should be instance of Logger"
        self.titles = title_list
        self.len = len(title_list)
        self.log_file_name = os.path.join(logger.log_dir, log_name + '.csv')
        with open(self.log_file_name, 'w') as f:
            f.write(','.join(title_list) + '\n')

    def add_log(self, value_list):
        assert len(value_list) == self.len, "Log Value doesn't match"
        for i in range(self.len):
            if not isinstance(value_list[i], str):
                value_list[i] = str(value_list[i])
        with open(self.log_file_name, 'a') as f:
            f.write(','.join(value_list) + '\n')


class HTML:
    def __init__(self, logger, reflesh=0):
        self.logger = logger
        self.title = logger.opt.exp_name
        self.web_dir = logger.web
        self.img_dir = logger.img
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=self.title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):  #TODO:image
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


class Logger(object):
    """Logger for easy log the training process."""
    def __init__(self, name, exp_dir, opt, commend='', HTML_doc=False, log_dir='log', checkpoint_dir='checkpoint', sample='samples', web='web'):
        """
        Init the exp dirs and generate readme file
        :param name: experiment name
        :param exp_dir: dir name to store exp
        :param opt: argparse namespace
        :param log_dir:
        :param checkpoint_dir:
        :param sample:
        """
        self.name = name
        self.exp_dir = os.path.abspath(exp_dir)
        self.log_dir = os.path.join(self.exp_dir, log_dir)
        self.sample = os.path.join(self.exp_dir, sample)
        self.web = os.path.join(self.exp_dir, web)
        self.img = os.path.join(self.web, 'images')  #TODO:image
        self.checkpoint_dir = os.path.join(self.exp_dir, checkpoint_dir)
        self.opt = opt
        try:
            os.mkdir(self.exp_dir)
            os.mkdir(self.log_dir)
            os.mkdir(self.checkpoint_dir)
            os.mkdir(self.sample)
            os.mkdir(self.web)
            os.mkdir(self.img)
            print('Creating: %s\n          %s\n          %s\n          %s' % (self.exp_dir, self.log_dir, self.sample, self.checkpoint_dir))
        except NotImplementedError:
            raise Exception('Check your dir.')
        except FileExistsError:
            pass
        with open(os.path.join(self.exp_dir, 'run_commend.txt'), 'w') as f:
            f.write(commend)
        self.html_tag = HTML_doc
        if HTML_doc:
            self.html = HTML(self)
            self.html.add_header(opt.exp_name)
            self.html.save()

        self._parse()

    def _parse(self):
        """
        print parameters and generate readme file
        :return:
        """
        attr_list = list()
        exp_readme = os.path.join(self.exp_dir, 'README.txt')

        for attr in dir(self.opt):
            if not attr.startswith('_'):
                attr_list.append(attr)
        print('Init parameters...')
        with open(exp_readme, 'w') as readme:
            readme.write(self.name + '\n')
            for attr in attr_list:
                line = '%s : %s' % (attr, self.opt.__getattribute__(attr))
                print(line)
                readme.write(line)
                readme.write('\n')

    def init_scala_log(self, log_name, title_list):
        """
        Init a new log term
        :param log_name: The name of the log
        :param title_list: list, titles to store
        :return:
        """
        return _FileLogger(self, log_name, title_list)

    def _parse_save_name(self, tag, epoch, step='_', type='.pth'):
        return str(epoch) + step + tag + type

    def save_epoch(self,  epoch, name, state_dict):
        """
        Torch save
        :param name:
        :param state_dict:
        :return:
        """
        torch.save(state_dict, os.path.join(self.checkpoint_dir, self._parse_save_name(name, epoch)))

    def save(self, name, state_dict):
        """
        Torch save
        :param name:
        :param state_dict:
        :return:
        """
        torch.save(state_dict, os.path.join(self.checkpoint_dir, name))

    def load_epoch(self, name, epoch):
        return torch.load(os.path.join(self.checkpoint_dir, self._parse_save_name(name, epoch)))

    def print_log(self, string):
        print(string)
        with open(os.path.join(self.log_dir, 'output.log'), 'a') as f:
            f.write(string)

    def _parse_web_image_name(self, Nom, tag, step='_', type='.png'):
        return 'No.' + str(Nom) + step + tag + type

    def _save_web_images(self, pil, name):
        save_path = os.path.join(self.img, name)
        pil.save(save_path)

    def _add_image_table(self, img_list, tag_list):
        assert len(img_list) == len(tag_list), 'check input'
        self.html.add_images(img_list, tag_list, img_list)
        self.html.save()

    def save_image_record(self, epoch, image_dict):
        img_list = list()
        tag_list = list()
        for tag, image in image_dict.items():
            image_name = self._parse_web_image_name(epoch, tag)
            img_list.append(image_name)
            tag_list.append(tag)
            image.save(os.path.join(self.img, image_name))
        self.html.add_header('Epoch: %d' % epoch)
        self._add_image_table(img_list, tag_list)

    def save_logger(self):
        with open(os.path.join(self.exp_dir, 'logger.pkl'), 'w') as f:
            pickle.dump(self, f)
