import os
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


class Logger(object):
    """Logger for easy log the training process."""
    def __init__(self, name, exp_dir, opt, log_dir='log', checkpoint_dir='checkpoint', sample='samples'):
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
        self.checkpoint_dir = os.path.join(self.exp_dir, checkpoint_dir)
        self.opt = opt
        try:
            os.mkdir(self.exp_dir)
            os.mkdir(self.log_dir)
            os.mkdir(self.checkpoint_dir)
            os.mkdir(self.sample)
            print('Creating: %s\n          %s\n          %s\n          %s' % (self.exp_dir, self.log_dir, self.sample, self.checkpoint_dir))
        except NotImplementedError:
            raise Exception('Check your dir.')
        except FileExistsError:
            pass

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

    def save(self, name, state_dict):
        """
        Torch save
        :param name:
        :param state_dict:
        :return:
        """
        torch.save(state_dict, os.path.join(self.checkpoint_dir, name))
