import os
from collections import OrderedDict
import torch


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
            print('Creating: %s\n          %s\n          %s\n          %s' % (self.exp_dir, self.log_dir, self.sample, self.checkpoint_dir))
            os.mkdir(self.exp_dir)
            os.mkdir(self.log_dir)
            os.mkdir(self.checkpoint_dir)
            os.mkdir(self.sample)
        except NotImplementedError:
            raise Exception('Check your dir.')
        except FileExistsError:
            pass

        self._parse()
        self.log_dict = OrderedDict()
        self.log_titles = OrderedDict()

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
        self.log_titles[log_name] = title_list
        log_file_name = os.path.join(self.log_dir, log_name + '.csv')
        self.log_dict[log_name] = log_file_name
        with open(log_file_name, 'w') as f:
            f.write(','.join(title_list) + '\n')

    def add_log(self, name, args):
        """
        Add record to log `name`
        :param name: name of log to add
        :param args: list, according to the title
        :return:
        """
        assert len(self.log_titles[name]) == len(args), 'Bad log input.'
        with open(self.log_dict[name], 'a') as f:
            f.write(','.join(args) + '\n')

    def save(self, name, state_dict):
        """
        Torch save
        :param name:
        :param state_dict:
        :return:
        """
        torch.save(state_dict, os.path.join(self.checkpoint_dir, name))








