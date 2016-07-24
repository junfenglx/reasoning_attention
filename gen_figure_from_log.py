# -*- coding: utf-8 -*-

# Created by junfeng on 4/28/16.

# logging config
import logging
import traceback

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

import sys

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd


def get_acc_from_logfile(log_filename):
    trains_accuracy = []
    devs_accuracy = []
    tests_accuracy = []
    epochs = []
    with open(log_filename, 'r') as f:
        for line in f:
            if 'Epoch' in line:
                epoch = int(line.split()[1])
                epochs.append(epoch)
            elif 'training accuracy' in line and 'current' not in line:
                train_accuracy = float(line.split()[-2]) / 100
                trains_accuracy.append(train_accuracy)
            elif 'validation accuracy' in line:
                dev_accuracy = float(line.split()[-2]) / 100
                devs_accuracy.append(dev_accuracy)
            elif 'test accuracy' in line:
                test_accuracy = float(line.split()[-2]) / 100
                tests_accuracy.append(test_accuracy)
    assert len(trains_accuracy) == len(devs_accuracy)
    assert len(trains_accuracy) == len(tests_accuracy)
    assert len(trains_accuracy) == len(epochs)
    return trains_accuracy, devs_accuracy, tests_accuracy, epochs


def to_float(s):
    try:
        f = float(s)
    except ValueError:
        f = float(s[:-1])
    return f


def gen_figure(acc_df, log_filename):
    title = log_filename.rsplit('.', 1)[0]
    figure, ax = plt.subplots()
    acc_df.plot(x='epoch', y='train_acc', ax=ax)
    acc_df.plot(x='epoch', y='dev_acc', ax=ax)
    acc_df.plot(x='epoch', y='test_acc', ax=ax)
    plt.ylabel('accuracy')
    plt.title(title)
    figure.savefig(title + '.png')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('run with {0} log_file'.format(sys.argv[0]))
        sys.exit(1)

    log_filename = sys.argv[1]
    logger.info('parsing log file {0}'.format(log_filename))
    columns = ['train_acc', 'dev_acc', 'test_acc', 'epoch']
    data = {}
    try:
        ret = get_acc_from_logfile(log_filename)
        for k, v in zip(columns, ret):
            data[k] = v
        acc_df = pd.DataFrame(data)
        print(acc_df[-10:])
        logger.info('generate figure ...')
        gen_figure(acc_df, log_filename)
    except Exception as e:
        traceback.print_exc()
