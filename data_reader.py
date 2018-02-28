'''
Author: Diego Aguirre
Last Modified: Sep 12, 2017
Description: Definition of a simple function that
            generates random data to test the implementation of
            Skanteze's LSTM network
'''
import numpy as np
import scipy.io as sio


def get_data():
    mat = sio.loadmat('./Maptask_Data_v5.mat')

    x_train = mat['trainFeatures']
    y_train = mat['trainLabels']
    x_test = mat['testFeatures']
    y_test = mat['testLabels']

    return x_train, y_train, x_test, y_test