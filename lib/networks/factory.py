# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from .lstmTrain import lstmTrain
from .lstmTest import lstmTest

__sets = {}


def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'LSTM':
        if name.split('_')[1] == 'train':
            return lstmTrain()
        elif name.split('_')[1] == 'test':
            return lstmTest()
        else:
            raise KeyError('Unknown dataset: {}'.format(name))


def list_networks():
    """List all registered imdbs."""
    return list(__sets.keys())
