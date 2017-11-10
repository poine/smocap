#!/usr/bin/env python

import numpy as np
import matplotlib, matplotlib.pyplot as plt


if __name__ == '__main__':
    npzfile = np.load('/tmp/lost_log.npz')
    pos = npzfile['arr_0']
    plt.scatter(pos[:,0], pos[:,1])
    plt.show()
