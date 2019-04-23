#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Shapelets.io
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import scipy.io
import matrixprofile as mp
import numpy as np
import matplotlib.pyplot as plt
import time

# Load data from file
mat = scipy.io.loadmat('./datasets/ecg/ssel102m.mat')
min = 0
max = mat['val'][1].size  #225000 points

ts = np.array(mat['val'][1][min:max], dtype=float)
plt.plot(ts)
plt.show()

m = 400

start = time.time()
profile, index = mp.stompSJ(ts, m)
end = time.time()
print("Execution Time:" + str(end - start))

best_discord = mp.find_best_discord(profile)

# Position of best discord
print(best_discord)

mp.plot_motif(ts, profile, index, m)
