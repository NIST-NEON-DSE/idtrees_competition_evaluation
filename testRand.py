# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:23:24 2020

@author: d.stewart
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import scipy.io as sio
from PIL import Image
from RandCrowns import RandNeon
from RandCrowns import halo_parameters as p

im = Image.open('C:\\Users\\d.stewart\\NAVYREPO\\2020_NEON_Competition\\idtrees_competition_evaluation\\SJER_012.tif')
GT = sio.loadmat('C:\\Users\\d.stewart\\NAVYREPO\\2020_NEON_Competition\\idtrees_competition_evaluation\\SJER12GT.mat')
GT = GT['GT'][0].astype(int)

#Test Rand scores
xs = np.arange(110,160,1,dtype=int)
ys = np.arange(180,250,1,dtype=int)
ws = np.arange(70,140,1,dtype=int)
hs = np.arange(70,140,1,dtype=int)

K = 10
Xk = []
Scores = np.zeros((K,1))
Scores2 = np.zeros((K,1))
for k in range(K):
    par = p()
    xc = np.random.choice(len(xs))
    yc = np.random.choice(len(ys))
    wc = np.random.choice(len(ws))
    hc = np.random.choice(len(hs))
    det = np.array([xs[xc],ys[yc],ws[wc],hs[hc]]).astype(int)
    Xk.append(det)
    Scores[k] = RandNeon(GT,det,par)