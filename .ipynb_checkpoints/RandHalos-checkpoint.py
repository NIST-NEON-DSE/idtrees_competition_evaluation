# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:52:00 2020

@author: d.stewart
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import scipy.io as sio
from PIL import Image

im = Image.open('C:\\Users\\d.stewart\\NAVYREPO\\2020_NEON_Competition\\idtrees_competition_evaluation\\SJER_012.tif')
GT = sio.loadmat('C:\\Users\\d.stewart\\NAVYREPO\\2020_NEON_Competition\\idtrees_competition_evaluation\\SJER12GT.mat')
GT = GT['GT'][0]

def halo_parameters(Nout,Nin,Nedge):
    par = {}
    par['inner'] = Nin  
    par['outer'] = Nout
    par['edge'] = Nedge
    par['plot'] = 1
    return par

def halo_corners(GT,im,par):

    #plot the inner halo
    innerCo = np.array([GT[0]+par['inner'],400-GT[1]+par['inner'],GT[2]-2*par['inner'],GT[3]-2*par['inner']])
    
    #plot the outer halo
    outerCo = np.array([GT[0]-par['outer'],400-GT[1]-par['outer'],GT[2]+2*par['outer'],GT[3]+2*par['outer']])
    
    #plot the edge halo
    edgeCo = np.array([GT[0]-par['edge'],400-GT[1]-par['edge'],GT[2]+2*par['edge'],GT[3]+2*par['edge']])
  
    
    
    #plot the boxes
    if par['plot']:
        fig,ax = plt.subplots(1)
        plt.imshow(im)
        
        #get GT rectangle
        rectGT = pat.Rectangle((GT[0],400-GT[1]),GT[2],GT[3],linewidth=2,edgecolor='r',fill=0)
        ax.add_patch(rectGT)
        
        #inner
        rectIn = pat.Rectangle((innerCo[0],innerCo[1]),innerCo[2],innerCo[3],linewidth=2,edgecolor='m',fill=0)
        ax.add_patch(rectIn)
        
        # outer
        rectOut = pat.Rectangle((outerCo[0],outerCo[1]),outerCo[2],outerCo[3],linewidth=2,edgecolor='tab:purple',fill=0)
        ax.add_patch(rectOut)
        
        #edge
        rectEdge = pat.Rectangle((edgeCo[0],edgeCo[1]),edgeCo[2],edgeCo[3],linewidth=2,edgecolor='tab:blue',fill=0)
        ax.add_patch(rectEdge)
    
    corners = {}
    corners['inner'] = innerCo
    corners['outer'] = outerCo
    corners['edge'] = edgeCo
    return corners,ax

def get_halo_indices(corners,im):
    
    halo_indices = {}
    
    #get inner
    inxywh = corners['inner']
    x = np.arange(inxywh[0], inxywh[0]+GT[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+GT[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    halo_indices['inner'] = set(indices)
    
    #get outer
    inxywh = corners['outer']
    x = np.arange(inxywh[0], inxywh[0]+GT[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+GT[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    halo_indices['outer'] = set(indices)
    
    #get edge
    inxywh = corners['edge']
    x = np.arange(inxywh[0], inxywh[0]+GT[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+GT[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    halo_indices['edge'] = set(indices)
    
    return halo_indices


def TestDet(par):
    return np.array([200,250,GT[2]+2*par['edge'],GT[3]+2*par['edge']])
#    return GT
    
def get_det_indices(det):
    x = np.arange(det[0], det[0]+det[2], 1)
    y = np.arange(det[1], det[1]+det[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = set(np.ravel_multi_index(XY,np.size(im)))
    return indices
    

def RandNeon(GT,detection,im,par):
    
    #get halos
    hcorners,ax = halo_corners(GT,im,par)
    
    #get sets for each halo
    halos = get_halo_indices(hcorners,im)
    
    #get set for detection
    det = get_det_indices(detection)
    
    #compute a
    a_set = det.intersection(halos['inner'])
    a = len(a_set)**2
    
    #compute b
    edge_wo = halos['edge'].difference(halos['outer'])
    det_edge = det.intersection(halos['edge'])
    b_set = edge_wo.difference(det_edge)
    b = len(b_set)**2
    
    #compute c
    out_only = halos['edge'].difference(halos['outer'])
    c_set = det.intersection(out_only)
    c = len(c_set)**2
    
    #compute d
    det_in = det.intersection(halos['inner'])
    d_set = halos['inner'].difference(det_in)
    d = len(d_set)**2
    
    #plot detection
    if par['plot']:
        rectDet = pat.Rectangle((detection[0],detection[1]),detection[2],detection[3],linewidth=2,edgecolor='k',fill=0)
        ax.add_patch(rectDet)
    
    correct = a+b
    incorrect = c+d
    score = correct/(correct+incorrect)
    return score
    