# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:09:47 2020

@author: d.stewart
04/02/2020 - Dylan Stewart
"""

import numpy as np


def halo_parameters():
    
    par = {}
    par['inner'] = 1 #usually 1  
    par['outer'] = 1 #usually 1-5
    par['edge'] = 2 #usually 2-10
    par['area'] = 1000 #min area for scaling the parameters
    par['plot'] = np.array([400,400]).astype(int) #size of each plot
    return par

def halo_corners(GT,par):

    #plot the inner halo
    innerCo = np.array([GT[0]+par['inner'],GT[1]+par['inner'],GT[2]-2*par['inner'],GT[3]-2*par['inner']])
    
    #plot the outer halo
    outerCo = np.array([GT[0]-par['outer'],GT[1]-par['outer'],GT[2]+2*par['outer'],GT[3]+2*par['outer']])
    
    #plot the edge halo
    edgeCo = np.array([GT[0]-par['edge'],GT[1]-par['edge'],GT[2]+2*par['edge'],GT[3]+2*par['edge']])
      
    corners = {}
    corners['inner'] = innerCo
    corners['outer'] = outerCo
    corners['edge'] = edgeCo
    
    return corners

def get_halo_indices(corners,par):
    
    halo_indices = {}
    
    #get inner corners
    inxywh = corners['inner']
    
    #get span of the box
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    
    #convert to indices
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,par['plot'],mode='clip')
    halo_indices['inner'] = set(indices)
    
    #get outer corners
    inxywh = corners['outer']
    
    #get span of box
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
   
    #convert to indices
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,par['plot'],mode='clip')
    halo_indices['outer'] = set(indices)
    
    #get edge corners
    inxywh = corners['edge']
    
    #get span of the box
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    
    #convert to indices    
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,par['plot'],mode='clip')
    halo_indices['edge'] = set(indices)
    
    return halo_indices
    
def get_det_indices(det,par):
    
    x = np.arange(det[0], det[0]+det[2], 1)
    y = np.arange(det[1], det[1]+det[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,par['plot'],mode='clip')
    
    return set(indices)

def check_GT_area(GT,par):
    GT_area = GT[2]*GT[3]
    if GT_area>par['area']:
        scaling = int(np.round(np.log(GT_area/par['area'])))
        par['outer'] = scaling*par['outer']
        par['edge'] = scaling*par['edge']
    
    return par
    
def RandNeon(GT,detection,par):
    
    #get set for detection
    det = get_det_indices(detection,par)
    
    #check/modify par based on log ratio to set small area
    par = check_GT_area(GT,par) 
    
    #get halos
    hcorners = halo_corners(GT,par)
            
    #get sets for each halo
    halos = get_halo_indices(hcorners,par)
    
    #if detection contains outside of edge, extend edge halo
    if det.difference(halos['edge']):
        halos['edge']=halos['edge'].union(det)
    
    #compute a
    a_set = det.intersection(halos['inner'])
    a = len(a_set)**2      
 
    #compute b
    edge_wo = halos['edge'].difference(halos['outer'])
    b_set = edge_wo.difference(det)
    b = len(b_set)**2
    
    #compute c
    out_only = halos['edge'].difference(halos['outer'])
    c_set = det.intersection(out_only)
    c = len(c_set)**2
    
    #compute d
    d_set = halos['inner'].difference(det)
    d = len(d_set)**2
         
    correct = a+b
    incorrect = c+d
    score = correct/(correct+incorrect)
    
    return score