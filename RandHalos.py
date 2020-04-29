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
GT = GT['GT'][0].astype(int)

def halo_parameters():
    par = {}
    par['inner'] = 1 #usually 1  
    par['outer'] = 1 #usually 1-5
    par['edge'] = 2 #usually 2-10
    par['plot'] = 1 #plot the halos
    par['area'] = 1000 #min area for scaling the parameters
    return par

def halo_corners(GT,im,par):

    #plot the inner halo
    innerCo = np.array([GT[0]+par['inner'],GT[1]+par['inner'],GT[2]-2*par['inner'],GT[3]-2*par['inner']])
    
    #plot the outer halo
    outerCo = np.array([GT[0]-par['outer'],GT[1]-par['outer'],GT[2]+2*par['outer'],GT[3]+2*par['outer']])
    
    #plot the edge halo
    edgeCo = np.array([GT[0]-par['edge'],GT[1]-par['edge'],GT[2]+2*par['edge'],GT[3]+2*par['edge']])
  
    
    
    #plot the boxes
    if par['plot']:
        fig,ax = plt.subplots(1)
        plt.imshow(im)
        
        #get GT rectangle
        rectGT = pat.Rectangle((GT[0],GT[1]),GT[2],GT[3],linewidth=2,edgecolor='r',fill=0)
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
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    halo_indices['inner'] = set(indices)
    
    #get outer
    inxywh = corners['outer']
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    halo_indices['outer'] = set(indices)
    
    #get edge
    inxywh = corners['edge']
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    halo_indices['edge'] = set(indices)
    
    return halo_indices


def TestDet():
    return np.array([100,170,150,160])
#    return np.array([125,30,30,40])
#    return GT
    
def get_det_indices(det):
    x = np.arange(det[0], det[0]+det[2], 1)
    y = np.arange(det[1], det[1]+det[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    return set(indices)

def check_GT_area(GT,par):
    GT_area = GT[2]*GT[3]
    if GT_area>par['area']:
        scaling = int(np.round(np.log(GT_area/par['area'])))
        par['outer'] = scaling*par['outer']
        par['edge'] = scaling*par['edge']
    return par
    
    

def RandNeon(GT,detection,im,par):
    
    #get set for detection
    det = get_det_indices(detection)
    
    #check/modify par based on log ratio to set small area
    par = check_GT_area(GT,par) 
    print(par)
    #get halos
    hcorners,ax = halo_corners(GT,im,par)
            
    #get sets for each halo
    halos = get_halo_indices(hcorners,im)
    
    #if detection contains outside of edge, extend edge halo
    if det.difference(halos['edge']):
        halos['edge']=halos['edge'].union(det)
#        print("Extending edge halo")
    
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
    
     #plot detection
    if par['plot']:
        rectDet = pat.Rectangle((detection[0],detection[1]),detection[2],detection[3],linewidth=2,edgecolor='k',fill=0)
        ax.add_patch(rectDet)
        plt.title('a= '+str(a)+', b = '+str(b)+', c= '+str(c)+', d= '+str(d)+'\n'+'Rand= '+str(np.round(score,2)),fontsize=10)
    
    return score

def get_halo_indices2(corners,GT,im):
    
    halo_indices = {}
    
    #get inner
    inxywh = corners['inner']
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    halo_indices['inner'] = set(indices)
    
    #get outer
    inxywh = corners['outer']
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    halo_indices['outer'] = set(indices)
    
    #get edge
    inxywh = corners['edge']
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,np.size(im))
    halo_indices['edge'] = set(indices)
    
    #check area of edge to inner area ratio and modify edge until the ratio is 1:1
    b_area = halo_indices['edge'].difference(halo_indices['outer'])
    b_ratio = len(b_area)/len(halo_indices['inner'])
    print(b_ratio)
    
    W = corners['outer'][2]
    H = corners['outer'][3]
    C = 2*W+2*H
    ww = GT[2]
    hh = GT[3]
    t = (-C+(C**2+4*ww*hh*4)**.5)/(8)
    inxyf = corners['outer']
    print("before solve" + str(inxyf))
    inxyf[0]-=t
    inxyf[1]-=t
    inxyf[2]+=2*t
    inxyf[3]+=2*t 
    print("quatsolver "+ str(inxyf))
    while b_ratio < 1:
         inxywh = corners['edge']
         inxywh[0]-=1
         inxywh[1]-=1
         inxywh[2]+=2
         inxywh[3]+=2
         x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
         y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
         X,Y = np.meshgrid(x,y)
         XY=np.array([X.flatten(),Y.flatten()])
         indices = np.ravel_multi_index(XY,np.size(im))
         halo_indices['edge'] = set(indices)
    
        #check area of edge to inner area ratio and modify edge until the ratio is 1:1
         b_area = halo_indices['edge'].difference(halo_indices['outer'])
         b_ratio = len(b_area)/len(halo_indices['inner'])
         print(b_ratio)
         print(inxywh)
    return corners, halo_indices

def halo_corners2(GT,im,par):

    #plot the inner halo
    innerCo = np.array([GT[0]+par['inner'],GT[1]+par['inner'],GT[2]-2*par['inner'],GT[3]-2*par['inner']])
    
    #plot the outer halo
    outerCo = np.array([GT[0]-par['outer'],GT[1]-par['outer'],GT[2]+2*par['outer'],GT[3]+2*par['outer']])
    
    #plot the edge halo
    edgeCo = np.array([GT[0]-par['edge'],GT[1]-par['edge'],GT[2]+2*par['edge'],GT[3]+2*par['edge']])
  
    
    
    #plot the boxes
    if par['plot']:
        fig,ax = plt.subplots(1)
        plt.imshow(im)
        
        #get GT rectangle
        rectGT = pat.Rectangle((GT[0],GT[1]),GT[2],GT[3],linewidth=2,edgecolor='r',fill=0)
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
    return corners,ax,fig

def plot_corners2(fig,GT,im,corners):
    
     #plot the inner halo
    innerCo = np.array(corners['inner'])
    
    #plot the outer halo
    outerCo = np.array(corners['outer'])
    
    #plot the edge halo
    edgeCo = np.array(corners['edge'])
    
    plt.close(fig)
    fig,ax = plt.subplots(1)
    plt.imshow(im)
        
    #get GT rectangle
    rectGT = pat.Rectangle((GT[0],GT[1]),GT[2],GT[3],linewidth=2,edgecolor='r',fill=0)
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
    
    return ax

def RandNeon2(GT,detection,im,par):
    
    #get set for detection
    det = get_det_indices(detection)
    
    #get halos
    hcorners,ax,fig = halo_corners2(GT,im,par)
            
    #get sets for each halo
    corners,halos = get_halo_indices2(hcorners,GT,im)
    
    #replot corners
    ax = plot_corners2(fig,GT,im,corners)
    
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
    
     #plot detection
    if par['plot']:
        rectDet = pat.Rectangle((detection[0],detection[1]),detection[2],detection[3],linewidth=2,edgecolor='k',fill=0)
        ax.add_patch(rectDet)
        plt.title('a= '+str(a)+', b = '+str(b)+', c= '+str(c)+', d= '+str(d)+'\n'+'Rand2= '+str(np.round(score,2)),fontsize=10)
    
    return score
    
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
    par = halo_parameters()
    xc = np.random.choice(len(xs))
    yc = np.random.choice(len(ys))
    wc = np.random.choice(len(ws))
    hc = np.random.choice(len(hs))
    det = np.array([xs[xc],ys[yc],ws[wc],hs[hc]]).astype(int)
    Xk.append(det)
    Scores[k] = RandNeon(GT,det,im,par)
    par = halo_parameters()
    Scores2[k] = RandNeon2(GT,det,im,par)
    
    