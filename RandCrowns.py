"""
@author: Dylan Stewart
updated: 04/16/2020
    
    input variables:
        GroundTruthBox - numpy array [x y width height]
        DetectionBox   - numpy array [x y width height]
    
    output:
        score - float

to use this code:    
    from parameters import evaluation_parameters
    from RandCrowns import RandNeon
    
    *if you want to see the plots of the halos
    set plot = 1 in parameters
    
    score = RandNeon(GroundTruthBox,DetectionBox,par)
    this will give you the score and plot the ground truth, inner, outer,
    and edge halo
    
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat

def get_det_indices(det,par):
    x = np.arange(det[0], det[0]+det[2], 1)
    y = np.arange(det[1], det[1]+det[3], 1)
    X,Y = np.meshgrid(x,y)
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,par.area,mode='clip')
    return set(indices)

def halo_corners(GT,im, par):

    #plot the inner halo
    innerCo = np.array([GT[0]+par.inner,GT[1]+par.inner,GT[2]-2*par.inner,GT[3]-2*par.inner])
    
    #plot the outer halo
    outerCo = np.array([GT[0]-par.outer,GT[1]-par.outer,GT[2]+2*par.outer,GT[3]+2*par.outer])
    
    #plot the edge halo
    edgeCo = np.array([GT[0]-par.edge,GT[1]-par.edge,GT[2]+2*par.edge,GT[3]+2*par.edge])
  
    corners = {}
    corners['inner'] = innerCo
    corners['outer'] = outerCo
    corners['edge'] = edgeCo
    
    return corners
    

def plot_corners(GT,corners,im):
    
     #plot the inner halo
    innerCo = np.array(corners['inner'])
    
    #plot the outer halo
    outerCo = np.array(corners['outer'])
    
    #plot the edge halo
    edgeCo = np.array(corners['edge'])
    
    fig,ax = plt.subplots(1)
    ax.invert_yaxis()
        
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
    
    return ax,fig

def get_halo_indices(corners,GT,par):
    
    halo_indices = {}
    
    #get inner
    inxywh = corners['inner']
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    #make sure outer halo doesn't go outside plot boundaries
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,par.area,mode='clip')
    halo_indices['inner'] = set(indices)
    
    #get outer
    inxywh = corners['outer']
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    #make sure outer halo doesn't go outside plot boundaries
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,par.area,mode='clip')
    halo_indices['outer'] = set(indices)
    
    #get edge
    inxywh = corners['edge']
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    #make sure outer halo doesn't go outside plot boundaries
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,par.area,mode='clip')
    halo_indices['edge'] = set(indices)
    
    #set parameters based on area of gt box
    W = corners['outer'][2]
    H = corners['outer'][3]
    C = 2*W+2*H
    ww = GT[2]
    hh = GT[3]
    t = (-C+(C**2+4*ww*hh*4)**.5)/(8)
    inxywh = corners['edge']
    inxywh[0]-=t
    inxywh[1]-=t
    inxywh[2]+=2*t
    inxywh[3]+=2*t
    corners['edge'] = inxywh
    x = np.arange(inxywh[0], inxywh[0]+inxywh[2], 1)
    y = np.arange(inxywh[1], inxywh[1]+inxywh[3], 1)
    X,Y = np.meshgrid(x,y)
    #make sure outer halo doesn't go outside plot boundaries
    XY=np.array([X.flatten(),Y.flatten()])
    indices = np.ravel_multi_index(XY,par.area,mode='clip')
    halo_indices['edge'] = set(indices)
    
    return corners, halo_indices


def RandNeon(GT,detection,im,par, pname = None):
   
    #get set for detection
    det = get_det_indices(detection,par)
    
    #get halos
    hcorners = halo_corners(GT,im, par)    
        
    #get sets for each halo
    corners,halos = get_halo_indices(hcorners,GT, par)
     
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
         
    correct = float(a+b)
    incorrect = float(c+d)
    
    if a == 0:
        score = 0.0
    else:
        score = correct/(correct+incorrect)
    
     #plot detection
    if par.save == 1 and pname is not None:
        ax,fig = plot_corners(GT,corners,im)
        rectDet = pat.Rectangle((detection[0],detection[1]),detection[2],detection[3],linewidth=2,edgecolor='k',fill=0)
        ax.add_patch(rectDet)
        im = np.uint8(im)
        plt.imshow(np.flipud(im),origin='lower')
        plt.title('a= '+str(a)+', b = '+str(b)+', c= '+str(c)+', d= '+str(d)+'\n'+'Rand= '+str(np.round(score,2)),fontsize=10)
        fig.savefig(par.outputdir+"imgs/"+pname[:-4]+'.png')
        plt.close(fig)
    return score