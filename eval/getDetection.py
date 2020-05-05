#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:28:41 2020

@author: sergiomarconi
"""

def get_vertex_per_plot(pl):
    site = pl.split("_")[0]
    pix_per_meter = 10
    detection_path = './submission/'+site+'_submission.csv'
    ras_path = "./RS/" + pl + ".tif"
    #read plot raster to extract detections within the plot boundaries
    raster = rasterio.open(ras_path)

    gdf = geopandas.read_file(
        detection_path,
        bbox=raster.bounds,
    )
    gtf = geopandas.read_file(
        './submission/'+site+'_ground.csv',
        bbox=raster.bounds,
    )
    # turn WTK into coordinates within in the image
    gdf_limits = gdf.bounds
    gtf_limits = gtf.bounds
    
    xmin = raster.bounds[0]
    ymin = raster.bounds[1]
    
    #length
    gdf_limits['maxy'] = (gdf_limits['maxy'] - gdf_limits['miny'])*pix_per_meter
    gtf_limits['maxy'] = (gtf_limits['maxy'] - gtf_limits['miny'])*pix_per_meter
    
    #width
    gdf_limits['maxx'] = (gdf_limits['maxx'] - gdf_limits['minx'])*pix_per_meter
    gtf_limits['maxx'] = (gtf_limits['maxx'] - gtf_limits['minx'])*pix_per_meter
    

    # translate coords to 0,0
    gdf_limits['minx'] = (gdf_limits['minx'] - xmin) * pix_per_meter
    gdf_limits['miny'] = (gdf_limits['miny'] - ymin) * pix_per_meter
    gdf_limits.columns = ['minx', 'miny', 'width', 'length']
    
    #same for groundtruth
    gtf_limits['minx'] = (gtf_limits['minx'] - xmin) * pix_per_meter
    gtf_limits['miny'] = (gtf_limits['miny'] - ymin) * pix_per_meter
    gtf_limits.columns = ['minx', 'miny', 'width', 'length']
    
    #be sure the limits don't go off the plot     
    gdf_limits[gdf_limits < 0] = 0
    gtf_limits[gtf_limits < 0] = 0
    
    gtf_limits['width'][gtf_limits['minx'] + gtf_limits['width'] > 200] = \
    gtf_limits['width'][gtf_limits['minx'] + gtf_limits['width'] > 200] -1
    gtf_limits['length'][gtf_limits['miny'] + gtf_limits['length'] > 200] = \
    gtf_limits['length'][gtf_limits['miny'] + gtf_limits['length'] > 200] -1

    
    gdf_limits = np.floor(gdf_limits).astype(int)
    gtf_limits =  np.floor(gtf_limits).astype(int)
    
    return(gdf_limits, gtf_limits)    
    
    
    
    

def from_raster_to_img(im_pt):
    import rasterio
    import numpy as np
    arr = rasterio.open(im_pt)
    arr = arr.read()
    arr = np.swapaxes(arr,0,1)
    arr = np.swapaxes(arr,1,2)
    arr = arr.astype('int16')
    #plt.imshow(arr)
    return arr[:, :, ::-1]
    
    
 
 





from scipy.optimize import linear_sum_assignment   
par = halo_parameters()

# get ith plot
for pl in list_plots:
    #get the RGB for plot ith
    im_pt = "./RS/" + pl + ".tif"
    im = from_raster_to_img(im_pt)
    #get coordinates of groundtruth and predictions
    gdf_limits, gtf_limits = get_vertex_per_plot(pl)
    
    #initialize rand index maxtrix GT x Detections
    R = np.zeros((gdf_limits.shape[0], gtf_limits.shape[0]))
    for obs_itc in range(gdf_limits.shape[0]):
        obs = gdf_limits.iloc[obs_itc,:].values
        for det_itc in range(gtf_limits.shape[0]):
            preds = gtf_limits.iloc[det_itc,:].values
            #calculate rand index
            R[obs_itc, det_itc] = RandNeon(obs,preds,im,par)
    #calculate the optimal matching using hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-R)

    #assigned couples
    plot_scores = R[row_ind, col_ind]
    plot_scores
    
    
    
# im_pt = "./RS/MLBS_4.tif"

# obs = gdf_limits.iloc[1,:].values
# par = halo_parameters()
# preds = gtf_limits.iloc[1,:].values
# im = from_raster_to_img(im_pt)
# RandNeon(obs,preds,im,par)

# for k in range(K):
#     par = halo_parameters()
#     xc = np.random.choice(len(xs))
#     yc = np.random.choice(len(ys))
#     wc = np.random.choice(len(ws))
#     hc = np.random.choice(len(hs))
#     det = np.array([xs[xc],ys[yc],ws[wc],hs[hc]]).astype(int)
#     Xk.append(det)
#     Scores[k] = RandNeon(GT,det,im,par)
#     par = halo_parameters()
#     Scores2[k] = RandNeon2(GT,det,im,par)
    
    