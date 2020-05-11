#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:28:41 2020

    input variables:
        GroundTruthBox - numpy array [x y width height]
        DetectionBox   - numpy array [x y width height]
    
    output:
        evaluation - list of floatsfloat

to use this code:    

    *if you want to run the evaluation for all your plots
    - save your submission file into the submission folder as *_submission.csv 
      (e.g. ./submission/OSBS_submission.csv)
    - save your groundtruth/evaluation set in the submission folder as *_ground.csv
      (e.g. ./submission/OSBS_submission.csv)
    - make sure you have the RGB of plots in ./RS/RGB/
    
    Run:
        
    evaluation = run_segmentation_evaluation()
    

@author:  Dylan Stewart & Sergio Marconi & ...
"""
from RandCrowns import RandNeon

# slightly modified from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def bb_intersection_over_union(boxA, boxB):
    #recalculate vertices for box a and b from length weight
    boxA[2] = boxA[0]+boxA[2]
    boxA[3] = boxA[1]+boxA[3]
    boxB[2] = boxB[0]+boxB[2]
    boxB[3] = boxB[1]+boxB[3]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_vertex_per_plot(pl,par):
    import rasterio
    import geopandas
    import numpy as np
    
    site = pl.split("_")[0]
    pix_per_meter = 10
    detection_path = par.datadir+'submission/'+site+'_submission.csv'
    ras_path = par.datadir+ 'RS/RGB/' + pl 
    #read plot raster to extract detections within the plot boundaries
    raster = rasterio.open(ras_path)

    gdf = geopandas.read_file(
        detection_path,
        bbox=raster.bounds,
    )
    gtf = geopandas.read_file(
        par.datadir+'submission/'+site+'_ground.csv',
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
    
    return(gdf_limits, gtf_limits, gtf.id)    
    
    
    
    

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
    



#get list of plots to evaluate 
def run_segmentation_evaluation(par):
    import glob, os
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment   
    list_plots = [os.path.basename(x) for x in glob.glob(par.datadir +'RS/RGB/*.tif')]
    
    evaluation_rand = np.array([])
    evaluation_iou = np.array([])
    itc_ids = np.array([])

    # get ith plot
    for pl in list_plots:
        #get the RGB for plot ith
        im_pt = par.datadir + "RS/RGB/" + pl 
        im = from_raster_to_img(im_pt)
        #get coordinates of groundtruth and predictions
        gdf_limits, gtf_limits, itc_name = get_vertex_per_plot(pl, par)
        
        #initialize rand index maxtrix GT x Detections
        R = np.zeros((gdf_limits.shape[0], gtf_limits.shape[0]))
        iou = np.zeros((gdf_limits.shape[0], gtf_limits.shape[0]))
        for obs_itc in range(gdf_limits.shape[0]):
            obs = gdf_limits.iloc[obs_itc,:].values
            for det_itc in range(gtf_limits.shape[0]):
                preds = gtf_limits.iloc[det_itc,:].values
                #calculate rand index
                R[obs_itc, det_itc] = RandNeon(obs,preds,im, par)
                #calculate the iou
                iou[obs_itc, det_itc] = bb_intersection_over_union(obs,preds)
                                                                  
        #calculate the optimal matching using hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-R)
        if par.save == 1:
            #redo Rindex for good pairs 
            pairs =  np.c_[row_ind, col_ind]
            for i in range(pairs.shape[0]):
                obs = gdf_limits.iloc[pairs[i,0],:].values
                preds = gtf_limits.iloc[pairs[i,1],:].values
                tmp = RandNeon(obs,preds,im, par, pname = str(i)+"_"+pl)
                

        #assigned couples
        plot_scores = R[row_ind, col_ind].T
        itc_ids = np.append(itc_ids, itc_name)
        evaluation_rand = np.append(evaluation_rand, plot_scores) #pl,plot_scores])
        #do the same for iou
        row_ind , col_ind = linear_sum_assignment(-iou)
        plot_scores = iou[row_ind, col_ind].T
        evaluation_iou =  np.append(evaluation_iou, plot_scores) #pl,plot_scores])
        
    #concatenate the three columns and save as a csv file
    task1_evaluation = np.c_[itc_ids, evaluation_rand, evaluation_iou]
    pd.DataFrame(task1_evaluation, columns =['itc_id', 'rand_index','IoU']).to_csv(par.outputdir + '/task1_evaluation.csv')
    return(evaluation_rand, evaluation_iou)