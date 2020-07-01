# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:21:33 2020

@author: Sergio Marconi and Dylan Stewart
"""
from parameters import evaluation_parameters
from RandCrowns import RandNeon
import time
from tqdm import tqdm

"""
Created on Tue May  5 10:28:41 2020

    input variables:
        GroundTruthBox - numpy array [x y width height]
        DetectionBox   - numpy array [x y width height]
    
    output:
        evaluation - list of floats

to use this code:    

    *if you want to run the evaluation for all your plots
    - save your submission file into the submission folder as *_submission.csv 
      (e.g. ./submission/OSBS_submission.csv)
    - save your groundtruth/evaluation set in the submission folder as *_ground.csv
      (e.g. ./submission/OSBS_submission.csv)
    - make sure you have the RGB of plots in ./RS/RGB/
    
    Run:
        
    python evaluation.py
    

@author:  Dylan Stewart & Sergio Marconi
"""

# slightly modified from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def bb_intersection_over_union(boxAA, boxBB):
    # recalculate vertices for box a and b from length weight
    boxA = boxAA.copy()
    boxB = boxBB.copy()
    boxA[2] = boxA[0] + boxA[2]
    boxA[3] = boxA[1] + boxA[3]
    boxB[2] = boxB[0] + boxB[2]
    boxB[3] = boxB[1] + boxB[3]
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


def get_vertex_per_plot(pl, par):
    import rasterio
    import geopandas
    import numpy as np

    site = pl.split("_")[0]
    pix_per_meter = 10
    detection_path = par.datadir + "submission/" + site + "_submission.csv"
    ras_path = "./RS/RGB/" + pl
    # read plot raster to extract detections within the plot boundaries
    raster = rasterio.open(ras_path)

    gdf = geopandas.read_file(detection_path, bbox=raster.bounds)
    gtf = geopandas.read_file(
        par.datadir + "submission/" + site + "_ground.csv", bbox=raster.bounds
    )
    # turn WTK into coordinates within in the image
    gdf_limits = gdf.bounds
    gtf_limits = gtf.bounds

    xmin = raster.bounds[0]
    ymin = raster.bounds[1]

    # length
    gdf_limits["maxy"] = (gdf_limits["maxy"] - gdf_limits["miny"]) * pix_per_meter
    gtf_limits["maxy"] = (gtf_limits["maxy"] - gtf_limits["miny"]) * pix_per_meter

    # width
    gdf_limits["maxx"] = (gdf_limits["maxx"] - gdf_limits["minx"]) * pix_per_meter
    gtf_limits["maxx"] = (gtf_limits["maxx"] - gtf_limits["minx"]) * pix_per_meter

    # translate coords to 0,0
    gdf_limits["minx"] = (gdf_limits["minx"] - xmin) * pix_per_meter
    gdf_limits["miny"] = (gdf_limits["miny"] - ymin) * pix_per_meter
    gdf_limits.columns = ["minx", "miny", "width", "length"]

    # same for groundtruth
    gtf_limits["minx"] = (gtf_limits["minx"] - xmin) * pix_per_meter
    gtf_limits["miny"] = (gtf_limits["miny"] - ymin) * pix_per_meter
    gtf_limits.columns = ["minx", "miny", "width", "length"]

    gdf_limits = np.floor(gdf_limits).astype(int)
    gtf_limits = np.floor(gtf_limits).astype(int)

    return (gdf_limits, gtf_limits, gtf.id)


def from_raster_to_img(im_pt):
    import rasterio
    import numpy as np

    arr = rasterio.open(im_pt)
    arr = arr.read()
    arr = np.moveaxis(arr,0,-1)
    arr = arr[:,:,::-1]

    return arr


# get list of plots to evaluate
def run_segmentation_evaluation(par):
    import glob, os
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    list_plots = [os.path.basename(x) for x in glob.glob("./RS/RGB/*.tif")]

    evaluation_rand = np.array([])
    evaluation_iou = np.array([])
    itc_ids = np.array([])
    # get ith plot
    for pl in list_plots:
        # get the RGB for plot ith
        im_pt = "./RS/RGB/" + pl
#        im_pt = par.datadir + "RS/RGB/" + pl
        im = from_raster_to_img(im_pt)
        # get coordinates of groundtruth and predictions
        gdf_limits, gtf_limits, itc_name = get_vertex_per_plot(pl, par)

        # initialize rand index maxtrix GT x Detections
        R = np.zeros((gdf_limits.shape[0], gtf_limits.shape[0]))
        iou = np.zeros((gdf_limits.shape[0], gtf_limits.shape[0]))
        pbar2 = tqdm(range(gdf_limits.shape[0]), position=0,ascii=True,leave=False)
        pbar2.set_description("Processing each detection for plot "+pl)
        for obs_itc in range(gdf_limits.shape[0]):
            obs = gdf_limits.iloc[obs_itc, :].values
            for det_itc in range(gtf_limits.shape[0]):
                preds = gtf_limits.iloc[det_itc, :].values
                # calculate rand index
                R[obs_itc, det_itc] = RandNeon(obs, preds, im, par)
                # calculate the iou
                iou[obs_itc, det_itc] = bb_intersection_over_union(obs, preds)
            pbar2.update(1)
        # calculate the optimal matching using hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-R)
        pbar2.close
        if par.save == 1:
            # redo Rindex for good pairs
            pairs = np.c_[row_ind, col_ind]
            for i in range(pairs.shape[0]):
                obs = gdf_limits.iloc[pairs[i, 0], :].values
                preds = gtf_limits.iloc[pairs[i, 1], :].values
                RandNeon(obs, preds, im, par, pname=str(i) + "_" + pl)
        # assigned couples
        foo = R[row_ind, col_ind]
        plot_scores = np.zeros(gtf_limits.shape[0])
        plot_scores[col_ind] = foo

        itc_ids = np.append(itc_ids, itc_name)
        evaluation_rand = np.append(evaluation_rand, plot_scores)  # pl,plot_scores])
        # do the same for iou
        row_ind, col_ind = linear_sum_assignment(-iou)
        foo = iou[row_ind, col_ind]
        plot_scores = np.zeros(gtf_limits.shape[0])
        plot_scores[col_ind] = foo
        evaluation_iou = np.append(evaluation_iou, plot_scores)  # pl,plot_scores])
    # concatenate the three columns and save as a csv file
    task1_evaluation = np.c_[itc_ids, evaluation_rand, evaluation_iou]
    pd.DataFrame(task1_evaluation, columns=["itc_id", "rand_index", "IoU"]).to_csv(
        par.outputdir + "/task1_evaluation.csv"
    )
    return (evaluation_rand, evaluation_iou)


def run_classification_evaluation(par=None):
    """
    Created on Fri May  8 13:15:23 2020

    @author: sergiomarconi
    """
    # load test dataset
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.metrics import log_loss
    from sklearn.metrics import confusion_matrix

    # compute F1, cross entropy and confusion matrix
    preds = pd.read_csv(par.datadir + "submission/task2_submission.csv")
    obs = pd.read_csv(par.datadir + "submission/task2_ground.csv")

     # compute cross entropy
    ce_preds = preds.pivot(index="ID", columns="taxonID", values="probability")
    #get name of missing species
    missing_cols = np.setdiff1d(ce_preds.columns,obs.speciesID)
    missing_sp = pd.DataFrame(np.zeros([ce_preds.shape[0], missing_cols.shape[0]]), columns = missing_cols)
    ce_preds = pd.concat([ce_preds.reset_index(drop=True), missing_sp], axis=1)

    log_loss = log_loss(y_true = obs["speciesID"], y_pred = ce_preds, labels = ce_preds.columns)
    # get class from majority vote and compute F1 and confusion matrix
    idx = preds.groupby(["ID"])["probability"].transform(max) == preds["probability"]
    preds = preds[idx]
    evaluation_data = preds.merge(obs, left_on="ID", right_on="ID")
    confusion_matrix = confusion_matrix(
        evaluation_data["taxonID"], evaluation_data["speciesID"]
    )

    classification_report = metrics.classification_report(
        evaluation_data["taxonID"], evaluation_data["speciesID"], output_dict=True
    )

    df = pd.DataFrame(classification_report).transpose()
    df = df.rename(index={"macro avg": "macro F1", "weighted avg": "micro F1"})
    df.to_csv(par.outputdir + "/task2_evaluation.csv")
    print(df)
    return (log_loss, df)


def main(args=None):
    par = evaluation_parameters(args)

    if par.task in ["task1", "both"]:
        run_segmentation_evaluation(par)
        print(
            "Task 1 segmentation results are in "
            + par.outputdir
            + "task1_evaluation.csv"
        )
    if par.task in ["task2", "both"]:
        run_classification_evaluation(par)
        print(
            "Task 2 classification results are in "
            + par.outputdir
            + "task2_evaluation.csv"
        )
    if par.save:
        print("RandCrowns images are in " + par.outputdir + "imgs/*.png")


if __name__ == "__main__":
    main()