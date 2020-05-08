#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:15:23 2020

@author: sergiomarconi
"""




def run_classification_evaluation(args=None):
    
    # load test dataset
    import pandas as pd
    from sklearn import metrics
    import warnings

    import sys
    if args is None:
        args = sys.argv[1:]
    
    args = evaluation_parameters(args)
    
    # compute F1, cross entropy and confusion matrix
    preds = pd.read_csv(par.datadir+'submission/task2_submission.csv')
    obs = pd.read_csv(par.datadir+'submission/task2_ground.csv')
    
    # compute cross entropy
    
    #get class from majority vote and compute F1 and confusion matrix
    idx = preds.groupby(['ID'])['probability'].transform(max) == preds['probability']
    preds = preds[idx]
    evaluation_data = preds.merge(obs, left_on="ID", right_on="ID")
    confusion_matrix = ​metrics.confusion_matrix(evaluation_data["taxonID"], 
                                                 evaluation_data["speciesID"])
    
    classification_report = metrics.classification_report(evaluation_data["taxonID"], 
                                                          evaluation_data["speciesID"],
                                                          output_dict=True)
    
    df = pd.DataFrame(classification_report).transpose()
    df = df.rename(index={'macro avg': 'macro F1', 'weighted avg': 'micro F1' })
    df.to_csv(par.outputdir + '/task2_evaluation.csv')

    return(df)
    

