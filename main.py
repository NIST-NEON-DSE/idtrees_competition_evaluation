# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:21:33 2020

@author: Sergio Marconi and Dylan Stewart
"""
from parameters import evaluation_parameters
from getDetection import run_segmentation_evaluation
from task_2_evaluation import run_classification_evaluation

def main(args=None):
    par = evaluation_parameters(args)
        
    if par.task == "both":
        run_segmentation_evaluation(par)
        run_classification_evaluation(par)
    if par.task == "task1":
        run_segmentation_evaluation(par)
    if par.task == "task2":
        run_classification_evaluation(par)
    

if __name__ == "__main__":
    main()