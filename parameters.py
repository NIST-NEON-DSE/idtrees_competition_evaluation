# -*- coding: utf-8 -*-
"""
@author: Dylan Stewart and Sergio Marconi
updated: 05/07/2020
    
    input variables:
        data_folder - string
        save_folder - string
    
    output:
        par         - dictionary

to use these parameters:
    from parameters import evaluation_parameters
    par = evaluation_parameters(args)
    
    args:
        
        folder paths
        'data'  : string for data folder
        'save'  : string for save folder
        
        halo parameters
        'inner' : integer (1)
        'outer' : integer (1-5)
        'edge'  : integer (2)
        
        plotting
        'plot'  : boolean (1)
        'area'  : integer size of images [r,c] [200,200] for IDTreeS
        
        
    example:
        to run with data in folder F and save in folder G with standard arguments
            args = evaluation_parameters(['--datadir','F:','--outputdir','G:'])
    
"""
import numpy as np
import argparse

def evaluation_parameters(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for IDTreeS.')
    def csv_list(string):
        return string.split(',')
    
    #Data path and save path
    parser.add_argument('--datadir', help='folder that holds the data', default='./eval', type=str)
    parser.add_argument('--outputdir',help='folder that output is saved to',default='./scores',type=str)
    
    #Halo parameters
    parser.add_argument('--inner', help='number of pixels between inner halo and ground truth', default=1, type=int)
    parser.add_argument('--outer', help='number of pixels between outer halo and ground truth', default=1, type=int)
    parser.add_argument('--edge',  help = 'initial number of pixels between edge and outer', default=1, type=int)
    
    #Output parameters
    parser.add_argument('--plot', help='plot the halos and ground truth boxes with the score',  default=1, type=int)
    parser.add_argument('--area', help='size of the plot', default=np.array([200,200]), type=int)
    
    return parser.parse_args(args)


