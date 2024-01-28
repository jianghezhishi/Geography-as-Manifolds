# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:11:58 2023

@author: ASUS
"""

from propagation import *
input_grid = Grid(714,217,216,76622)
beijing_manifold = IsomapPropagationManifold(scale=18,input_grid=input_grid,center_fnid=153145,save_folder="o18")
blist = [1,2,3,5,10,30,50,70,100,200,300,400,500,700,1000,1500,2000,5000,10000,100000]
functions = ["log1p_cases ~ log_map_distance",
             "log1p_cases ~ log_manifold_distance"]
br_heatmap(beijing_manifold,blist,10,ols_function_list=functions,fig_size=(15,10),auto_vrange=False)