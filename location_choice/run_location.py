# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:20:31 2023

@author: ASUS
"""

from location import *

shenzhen = LocationGrid(113.67561783007596,114.60880792079337,\
22.28129833936937,22.852485545898546,gridscale=150)
shenzhen_manifold = LocationManifold(shenzhen,"cover_cleantraj_scale150_visit.csv",save_folder="results1")

shenzhen_manifold.set_params(subc=1,subu=1,scale=6)

shenzhen_manifold.location_choice("greedy",k=300)
shenzhen_manifold.update("coverrate")
shenzhen_manifold.visualize()

distribute_analysis(shenzhen_manifold)

map_visualization_latlon(shenzhen_manifold,shp_file=r'E:\基础数据\行政区划\深圳乡镇街道\shenzhen.shp')