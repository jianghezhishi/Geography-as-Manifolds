# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:46:24 2023

@author: 86136
"""

from empirical import read_manifold,read_cover,data_prepare,voronoi_manifold,\
    map_poly,poly_to_point
cover=read_cover('test6_cover.csv')
res,nodes=read_manifold('test6_manifold.csv')
data_prepare(res,nodes,cover)
voronoi_manifold()
map_poly(nodes)