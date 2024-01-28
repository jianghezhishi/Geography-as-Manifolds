# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:40:30 2023

@author: 86136
"""

from empirical import read_cover,output_cover,output_res,get_large_scale_cover,\
    mat_by_coverrate,write_mat,sne_mat,plot_pop_sne
cover=read_cover('cover_cleantraj_scale150_visit.csv')
scale=6
opt='test'

plot_pop_sne(cover,scale,opt,rate=True,inf=10,minf=False,ynum=424)

opt='inf100_'

plot_pop_sne(cover,scale,opt,rate=True,inf=100,minf=False,ynum=424)