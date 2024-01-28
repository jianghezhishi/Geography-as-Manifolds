# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:07:51 2023

@author: 86136
"""

from empirical import read_cid_list,read_cp,plot_distri,cut_mat_by_dis,\
    town_mds_frommat,plot_town_a
file='test6_mat.csv'
cid_list=read_cid_list(file)
cp_d=read_cp(file)
maxd=2
minn=30000
ratio=1
town_mat=cut_mat_by_dis(cp_d,maxd,cid_list,minn,ratio,thre=2)
town_a=town_mds_frommat(town_mat,new=True)
title='test_local_mds.png'
plot_town_a(town_a,title,times=0.2,alpha=0.1)