# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:26:42 2023

@author: 86136
"""

from empirical import cut_mat_by_dis_sing,count_tri,output_cid_n,plot_intext,\
    read_cp,read_cid_list
cp_d=read_cp('inf100_6_mat.csv')
cid_list=read_cid_list('inf100_6_mat.csv')
maxd=2
minn=30000
ratio=0.1
cid_mat,cid_sing=cut_mat_by_dis_sing(cp_d,maxd,cid_list,minn,ratio,mindim=10)
cid_n=count_tri(cid_mat,maxd,mindim=10,deep=True,maxcal=5,mincal=3)
output_cid_n(cid_n,cid_sing,'test_simplex.csv')
plot_intext('test_simplex.csv',title='intext_simplex.png',k=5,maxcal=5)
