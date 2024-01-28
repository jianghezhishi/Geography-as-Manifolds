# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:43:44 2023

@author: 86136
"""

from empirical import read_cover,covisit_rank_xyoutput,reg_each_cid,plot_abr2,\
    test_iid_changerank

cover=read_cover('test6_cover.csv')
ratio=0.1
cid_xy,xlist,ylist=covisit_rank_xyoutput(cover,ratio)
cid_ab,cid_r2=reg_each_cid(cid_xy)
plot_abr2(cid_ab,cid_r2)
k=5
test_iid_changerank(cid_xy,k,rev=False,changecr=0.1,title='iid_changerate.png')
