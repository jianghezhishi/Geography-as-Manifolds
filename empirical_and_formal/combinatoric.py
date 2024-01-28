# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:33:27 2023

@author: 86136
"""

from empirical import read_cp,read_cid_list,cut_mat_by_dis_cand,ratio_analysis
import numpy as np
cp_d=read_cp('inf100_6_mat.csv')
cid_list=read_cid_list('inf100_6_mat.csv')
for maxd in [2.5,2.7]:

    for minn,maxn in [(7,10),(10,20),(20,30),(30,40)]:
        maxn=minn+1
        print(maxd,minn)
    
        ratio=0
        cid_mat,cid_cand=cut_mat_by_dis_cand(cp_d,maxd,cid_list,minn,ratio,maxn)
        mink=4
        #---内部连边比例分析，二级三级四级
        title='inratio_maxd'+str(maxd)+'_mink'+str(mink)+'_minn'+str(minn)+'_maxn'+str(maxn)+'.png'
        l1,l2,l3,l4,level1all,level2all,level3all,level4all=ratio_analysis(cid_mat,cid_cand,maxd,mink,title)
        print(np.average(level1all),np.average(level2all),np.average(level3all),np.average(level4all))
        