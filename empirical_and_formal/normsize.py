# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:53:12 2023

@author: 86136
"""

from empirical import reg_maxd_sizemustd,read_cp,read_cid_list,cut_mat_by_dis,\
    plot_size,norm_test,size_syn,output_maxd_norm,get_maxd_size_distri,\
        plot_maxd_size_distri
import numpy as np
file=r'inf100_6_mat.csv'
cp_d=read_cp(file)
cid_list=read_cid_list(file)

maxd_res1={}
for maxd in range(60,90):
    maxd=float(maxd)/10
    minn=30000
    ratio=1
    cid_mat=cut_mat_by_dis(cp_d,maxd,cid_list,minn,ratio,0)
    size=plot_size(cid_mat)
    d,p0=norm_test(size)#直接对size数据做正态检验的p值
    mu0,std0=np.mean(size),np.std(size, ddof=1)
    mu1,p1,std1=size_syn(size)
    maxd_res1[maxd]=(p0,mu0,std0,p1,mu1,std1)
    print('______________',maxd,maxd_res1[maxd])

title='maxd_norm.csv'
output_maxd_norm(maxd_res1,title)
#---对size均值和标准差对maxd进行回归
file='maxd_norm.csv'
reg_maxd_sizemustd(file)

#-----画size分布
maxd_size=get_maxd_size_distri(cp_d,cid_list)
plot_maxd_size_distri(maxd_size)