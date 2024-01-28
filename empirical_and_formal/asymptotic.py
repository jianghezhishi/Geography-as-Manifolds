# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:46:59 2023

@author: 86136
"""

from empirical import read_cp,read_cid_list,cut_mat_by_dis_cand,calculate_sumk,\
    output_sumk,sumk_thre_analysis,read_cover,cut_pop,cid_sumk_thre_analysis_cut,\
        size_pro_reg_plot,regression_size_pro
cp_d=read_cp('inf100_6_mat.csv')
cid_list=read_cid_list('inf100_6_mat.csv')

for maxd in [2,2.2,2.4,2.5,2.7,2.8,3]:
    for minn in range(7,80):
    
        maxn=minn+1
        print(maxd,minn)
    
        ratio=0
    
        cid_mat,cid_cand=cut_mat_by_dis_cand(cp_d,maxd,cid_list,minn,ratio,maxn)
        
        mink=4
        #-----计算给定size，给定maxd后，每个邻域中大于mink节点的度总和，用以输出分布
        cid_sumk=calculate_sumk(cid_mat,cid_cand,maxd,mink)
        title='sumk_maxd'+str(maxd)+'_mink'+str(mink)+'_minn'+str(minn)+'_maxn'+str(maxn)+'.csv'
        output_sumk(cid_sumk,title)

#---sumk分析
thre1,thre2=17.451440494252715,5.250561153723304
maxdlist=[2,2.2,2.4,2.5,2.7,2.8,3]
maxd_size_pro,maxd_data=sumk_thre_analysis(thre1,thre2,30)


#---sumk回归
file=r'test6_cover.csv'
cover=read_cover(file)
maxpop=5
poplist=cut_pop(maxpop,cover)
thre1,thre2=17.451440494252715,5.250561153723304
shell=False

window=30
log=False
maxd_size_pro,maxd_data=cid_sumk_thre_analysis_cut(poplist,thre1,thre2,cid_list,window,shell)
title='size_pro_pop'+str(maxpop)+str(shell)+'_window'+str(window)+'_log'+str(log)+'.png'
maxdlist=[2,2.2,2.4,2.5,2.7,2.8,3]
size_pro_reg_plot(maxd_size_pro,title,maxdlist,log)

#---分段回归
#print(maxd_size_pro)
title='size_pro_pop'+str(maxpop)+str(shell)+'_window'+str(window)+'.csv'

regression_size_pro(maxd_size_pro,title)