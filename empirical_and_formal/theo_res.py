# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:16:43 2023

@author: 86136
"""

from empirical import read_cid_list,read_cp,cut_mat_by_dis_sing,count_tri,\
    output_cid_n,plot_intext,read_res,output_maxd_size_ave,est_compare_emp
import csv
def rewriter_by_dim(cid_mat,cid_n,maxd,dim_list=[(7,10),(10,20),(20,30),(30,40)],maxcal=5):
    dim_key={}
    for minn,maxn in dim_list:
        for dim in range(minn,maxn):
            dim_key[dim]=(minn,maxn)
    key_cid_n={}
    for cid in cid_n:
        dim=len(cid_mat[cid])
        if dim in dim_key:
            key=dim_key[dim]
            if key not in key_cid_n:
                key_cid_n[key]={}
            key_cid_n[key][cid]=cid_n[cid]
    for mindim,maxdim in dim_list:
        file='numtridim_maxd'+str(maxd)+'_mindim'+str(mindim)+'_maxdim'+str(maxdim)+'_maxcal'+str(maxcal)+'.csv'
        output_cid_n(key_cid_n[(mindim,maxdim)],[],file)
            
    return

file=r'inf100_6_mat.csv'
cid_list=read_cid_list(file)
cp_d=read_cp(file)
mindim=7#因为会计算到5维单纯形，为了保证统计意义，至少矩阵是8*8的，这样1个5维单纯形时就是C8-2
#后面改变了计算最大维数就可以降低
minn=30000
maxcal=5
mincal=3

#给定maxd，数单纯形
for maxd in [2]:#作为示例，这里仅运行maxd=2的情况，完整版更换为[2,2.2,2.4,2.5,2.7,2.8,3]，但可能耗费更多时间
    
    cid_mat,cid_sing=cut_mat_by_dis_sing(cp_d,maxd,cid_list,minn,1,mindim)
    print(len(cid_mat))#该维数下满足的邻域数量
    cid_n=count_tri(cid_mat,maxd,mindim,True,maxcal,mincal)
    print(len(cid_n))#确认一下前面的邻域筛选对不对
    
    
    file='tri_maxd'+str(maxd)+'_mindim'+str(mindim)+'_maxcal'+str(maxcal)+'.csv'
    output_cid_n(cid_n,[],file)#cid_sing是之前0维需要另外记录的情况，
    #但现在我们设置了mindim以保证统计意义，所以不需要另外记录了，
    rewriter_by_dim(cid_mat,cid_n,maxd)
    
file='tri_maxd2_mindim7_maxcal5.csv'
plot_intext(file,title='intext_simplex.png',k=5)

maxcal=5
mincal=3
maxd_size_ave={}
maxd_size_nd={}
for maxd in [2]:
#for maxd in [3]:
    size_d_n={}
    size_d_nd={}
    for mindim,maxdim in [(7,10),(10,20),(20,30),(30,40)]:
        
        file='numtridim_maxd'+str(maxd)+'_mindim'+str(mindim)+'_maxdim'+str(maxdim)+'_maxcal'+str(maxcal)+'.csv'
        try:
            with open(file,'r') as f:
                rd=csv.reader(f)
        except:
            pass
        else:
            dim_num,dim_nd=read_res(file)
            print(file)
            #print(dim_num)
            size_d_n[mindim]=dim_num
            size_d_nd[mindim]=dim_nd
    maxd_size_nd[maxd]=size_d_nd

title='maxd_size_nd.csv'
output_maxd_size_ave(maxd_size_nd,title,[2])

est_compare_emp()