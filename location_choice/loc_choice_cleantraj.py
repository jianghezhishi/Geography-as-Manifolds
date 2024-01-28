# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 22:01:27 2023

@author: ASUS
"""

'''
基于清洗后深圳百度数据
需要读取业硕罡成做的home detection结果
从访问轨迹生成cover需要重新写
基于有home的个体，做cover，然后剪枝，选址
以选址结果为核心节点k，将所有格子按大尺度聚合作为n，构造n*k矩阵，总共是(n+k)*(n+k)
交互定义为共同访问人数，距离定义多种函数
对n*k矩阵做流形，把选址结果放上去
改为以subcover为k，把选址结果放上去
'''
import datetime
from gurobipy import *
from loc_choice import get_xy,get_wkt,output_cover,get_subcover,greedy_nodup,\
gurobi_cover,output_cover_res_by_rank,calculate_score,read_cover,output_res_int
import csv
from large_scale_covisit_manifold import get_large_scale_cover
from math import pi,cos,log
def read_uid_home(home_file,uid_file):
    #读取个体集合为filtered_users
    uid_list=[]
    with open(uid_file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        for row in rd:
            uid_list.append(row[0])
    print('uid read')
    
    uid_home={}
    with open(home_file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        for row in rd:
            
            uid_home[row[0]]=int(row[1])
    print('home read')
    uid_home1={}
    for uid in uid_list:
        uid_home1[uid]=uid_home[uid]
    return uid_home1

def get_cover_cleantraj(file,uid_home):
    """生成cover字典，使用清洗过后的访问轨迹

    Args:
        file (str): 访问轨迹
        uid_home (dict): 筛选过后的uid:home

    Returns:
        dict: cover字典
    """
    cover={}
    uid_set=set(uid_home.keys())
    count=0
    with open(file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        for row in rd:
            if count%2000000==0:
                print(count)
            count+=1
            uid=row[0]
            cid=int(row[4])
            if cid not in cover:
                cover[cid]=set()
            cover[cid].add(uid)
    print(len(cover))
    cover1={}
    for cid in cover:
        temp=cover[cid]&uid_set
        if len(temp)>0:
            cover1[cid]=temp
    return cover1

def gurobi_cover_mink(cover,con=True):
    """gurobi选址

    Args:
        cover (dict): 轨迹覆盖
        con (bool, optional): True连续近似，False整数规划. Defaults to True.

    Returns:
        x(dict): {cid:choice}每个格子被选址的情况，连续就是浮点
    """
    uid_cid={}
    for cid in cover:
        for uid in cover[cid]:
            if uid in uid_cid:
                uid_cid[uid].add(cid)
            else:
                uid_cid[uid]=set([cid])
    print(len(uid_cid))
    uid_list=list(uid_cid.keys())
    cid_list=list(cover.keys())
    mat={}
    for uid in uid_list:
        for cid in uid_cid[uid]:
            mat[(uid,cid)]=1
    print('DATA LOADED')
    print(datetime.datetime.now())
    
    m=Model('COVER')
    if con:
        x=m.addVars(cid_list,vtype=GRB.CONTINUOUS,name='x')
        y=m.addVars(uid_list,vtype=GRB.CONTINUOUS,name='y')
        m.addConstrs(x[cid]<=1 for cid in cid_list)
        m.addConstrs(x[cid]>=0 for cid in cid_list)
        m.addConstrs(y[uid]>=0 for uid in uid_list)
        m.addConstrs(y[uid]<=1 for uid in uid_list)
        xy=m.addVars(mat.keys(),vtype=GRB.CONTINUOUS,name='xy')
    else:
        x=m.addVars(cid_list,vtype=GRB.BINARY,name='x')
        y=m.addVars(uid_list,vtype=GRB.BINARY,name='y')
        xy=m.addVars(mat.keys(),vtype=GRB.BINARY,name='xy')
    
    cost=m.addVar(vtype=GRB.CONTINUOUS,name='cost')
    
    m.setObjective(cost,GRB.MINIMIZE)
    m.addConstrs(xy[uid,cid]<=mat[(uid,cid)]*x[cid] for (uid,cid) in mat)
    m.addConstrs(y[uid]<=xy.sum(uid,'*') for uid in uid_list)
    m.addConstrs(y[uid]>=0.99 for uid in uid_list)
    m.addConstr(cost==x.sum())
    
    print('PROBLEM FORMULATED')
    print(datetime.datetime.now())
    
    m.optimize()
    print('SOLVED')
    print(datetime.datetime.now())
    x=m.getAttr('x',x)
    
    tot=0
    for cid in x:
        tot+=x[cid]
        
    print(tot)
    return tot,x

    

if __name__=='__main__':
    #格子定义
    xmin, xmax, ymin, ymax =113.67561783007596,114.60880792079337,\
    22.28129833936937,22.852485545898546#深圳最大最小经纬度
    r=6371*1000
    ymid=(ymin+ymax)/2
    r1=r*cos(ymid/180*pi)
    scale=150
    xgap=scale/r1/pi*180
    ygap=scale/r/pi*180
    xnum=int((xmax-xmin)/xgap)+1
    ynum=int((ymax-ymin)/ygap)+1
    
    scale=6 # 格网的尺度 
    ynumls=int(ynum/scale)+1 # 大尺度下的ynum，后缀ls
    home_file=r'../data/Shenzhen/user_home.csv'
    uid_file=r'../data/Shenzhen/filtered_users.csv'
    # 得到有效用户
    uid_home=read_uid_home(home_file,uid_file) 
    
    file=r'../data/Shenzhen/traj_processed.csv'
    cover=get_cover_cleantraj(file,uid_home) # 生成cover
    title='cover_cleantraj_scale150_visit.csv'
    output_cover(cover,title) # 保存cover，第一列是cid，第二列是集合
    
    # 从文件读取cover
    #cover=read_cover('cover_cleantraj_scale150_visit.csv')
    # 剪枝，剪掉小于500的格子
    subcover=get_subcover(cover,500)
    output_cover(subcover,'subcover500_cleantraj_scale150_visit.csv')
    # 剪掉小于100的格子
    #subcover=get_subcover(cover,100)
    #output_cover(subcover,'subcover100_cleantraj_scale150_visit.csv')
    
    # 读取cover并剪枝
    #cover=read_cover('subcover100_cleantraj_scale150_visit.csv')
    #subcover=get_subcover(cover,300)
    #output_cover(subcover,'subcover300_cleantraj_scale150_visit.csv')
    
    #测试整数规划是否能跑动
    #subcover=get_subcover(cover,1000)
    k=20
    res_list,x=gurobi_cover(subcover,k,False)#291备选，选20个，算了45s；算的慢可能是个体数量太多
    print(x)
    # 优化方案：访问情况一样的个体进行合并，增加一个个体权重
    
    # 连续近似结果
    k=1000
    res_list,x=gurobi_cover(subcover,k)#sub100,18757备选点，选址1000个，提高覆盖率;sub300,5834
    title='clean_con_home_sub300_cho1000_scale150_visit.csv'
    # 输出前k个选点
    res_con=output_cover_res_by_rank(x,k,title)
    
    
    
    # 无重复贪心
    res=greedy_nodup(subcover,k) # [cid]
    title='clean_greedy_nodup_home_sub300_cho1000_scale150_visit.csv'
    output_res_int(res,title)
    # 测试贪心和连续近似效果哪个好
    print(calculate_score(res_con,subcover))
    print(calculate_score(res,subcover))
    
    
    
    # 涉及大尺度的测试
    coverls=get_large_scale_cover(cover,scale,ynumls) 
    
    # 计算被coverls统计的个体总量
    uid_set=set()
    for cid in coverls:
        uid_set|=coverls[cid]
    print(len(uid_set))
    
    # 连续近似，有效个体，选300点，6级（900米）尺度测试
    k=300
    res_list,x=gurobi_cover(coverls,k)#2156备选
    #print(x)
    title='ls_clean_con_home_cho300_scale900_visit.csv'
    res_con=output_cover_res_by_rank(x,k,title)
    
    # 大尺度无重复贪心测试
    res=greedy_nodup(coverls,k)
    title='ls_clean_greedy_nodup_home_cho300_scale900_visit.csv'
    output_res_int(res,title)
    print(calculate_score(res_con,coverls))
    print(calculate_score(res,coverls))
    
    
    
    output_cover(coverls,'coverls_cleantraj_scale900_visit.csv')
    # coverls=read_cover('coverls_cleantraj_scale900_visit.csv')
    # 覆盖所有个体，最小化k
    tot,x=gurobi_cover_mink(coverls)
    # 优化：减少个体数量，删除稀疏格子（比如5个），或随机筛选用户
    tot=1426
    output_cover_res_by_rank(x,tot,'ls_allcover_scale900.csv')
    


    