# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:08:52 2022

@author: ASUS
"""

import csv
import datetime
from math import pi,cos,log
from gurobipy import *
import random
import matplotlib.pyplot as plt
csv.field_size_limit(1024*1024*500)
def get_xy(cid,ynum=424):
    xid=int((cid-1)/ynum)
    yid=cid-xid*ynum-1
    return xid,yid
def get_wkt(cid,xmin=113.67561783007596,xgap=0.0014608354755808878,\
            ymax=22.852485545898546,ygap=0.0013489824088780957,ynum=424):
    xid,yid=get_xy(cid)
    x0=xmin+xid*xgap
    x1=x0+xgap
    y1=ymax-yid*ygap
    y0=y1-ygap
    wkt='POLYGON(('+str(x0)+' '+str(y0)+','+str(x1)+' '+str(y0)+','\
    +str(x1)+' '+str(y1)+','+str(x0)+' '+str(y1)+'))'
    return wkt
def print_sample(file,k):
    with open(file,'r') as f:
        rd=csv.reader(f)
        for i in range(k):
            header=next(rd)
            print(header)
def test_xy_cid_func():
    with open('home_scale150_stayt0_tstep10m_n7_20.csv','r') as f:
        rd=csv.reader(f)
        with open('test_xy_cid_func.csv','w',newline='') as g:
            wt=csv.writer(g)
            for row in rd:
                if random.random()<=0.01:
                    wt.writerow([get_wkt(int(row[1]))])
def xy_to_cid(x,y,xmin=113.67561783007596,xgap=0.0014608354755808878,\
            ymax=22.852485545898546,ygap=0.0013489824088780957,ynum=424):
    xid=int((x-xmin)/xgap)
    yid=int((ymax-y)/ygap)
    return xid*ynum+yid+1
def read_uid_num():
    uid_num={}
    with open(r'F:\防疫分区\20230515项目stage2基于深圳百度数据\number_uid.csv','r') as f:
        rd=csv.reader(f)
        for row in rd:
            uid_num[row[1]]=int(row[0])
    return uid_num
def trans_200_to_150(uid_num):
    file=r'F:\基础数据\轨迹\1204sorted_withcid_200.csv'
    title=r'F:\基础数据\轨迹\1204sorted_withcid_150.csv'
    count=0
    err=0
    
    with open(file,'r') as f:
        rd=csv.reader(f)
        with open(title,'w',newline='') as g:
            wt=csv.writer(g)
            for row in rd:
                if count%2000000==0:
                    print(count,err,datetime.datetime.now())
                count+=1
                uid,t,x,y=row[2],int(row[3]),float(row[4]),float(row[5])
                cid=xy_to_cid(x,y)
                if uid in uid_num:
                    num=uid_num[uid]
                    wt.writerow([uid,num,t,x,y,cid])
                else:
                    err+=1
                
    return

def get_cover(file):
    cover={}
    count=0
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            if count%2000000==0:
                print(count,datetime.datetime.now())
            count+=1
            cid=int(row[5])
            uid=int(row[1])
            if cid in cover:
                cover[cid].add(uid)
            else:
                cover[cid]=set([uid])
                
    return cover
def output_cover(cover,title):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for cid in cover:
            wt.writerow([cid,cover[cid]])
def plot_cover_distri(cover,title,scale=5,bins=50):
    nlist=[]
    dict1={}
    for cid in cover:
        n=len(cover[cid])
        nlist.append(n)
        key=int(log(n,10)*scale)
        if key not in dict1:
            dict1[key]=0
        dict1[key]+=1
    x_list,y_list=[],[]
    for key in dict1:
        x=10**(key/scale)
        y=dict1[key]/(10**((key+1)/scale)-10**(key/scale))
        x_list.append(x)
        y_list.append(y)
    ax=plt.subplot(1,2,1)
    ax.set_xlabel('num of individuals covered')
    ax.set_ylabel('probability')
    plt.hist(nlist,bins=bins)
    ax=plt.subplot(1,2,2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('num of individuals covered')
    ax.set_ylabel('probability')
    plt.plot(x_list,y_list,'o',markersize=5,color='b')
    plt.savefig(title,dpi=150)
    plt.show()
    return
        
    
       
def get_subcover(cover,k):#生成subcover，剪枝覆盖人数小于k的格子
    subcover={}
    for cid in cover:
        if len(cover[cid])>=k:
            subcover[cid]=cover[cid]
    print('subcover ready',datetime.datetime.now())
    return subcover

def calculate_tot_con(file):
    #读取的是"结果.csv"，即loc_choice的输出
    cid_res={}
    tot=0
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            cid=int(row[0][5:])
            r=float(row[1])
            if r>0.000001:
                cid_res[cid]=r
                tot+=r
    print(tot)
    return cid_res
def res_con_int(cid_res,minr):
    #loc_choice的输出是连续近似结果，格子的取值为0到1之间的浮点，这里设置大于等于minr的为最终选址，
    res_list=[]
    for cid in cid_res:
        if cid_res[cid]>=minr:
            res_list.append(cid)
    print(len(res_list))
    return res_list
def output_res_int(res_list,title):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for cid in res_list:
            wkt=get_wkt(cid)
            wt.writerow([cid,wkt])
def greedy(cover,k):
    #贪心有重复，就是简单地把格子覆盖人数排序
    cid_num={}
    for cid in cover:
        cid_num[cid]=len(cover[cid])
    temp=sorted(cid_num.items(),key=lambda x:x[1],reverse=True)
    res_list=[]
    for i in range(k):
        res_list.append(temp[i][0])
    return res_list
def greedy_nodup(cover,k):
    """无重复贪心，每次选择删掉覆盖过的个体

    Args:
        cover (_type_): 轨迹覆盖
        k (_type_): 选点数

    Returns:
        _type_: 选点结果[cid]
    """
    #贪心无重复，覆盖人数多的格子先输出，已被它覆盖的个体将不再被计入后续格子的覆盖人数
    cover1=cover.copy()
    res_list=[]
    for i in range(k):
        cid_num={}
        for cid in cover1:
            cid_num[cid]=len(cover1[cid])
        temp=sorted(cid_num.items(),key=lambda x:x[1],reverse=True)
        cid0=temp[0][0]
        uid_set=cover1[cid0]
        res_list.append(cid0)
        del cover1[cid0]
        for cid in cover1:
            cover1[cid]=cover1[cid]-uid_set
    return res_list
def gurobi_cover(cover,k,con=True):
    """基于gurobipy的整数规划，之前没跑出结果，在toymodel上测试没出错

    Args:
        cover (dict): 轨迹覆盖 
        k (int): 选址数量 
        con (bool, optional): 是否连续. Defaults to True.

    Returns:
        res_list(list): 大于0.5的选址结果
        x(dict): 选址权重
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
    
    score=m.addVar(vtype=GRB.CONTINUOUS,name='score')
    cost=m.addVar(vtype=GRB.CONTINUOUS,name='cost')
    
    m.setObjective(score,GRB.MAXIMIZE)
    m.addConstrs(xy[uid,cid]<=mat[(uid,cid)]*x[cid] for (uid,cid) in mat)
    m.addConstrs(y[uid]<=xy.sum(uid,'*') for uid in uid_list)
    m.addConstr(cost==x.sum())
    m.addConstr(cost<=k)
    m.addConstr(score==y.sum())
    print('PROBLEM FORMULATED')
    print(datetime.datetime.now())
    
    m.optimize()
    print('SOLVED')
    print(datetime.datetime.now())
    x=m.getAttr('x',x)
    
    res_list=[]
    for cid in x:
            if x[cid]>0.5:
                res_list.append(cid)
    print(len(res_list))
    return res_list,x
def read_cover(file):
    cover={}
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            cover[int(row[0])]=eval(row[1])
    print('subcover read',datetime.datetime.now())
    return cover
def output_cover_res_by_rank(x,k,title):
    """_summary_

    Args:
        x (_type_): {cid:value}
        k (_type_): 选择点的数量
        title (_type_): 保存的title

    Returns:
        _type_: 按权重选前k个
    """
    temp={}
    
    for cid in x:
        temp[cid]=x[cid]
    temp1=sorted(temp.items(),key=lambda x:x[1], reverse=True)
    res=[]
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for i in range(k):
            cid=temp1[i][0]
            wt.writerow([cid,get_wkt(cid)])
            res.append(cid)
    return res
def calculate_score(res,cover):
    covered=set()
    for cid in res:
        covered|=set(cover[cid])
    return len(covered)
if __name__=='__main__':
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
    #file='home_scale150_stayt0_tstep10m_n7_20.csv'
    #file='home_scale150_stayt0_tstep10m_n7_20.csv'
    #test_xy_cid_func()
    #file=r'F:\基础数据\轨迹\1204staynew_150_0.csv'
    #print_sample(file,5)
    
    #file=r'F:\基础数据\轨迹\1204sorted_withcid_200.csv'
    #print_sample(file,5)
    #k=10
    #print_sample(file,k)
    uid_num=read_uid_num()
    #trans_200_to_150(uid_num)
    #file=r'F:\基础数据\轨迹\1204sorted_withcid_150.csv'
    #cover=get_cover(file)
    #title='cover_150_visit.csv'
    #output_cover(cover,title)
    #title='cover_distri_150_visit.png'
    #plot_cover_distri(cover,title)
    #k=500
    #subcover=get_subcover(cover,k)
    #title='subcover_500_150_visit.csv'
    #output_cover(subcover,title)
    
    #cover=read_cover('subcover_500_150_visit.csv')
    
    #subcover=get_subcover(cover,1000)
    #output_cover(subcover,'subcover_1000_150_visit.csv')
    #k=100
    #res_list,x=gurobi_cover(subcover,k)
    #title='con_sub1000_cho100_scale150_visit.csv'
    #res_con=output_cover_res_by_rank(x,k,title)
    
    #res=greedy_nodup(subcover,k)
    #title='greedy_nodup_sub1000_cho100_scale150_visit.csv'
    #output_res_int(res,title)
    #print(calculate_score(res_con,cover))
    #print(calculate_score(res,cover))
    '''
    cover=get_cover(file)
    k=1000
    subcover=get_subcover(cover,k)
    title='subcover_1000.csv'
    output_cover(subcover,title)
    '''
    
    #res_file='连续近似结果.csv'
    #cid_res=calculate_tot_con(res_file)
    '''
    minr=0.4
    res_list=res_con_int(cid_res,minr)
    title='连续近似结果_minr'+str(minr)+'.csv'
    output_res_int(res_list,title)
    '''
    
    #k=100
    #res_list=greedy(subcover,k)
    #title='贪心有重复.csv'
    #output_res_int(res_list,title)
    '''
    k=100
    res_list=greedy_nodup(subcover,k)
    title='贪心无重复.csv'
    output_res_int(res_list,title)
    '''
    '''
    file='subcover_1000.csv'
    subcover=read_cover(file)
    #subcover={1:['a','b','c','d'],2:['e'],3:['f'],4:['b','c','d'],5:['e','f']}
    #k=2
    k=100
    res_list=gurobi_cover(subcover,k)
    print(res_list)
    '''