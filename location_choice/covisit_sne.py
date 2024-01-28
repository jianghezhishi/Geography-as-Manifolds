# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:12:55 2023

@author: ASUS
"""

import csv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from loc_choice import read_cover,get_xy
def read_home():
    uid_home={}
    with open(r'F:\流形研究\20230624选址流形_深圳数据\home_scale150_stayt0_tstep10m_n7_20.csv','r') as f:
        rd=csv.reader(f)
        for row in rd:
            uid_home[int(row[0])]=int(row[1])
    return uid_home
def reverse(x,inf=10):
    if x>0:
        return 1/x
    else:
        return inf
def covisit_withhome_from_cover(cover,func=reverse):
    """使用cover交集生成距离矩阵

    Args:
        cover (dict): cover字典{cid:set(uid)}，不同尺度、是否剪枝都可以
        func (function, optional): 交互转距离的函数. Defaults to reverse.默认是倒数

    Returns:
        mat(list(list)): n*n距离矩阵，嵌套的list
        nodes(list): [cid]矩阵列标签
    """
    #用于生成交互矩阵
    #节点是subcover中的备选门店
    #交互定义为穿过两个点的人数，即cover的交集大小
    #交互人数转为距离的方式，默认倒数
    nodes=list(cover.keys())
    mat=[]
    step=int(len(nodes)/20)
    count=0
    for node in nodes:
        if count%step==0:
            print(count)
        count+=1
        temp=[]
        for node1 in nodes:
            if node1==node:
                temp.append(0)
            else:
                temp.append(func(len(cover[node]&cover[node1])))
        mat.append(temp)
    return mat,nodes

def write_mat(mat,nodes,title):
    """把matrix写成文件

    Args:
        mat (list(list)): 距离矩阵
        nodes (_type_): 表头
        title (_type_): 文件名
    """
    
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        wt.writerow([nodes])
        for row in mat:
            wt.writerow([row])
    return

def sne_mat(mat,nc=2,perp=30,rs=0):
    """将距离矩阵输入sne，得到嵌入坐标

    Args:
        mat (list(list)): 距离矩阵[[dist]]
        nc (int, optional): 嵌入维度. Defaults to 2.
        perp (int, optional): sne的参数perplexity. Defaults to 30.
        rs (int, optional): 随机参数. Defaults to 0.

    Returns:
        x_embedded(ndarray): 嵌入坐标
    """
    x_embedded = TSNE(n_components=nc, metric='precomputed',init='random',\
                  random_state=rs,perplexity=perp).fit_transform(np.array(mat))
    return x_embedded

def output_res(res,nodes,title):
    """保存嵌入坐标

    Args:
        res (_type_): 嵌入坐标
        nodes (_type_): 表头
        title (_type_): 文件名
    """
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for i in range(len(nodes)):
            wt.writerow([nodes[i],res[i]])
    return

def plot_res_2d_simp(res):
    ax=plt.subplot()
    x_list,y_list=[],[]
    x0,x1,y0,y1=100,-100,100,-100
    for xy in res:
        x,y=xy[0],xy[1]
        if x<x0:
            x0=x
        if x>x1:
            x1=x
        if y<y0:
            y0=y
        if y>y1:
            y1=y
        x_list.append(x)
        y_list.append(y)
    xs,ys=x1-x0,y1-y0
    scale=5/xs
    plt.figure(figsize=(5,ys*scale))
    plt.plot(x_list,y_list,'o',markersize=3,color='b')
    plt.show()
        
if __name__=='__main__':
    #uid_home=read_home()
    #cover=read_cover(r'F:\流形研究\20230624选址流形_深圳数据\subcover_home_500_150_visit.csv')
    #mat,nodes=covisit_withhome_from_cover(cover)
    #title='covisit_mat_home_sub500_scale150.csv'
    #write_mat(mat,nodes,title)
    
    #res=sne_mat(mat)
    #nodes=list(cover.keys())
    #title='covisit_emb_home_sub500_scale150.csv'
    #output_res(res,nodes,title)
    plot_res_2d_simp(res)