# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 22:25:28 2023

@author: ASUS
"""

'''
类似单点传播的思路，把选址结果作为k，看它们与备选点之间的交互
'''
import csv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from loc_choice import read_cover,get_xy
from covisit_sne import read_home,write_mat,reverse,sne_mat,output_res,plot_res_2d_simp
from math import log
def read_locc(file):
    #read the result of location choice
    locc=[]
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            locc.append(int(row[0]))
    return locc
def cut_cover_with_locc(cover,locc):
    #将未被选址结果覆盖的待覆盖点从流形中删去，防止出现离群点
    #发现在150米，限定有home个体，500sub，按访问覆盖个体情况下，所有sub500的格子都被选址结果有至少一个个体的覆盖
    covered=set()
    for cid in locc:
        covered|=cover[cid]
    cover1={}
    for cid in cover:
        if cover[cid]&covered!=set():
            cover1[cid]=cover[cid]
    return cover1
def covisit_with_home_from_cover_locck(locc,cover,func=reverse):
    #用于生成交互矩阵
    #节点是subcover中的备选门店，为n，有问题，需要替换为所有待覆盖点，暂时不改
    #交互定义为穿过两个点的人数，即cover的交集大小，但必须以选址门店为一端，即k
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
            elif node in locc or node1 in locc:
                temp.append(func(len(cover[node]&cover[node1])))
                
            else:
                temp.append(func(0))
            
        mat.append(temp)
    return mat,nodes
def plot_with_locc(locc,nodes,res,cover,title,color='cover'):
    #设置颜色按照方位还是人口来可视化
    x_list,y_list,locc_x,locc_y=[],[],[],[]
    x0,x1,y0,y1=100,-100,100,-100
    c=[]
    for i in range(len(nodes)):
        nod=nodes[i]
        lng,lat=get_xy(nod)
        co=len(cover[nod])
        if color=='cover':
            c.append(log(co,10))
        elif color=='lng':
            c.append(lng)
        elif color=='lat':
            c.append(lat)
        else:
            c.append(0)
        xy=res[i]
        x,y=xy[0],xy[1]
        x_list.append(x)
        y_list.append(y)
        if x<x0:
            x0=x
        if x>x1:
            x1=x
        if y<y0:
            y0=y
        if y>y1:
            y1=y
        if nod in locc:
            locc_x.append(x)
            locc_y.append(y)
    xs,ys=x1-x0,y1-y0
    scale=5/xs
    plt.figure(figsize=(6*2,ys*scale*2))
    plt.scatter(x_list,y_list,s=20,c=c, cmap=plt.cm.Blues, edgecolors='none')
    plt.scatter(locc_x,locc_y,color='', marker='o', edgecolors=(0.6,0.1,0.1,0.3), s=20)
    plt.colorbar()
    plt.savefig(title,dpi=150)
    plt.show()
def xyid_to_cid(xid,yid,ynum=424,xnum=639):
    if 0<=xid<xnum and 0<=yid<ynum:
        return xid*ynum+yid+1
    else:
        return ''

def get_grid(cid_xy):
    cid_list=list(cid_xy.keys())
    edge_list=[]
    while cid_list:
        cid0=cid_list[0]
        x0,y0=cid_xy[cid0]
        xid0,yid0=get_xy(cid0)
        for dxy in [(-1,0),(1,0),(0,-1),(0,1)]:
            dx,dy=dxy
            xid1,yid1=xid0+dx,yid0+dy
            cid1=xyid_to_cid(xid1,yid1)
            
            if cid1 in cid_list:
                x1,y1=cid_xy[cid1]
                edge_list.append(([x0,x1],[y0,y1]))
                
        cid_list.remove(cid0)
    return edge_list
def get_plot_input(locc,nodes,res):
    cid_xy={}
    x_list,y_list,locc_x,locc_y=[],[],[],[]
    x0,x1,y0,y1=100,-100,100,-100
    for i in range(len(nodes)):
        nod=nodes[i]
        xy=res[i]
        x,y=xy[0],xy[1]
        x_list.append(x)
        y_list.append(y)
        cid_xy[nod]=(x,y)
        if x<x0:
            x0=x
        if x>x1:
            x1=x
        if y<y0:
            y0=y
        if y>y1:
            y1=y
        if nod in locc:
            locc_x.append(x)
            locc_y.append(y)
    return (cid_xy,x_list,y_list,locc_x,locc_y,x0,x1,y0,y1)
def plot_with_locc_grid(inp,edge_list,title):
    #将地图网格在流形上画出来
    cid_xy,x_list,y_list,locc_x,locc_y,x0,x1,y0,y1=inp
    edge_list
    xs,ys=x1-x0,y1-y0
    scale=5/xs
    plt.figure(figsize=(6,ys*scale))
    plt.plot(x_list,y_list,'o',markersize=1,color='b')
    plt.plot(locc_x,locc_y,'o',markersize=1.5,color='r')
    for edge in edge_list:
        plt.plot(edge[0],edge[1],'-',linewidth=1,color='gray')
    plt.savefig(title,dpi=150)
    plt.show()


if __name__=='__main__':
    #uid_home=read_home()
    #cover=read_cover(r'F:\流形研究\20230624选址流形_深圳数据\subcover_home_500_150_visit.csv')
    #file=r'F:\流形研究\20230624选址流形_深圳数据\greedy_nodup_home_sub500_cho500_scale150_visit.csv'
    #locc=read_locc(file)
    #cover1=cut_cover_with_locc(cover,locc)
    #mat,nodes=covisit_with_home_from_cover_locck(locc,cover1)
    #title='covisit_locck_mat_home_sub500_cho500_scale150.csv'
    #write_mat(mat,nodes,title)
    #res=sne_mat(mat)
    #title='covisit_locck_emb_home_sub500_cho500_scale150.csv'
    #output_res(res,nodes,title)
    #plot_res_2d_simp(res)
    #plot_with_locc(locc,nodes,res,cover,'locck_home_sub500_cho500_scale150.png')
    #cid_xy,x_list,y_list,locc_x,locc_y,x0,x1,y0,y1=get_plot_input(locc,nodes,res)
    #edge_list=get_grid(cid_xy)
    #inp=(cid_xy,x_list,y_list,locc_x,locc_y,x0,x1,y0,y1)
    #plot_with_locc_grid(inp,edge_list,'locck_home_sub500_cho500_scale150_grid.png')
    plot_with_locc(locc,nodes,res,cover,'locck_home_sub500_cho500_scale150_pop.png')