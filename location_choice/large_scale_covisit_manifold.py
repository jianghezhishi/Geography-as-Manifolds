# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:48:37 2023

@author: ASUS
"""

'''
与locck相比，允许k定义为subcover的key或其他集合，称为关键点集合

共同访问或覆盖比例计算基于关键点集合覆盖的个体集合

k是关键点，k内部是单位矩阵（这么干有耍赖的嫌疑，让潜在选址之间的距离都不会太近）
（但这样设计后，关键点之间的距离由它们的共同交互对象决定，覆盖都比较高的时候，两个关键点就会近）

N是大网格，左上右下是k和N的单位矩阵，左下右上是n*k和转置
'''
from loc_choice import read_cover,get_xy
from covisit_locck_sne import cut_cover_with_locc,read_locc
from covisit_sne import reverse,covisit_withhome_from_cover,sne_mat,output_res
from math import pi,cos,log
import csv
import matplotlib.pyplot as plt
#从sub定义关键点时，也需要cut，因为subcover只有关键点为key，需要cut cover
def get_large_scale_cover(cover,scale,ynumls):
    """生成大尺度cover字典

    Args:
        cover (_type_): 原来的{cid:cover}没有剪枝过
        scale (_type_): 把多少个格子捏成一个
        ynumls (_type_): 大尺度的ynum，后缀ls

    Returns:
        _type_: 大尺度的cover字典
    """
    #从cover生成交互
    #cover是基于有效，有home个体
    #将cover关系聚合到大尺度，以便后续分析，加快矩阵计算
    lscover={}#large scale cover
    for cid in cover:
        xid,yid=get_xy(cid)
        xidls,yidls=int(xid/scale),int(yid/scale)
        cidls=xidls*ynumls+yidls+1
        if cidls not in lscover:
            lscover[cidls]=set()
        lscover[cidls]|=cover[cid]
    return lscover

def covisit_2cover(subcover,coverls):
    """尝试输出n+k的交互矩阵

    Args:
        subcover (_type_): _description_
        coverls (_type_): _description_

    Returns:
        _type_: _description_
    """
    cid_list=[]
    mat=[]
    for cid in subcover:
        cid_list.append(cid)
        row=[]
        for cid1 in subcover:
            if cid1 !=cid:
                row.append(reverse(0))
            else:
                row.append(0)
        for cidls in coverls:
            inter=len(coverls[cidls]&subcover[cid])
            row.append(reverse(inter))
        mat.append(row)
    return cid_list,mat

def plot_with_locc(locc,nodes,res,cover,title,color='cover'):
    """把选址画在流形上

    Args:
        locc (list): 选址结果
        nodes (list): 流形学习嵌入的节点列表
        res (ndarray): embedding坐标
        cover (dict): {cid:set(uid)}轨迹覆盖，用于画人口热力
        title (str): 输出标题
        color (str, optional): 节点颜色选项，设置颜色按照方位还是人口来可视化。cover则按人口热力画，lat按纬度，lon按经度. Defaults to 'cover'.
    """
    print('start')
    x_list,y_list,locc_x,locc_y=[],[],[],[] # 输入数据
    x0,x1,y0,y1=100,-100,100,-100
    c=[] # 颜色
    maxnum,minnum=-100,100000 # 颜色最大最小值
    for i in range(len(nodes)): # 读取数据
        nod=nodes[i]
        lng,lat=get_xy(nod)
        co=len(cover[nod])
        if color=='cover':
            num=log(co,10)
            #num=co
            c.append(num)
        elif color=='lng':
            c.append(lng)
            num=lng
        elif color=='lat':
            c.append(lat)
            num=lat
        else:
            c.append(0)
            num=0
        if num<minnum:
            minnum=num
        if num>maxnum:
            maxnum=num
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
    print(len(locc_x))
    
    # 计算合理的figsize
    xs,ys=x1-x0,y1-y0
    scale=5/xs
    plt.figure(figsize=(6*2,ys*scale*2))
    
    plt.scatter(x_list,y_list,s=10,c=c, cmap=plt.cm.Blues, edgecolors='none')
    plt.scatter(locc_x,locc_y, marker='o', edgecolors=(0.9,0.7,0.1,0.7), s=20) # 圈出选址结果
    plt.colorbar(cmap=plt.cm.Blues,label='Logarithm of population based on base 10')
    plt.clim(minnum,maxnum)
    
    plt.savefig(title,dpi=150)
    plt.show()
    return

def mat_by_coverrate(cover,func=reverse):
    """生成基于覆盖率的矩阵，覆盖率定义为两个节点重复覆盖用户的比例，目标是

    Args:
        cover (dict): 轨迹覆盖
        func (function, optional): 距离函数. Defaults to reverse.

    Returns:
        _type_: _description_
    """
    #覆盖率非对称，使用两边算出来覆盖率的较大者
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
                num=len(cover[node]&cover[node1])/min(len(cover[node]),len(cover[node1]))
                temp.append(func(num))
        mat.append(temp)
    return mat,nodes
def read_manifold(file):
    """读流形学习嵌入

    Args:
        file (str): 文件名

    Returns:
        res: 嵌入结果
        nodes: 节点列表
        
    """
    res,nodes=[],[]
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            temp=row[1][1:-1]
            xy=[]
            flag=0
            for it in temp.split(' '):
                try:
                    num=float(it)
                except:
                    pass
                else:
                    flag+=1
                    xy.append(num)
            if flag==2:
                nodes.append(int(row[0]))
                res.append(xy)
            else:
                print('err')
                
    return res,nodes
    
    


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
    
    scale=6#600m*600m
    ynumls=int(ynum/scale)+1
    '''
    cover=read_cover('cover_cleantraj_scale150_visit.csv')
    subcover=read_cover('subcover300_cleantraj_scale150_visit.csv')#关键点为sub300的备选点
    cover_cut=cut_cover_with_locc(cover,list(subcover.keys()))
    coverls=get_large_scale_cover(cover_cut,scale,ynumls)
    '''
    
    # 从文件读取大尺度轨迹覆盖
    #coverls=read_cover('coverls_cleantraj_scale900_visit.csv')
    
    # mat,nodes=covisit_withhome_from_cover(coverls)
    '''
    # 流形学习
    res=sne_mat(mat)
    title='ls_manifold_scale900.csv'
    output_res(res,nodes,title)
    
    # 读取选址结果
    locc=read_locc('ls_clean_greedy_nodup_home_cho300_scale900_visit.csv')
    '''
    
    # 画图，把选址结果画在流形上
    # title='ls_manifold_scale900.png'
    # plot_with_locc(locc,nodes,res,coverls,title,'cover')
    # mat,nodes=mat_by_coverrate(coverls)
    '''
    res=sne_mat(mat)
    title='ls_manifold_coverrate_scale900.csv'
    output_res(res,nodes,title)
    '''
    
    # 读取最小全覆盖选址结果
    locc=read_locc('ls_allcover_scale900.csv')
    res,nodes=read_manifold('ls_manifold_coverrate_scale900.csv')
    title='ls_manifold_coverrate_allcover_scale900.png'
    plot_with_locc(locc,nodes,res,coverls,title,'cover')
    
    


