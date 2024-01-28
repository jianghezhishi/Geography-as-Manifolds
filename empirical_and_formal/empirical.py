# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:47:10 2023

@author: ASUS
"""

import imageio
import csv
from math import cos,pi,log,e
import math
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.manifold import TSNE
from networkx import Graph,shortest_path_length,NetworkXNoPath
import networkx as nx
from itertools import combinations
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.patches as patches
from sklearn.neighbors import NearestNeighbors
from shapely.wkt import loads,dumps
from scipy.optimize import minimize
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi, convex_hull_plot_2d
csv.field_size_limit(1024*1024*500)
'''
栅格操作
'''
def get_xy(cid,ynum=424):
    xid=int((cid-1)/ynum)
    yid=cid-xid*ynum-1
    return xid,yid
'''
cover字典相关
'''
def read_cover(file):
    cover={}
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            cover[int(row[0])]=eval(row[1])
    print('subcover read',datetime.datetime.now())
    return cover
def output_cover(cover,title):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for cid in cover:
            wt.writerow([cid,cover[cid]])
    return


'''
流形学习
'''
def output_res(res,nodes,title):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for i in range(len(nodes)):
            wt.writerow([nodes[i],res[i]])
    return
def reverse(x,inf=10):
    if x>0:
        return 1/x
    else:
        return inf
def get_large_scale_cover(cover,scale,ynumls,ynum=424):
    #从cover生成交互
    #cover是基于有效，有home个体
    #将cover关系聚合到大尺度，以便后续分析，加快矩阵计算
    lscover={}#large scale cover
    for cid in cover:
        xid,yid=get_xy(cid,ynum)
        xidls,yidls=int(xid/scale),int(yid/scale)
        cidls=xidls*ynumls+yidls+1
        if cidls not in lscover:
            lscover[cidls]=set()
        lscover[cidls]|=cover[cid]
    return lscover

def mat_by_coverrate(cover,func=reverse,inf=100):
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
                temp.append(func(num,inf))
        mat.append(temp)
    return mat,nodes
def covisit_withhome_from_cover(cover,func=reverse):
    #用于生成交互矩阵——OD形式，即共同访问
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
def minflow_mat(cover,minf,func=reverse,inf=100):
    #对cover剪枝后，再计算coverrate的矩阵，是用于分析位序规模的
    #因为在输入的空间交互中，有些流量很小，没有统计意义，需要社区，所以进行剪枝
    #覆盖率非对称，使用两边算出来覆盖率的较大者
    #计算覆盖率时，仅保留交集大于阈值的
    nodes=list(cover.keys())
    mat=[]
    print(len(nodes))
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
                f=len(cover[node]&cover[node1])
                if f>=minf:
                    num=f/min(len(cover[node]),len(cover[node1]))
                    temp.append(func(num,inf))
                else:
                    temp.append(inf)
        mat.append(temp)
    return mat,nodes
def write_mat(mat,nodes,title):
    
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        wt.writerow([nodes])
        for row in mat:
            wt.writerow([row])
    return
def sne_mat(mat,nc=2,perp=30,rs=0):
    x_embedded = TSNE(n_components=nc, metric='precomputed',init='random',\
                  random_state=rs,perplexity=perp).fit_transform(np.array(mat))
    return x_embedded


def plot_pop_sne(cover,scale,opt,rate=False,inf=10,minf=False,ynum=424):
    if scale!=1:#对于sub的cover，不需要做大尺度，因为大尺度会改变sub性质，因此scale可能为1
        ynumls=int(ynum/scale)+1
        lscover=get_large_scale_cover(cover,scale,ynumls,ynum)
    else:
        lscover=cover
    title=opt+str(scale)+'_cover.csv'
    output_cover(lscover,title)
    print('lscover ready')
    if minf==False:
        if rate:
            mat,nodes=mat_by_coverrate(lscover,reverse,inf)
        else:
            mat,nodes=covisit_withhome_from_cover(lscover)
    else:
        mat,nodes=minflow_mat(lscover,minf,reverse,inf)
    print('mat ready')
    title=opt+str(scale)+'_mat.csv'
    write_mat(mat,nodes,title)
    res=sne_mat(mat)
    print('sne ready')
    title=opt+str(scale)+'_manifold.csv'
    output_res(res,nodes,title)
    title=opt+str(scale)+'_manifold.png'
    plot_with_locc([],nodes,res,lscover,title)
    return res,lscover,mat,nodes

'''
选址流形人口变形
'''
def read_manifold(file):
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

def read_locc(file):
    #read the result of location choice
    locc=[]
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            locc.append(int(row[0]))
    return locc

def create_test_trans():
    test_res,test_nodes,test_cover=[],[],{}
    cid=0
    for i in range(5):
        for j in range(5):
            test_res.append([float(i),float(j)])
            cid+=1
            test_nodes.append(cid)
            test_cover[cid]=[0]*(5-j)
    return test_res,test_nodes,test_cover
            
def data_prepare(res,nodes,cover):
    with open('carto_input.csv','w',newline='') as f:
        wt=csv.writer(f)
        for i in range(len(nodes)):
            cid=nodes[i]
            xy=res[i]
            if cid in cover:
                wt.writerow([cid]+xy+[len(cover[cid])])
    return
def voronoi_manifold():
    # Step 1: Read the CSV file
    data = pd.read_csv('carto_input.csv')
    # Adjusting the column names based on the CSV structure
    points = [Point(xy) for xy in zip(data[data.columns[1]], data[data.columns[2]])]
    gdf_points = gpd.GeoDataFrame(data, geometry=points)
    
    # Calculate the convex hull
    convex_hull = gdf_points.unary_union.convex_hull
    
    # Calculate the Voronoi diagram
    vor = Voronoi(data[[data.columns[1], data.columns[2]]].to_numpy())
    
    # Clip Voronoi regions to match the convex hull
    regions = []
    #tag=[]
    temp=0
    for region in vor.regions:
        '''
        polygon = Polygon([vor.vertices[i] for i in region]).buffer(0.001)
        regions.append(polygon.intersection(convex_hull))
        '''
        if not -1 in region:
            polygon = Polygon([vor.vertices[i] for i in region])
            regions.append(polygon.intersection(convex_hull))
        '''
        else:
            tag.append(temp)
        '''
        temp+=1
    print(len(data))
    print(len(regions))
    #gdf_filtered = gdf_voronoi[~gdf_voronoi['geometry'].apply(lambda geom: any([coord == -1 for x, y in geom.exterior.coords for coord in [x, y]]))]
    
    # Create a GeoDataFrame with the Voronoi polygons
    tag=[]
    reg1=[]
    for i in range(len(points)):
        if i%200==0:
            print(i)
        p=points[i]
        flag=0
        for pol in regions:
            if p.within(pol.buffer(0.001)):
                flag+=1
                temp=pol
        if flag==1:
            reg1.append(temp)
        elif flag==0:
            tag.append(i)
        else:
            print(p.coords)
            tag.append(i)
    data.drop(tag,inplace=True)
    print(len(reg1))
    gdf_voronoi = gpd.GeoDataFrame(data,geometry=reg1)
    
    print(gdf_voronoi.head())
    
    # Save the Voronoi polygons to a shp file
    output_path = "voronoi_polygons_cid.shp"
    gdf_voronoi.to_file(output_path)
    return
def map_poly(nodes):
    #使用get_xy
    ynum=int(424/6)+1
    map_res=[]
    cid_pop={}
    with open('carto_input.csv','r') as f:
        rd=csv.reader(f)
        header=next(rd)
        for row in rd:
            cid=int(row[0])
            cid_pop[cid]=int(row[3])
    with open('map_poly.csv','w',newline='') as f:
        wt=csv.writer(f)
        wt.writerow(['cid','pop','wkt'])
        
        for cid in nodes:
            x,y=get_xy(cid,ynum)
            y=-y
            map_res.append([float(x),-float(y)])
            if cid in cid_pop:
                wkt='POLYGON(('+str(x-0.5)+' '+str(y-0.5)+','+str(x+0.5)+' '+str(y-0.5)+\
                ','+str(x+0.5)+' '+str(y+0.5)+','+str(x-0.5)+' '+str(y+0.5)+','\
                +str(x-0.5)+' '+str(y-0.5)+'))'
                wt.writerow([cid,cid_pop[cid],wkt])
    return map_res
def poly_to_point(file,title):
    #使用arcmap对上一函数的输出进行cartogram，基于gastner2004的算法，
    #本函数将所得结果重新转化为流形上的坐标
    with open(file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        with open(title,'w',newline='') as g:
            wt=csv.writer(g)
            for row in rd:
                cid=int(float(row[2]))
                wkt=row[-1]
                x,y=loads(wkt).centroid.coords[0]
                wt.writerow([cid,'['+str(x)+' '+str(y)+']'])
    return

'''
可视化
'''


def plot_with_locc(locc,nodes,res,cover,title,color='cover'):
    #设置颜色按照方位还是人口来可视化
    print('start')
    x_list,y_list,locc_x,locc_y=[],[],[],[]
    x0,x1,y0,y1=100,-100,100,-100
    c=[]
    maxnum,minnum=-100,100000
    for i in range(len(nodes)):
        nod=nodes[i]
        lng,lat=get_xy(nod)
        if nod in cover:
            co=len(cover[nod])
            if color=='cover':
                num=log(co,10)
                #num=co
                c.append(num)
            elif color=='cover nonlog':
                c.append(co)
                num=co
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
    xs,ys=x1-x0,y1-y0
    scale=5/xs
    plt.figure(figsize=(6*2,ys*scale*2))
    plt.scatter(x_list,y_list,s=10,c=c, cmap=plt.cm.Blues, edgecolors='none')
    if locc_x!=[]:
        #plt.scatter(locc_x,locc_y,color='', marker='o', edgecolors=(0.9,0.7,0.1,0.7), s=20)
        plt.scatter(locc_x,locc_y,color='', marker='o', edgecolors='r', s=20)
    plt.colorbar(cmap=plt.cm.Blues,label='Logarithm of population based on base 10')
    plt.clim(minnum,maxnum)
    
    plt.savefig(title,dpi=150)
    plt.show()
    return

'''
实证——局部mds
'''
'''
数据读取
'''
def read_cid_list(file):
    with open(file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
    header=eval(header[0])
    cid_list=[]
    for cid in header:
        cid_list.append(int(cid))
    return cid_list


def read_cp(file):
    with open(file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        header=eval(header[0])
        count=0
        cp_d={}
        for row in rd:
            row=eval(row[0])
            cid0=header[count]
            count+=1
            for i in range(len(row)):
                cid1=header[i]
                if cid1>=cid0:
                    cp_d[(cid0,cid1)]=row[i]
    return cp_d

def plot_distri(cp_d,k,title='distribution of d.png'):
    dict1={}
    for cp in cp_d:
        if cp_d[cp]!=0:
            d=cp_d[cp]
            key=int(log(d,10)*k)
            if key not in dict1:
                dict1[key]=0
            dict1[key]+=1
    temp=sorted(dict1.items(),key=lambda x:x[0])
    x,y=[],[]
    for it in temp:
        x.append(10**(it[0]/k))
        y.append(it[1]/len(cp_d)/(10**(it[0]+1)/k-10**(it[0]/k)))
    ax=plt.subplot()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.plot(x,y,'o',markersize=2,color='b')
    plt.savefig(title,dpi=150)
    plt.show()
    return
    '''
    mds
    '''
def cut_mat_by_dis(cp_d,maxd,cid_list,minn,ratio,thre=2):
    cid_mat={}
    count=0
    step=max(1,int(len(cid_list)/20))
    for cid in cid_list:
        if count%step==0:
            print(count/len(cid_list))
        count+=1
        cand=[]
        for cid1 in cid_list:
            cp=(min(cid,cid1),max(cid,cid1))
            if cp_d[cp]<=maxd:
                cand.append(cid1)
        if len(cand)>=minn or random.random()<=ratio:
            mat=[]
            for cid1 in cand:
                row=[]
                for cid2 in cand:
                    cp=(min(cid1,cid2),max(cid1,cid2))
                    row.append(cp_d[cp])
                mat.append(row)
            if len(mat)>thre:
                cid_mat[cid]=mat
    return cid_mat

def mds(mat):
    #print(mat)
    D=np.array(mat)
    length=len(mat)
    re= np.zeros((length, length),np.float32)
    ss = 1.0 /length ** 2 * np.sum(D ** 2)
    for i in range(length):
        for j in range(length):
            re[i, j] = -0.5 * (D[i, j] ** 2 - 1.0 / length * np.dot(D[i, :], D[i, :]) - 1.0 / length * np.dot(D[:, j], D[:, j]) + ss)
 
    A, V = np.linalg.eig(re)
    a=list(A)
    a.sort(reverse=True)
    return a


def classical_mds(D, k):
    """Apply classical MDS algorithm to reduce dimensionality.
    
    Args:
    - D: Pairwise distance matrix.
    - k: Number of dimensions for the output.
    
    Returns:
    - Reduced dimensionality data.
    """
    D=np.array(D)
    n = D.shape[0]
    
    # Create the centering matrix
    I = np.eye(n)
    H = I - 1/n * np.ones((n, n))
    
    # Create the B matrix (double centered matrix)
    B = -1/2 * np.dot(H, np.dot(D**2, H))
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    '''
    # Sort by eigenvalue in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take the square root of the eigenvalues
    L = np.diag(np.sqrt(eigvals[:k]))
    
    return np.dot(eigvecs[:, :k], L)
    '''
    a=list(eigvals)
    a.sort(reverse=True)
    return a
def town_mds_frommat(town_mat,new=False):
    town_a={}
    for town in town_mat:
        #print('--------')
        #print(town)
        
        mat=town_mat[town]
        if new:
            a=classical_mds(mat, 3)
        else:
            a=mds(mat)
        #print(a)
        town_a[town]=a
    return town_a
    '''
    mds结果输出和分析
    '''
def output_town_a(town_a,title):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for town in town_a:
            wt.writerow([town,town_a[town]])
    return
def read_town_a(file):
    town_a={}
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            town_a[int(row[0])]=eval(row[1])#这个读取函数是针对cid半径邻域写的，所以key是int
    return town_a
def a_to_xy(a_list):
    x_list=[]
    y_list=[]
    for x in range(len(a_list)):
        x_list.append(x)
        y_list.append(a_list[x]/a_list[0])
    return x_list,y_list
def a_to_end(a_list,times=0.2):
    end=0
    a=a_list[end]/a_list[0]
    while a>times:
        end+=1
        a=a_list[end]/a_list[0]
    return end
    
def plot_town_a(town_a,title,times=0.2,alpha=0.1):
    plt.figure(figsize=(10,5))
    ax=plt.subplot(1,2,1)
    ax.set_xlabel('Ranking of eigenvalues')
    ax.set_ylabel('Eigenvalues')
    ax.set_xscale('log')
    x0,y0=[3,3],[1,-1]
    plt.plot(x0,y0,'--',linewidth=1,color='r',alpha=0.2)
    for town in town_a:
        a_list=town_a[town]
        x_list,y_list=a_to_xy(a_list)
        plt.plot(x_list,y_list,'o-',linewidth=1,markersize=2,color='b',alpha=alpha)
    end_n={}
    highdim=[]
    maxl=0
    for town in town_a:
        if len(town_a[town])>maxl:
            maxl=len(town_a[town])
        end=a_to_end(town_a[town],times)
        if end not in end_n:
            end_n[end]=0
        end_n[end]+=1
        if end>3:
            highdim.append(town)
    #print(end_n)
    temp=sorted(end_n.items(),key=lambda x:x[0])
    print(temp)
    tot=0
    ax=plt.subplot(1,2,2)
    ax.set_xlabel('dimensionality')
    ax.set_ylabel('number of areas with dimension less than x')
    x,y=[0],[0]
    for i in range(maxl+1):
        x.append(i)
        if i in end_n:
            tot+=end_n[i]
        y.append(tot)
            
    '''
    for it in temp:
        tot+=it[1]
        #plt.text(it[0],0,str(tot))
        x.append(it[0])
        y.append(tot)
    '''
    plt.plot(x,y,'o-',markersize=2,linewidth=1,color='b')
    #print(end_n)
    #for end in end_n:
        
        #plt.text()
    plt.savefig(title,dpi=150)
    plt.show()
    return highdim,temp

    '''
    最短路径替换，inf分析
    '''
def compute_shortest_distances(matrix, threshold):
    n = len(matrix)
    
    # 构建图形
    G = Graph()
    for i in range(n):
        for j in range(n):
            if matrix[i][j] <= threshold:
                G.add_edge(i, j, weight=matrix[i][j])

    # 对于大于阈值的距离，计算其在图中的最短路径距离并替换
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > threshold:
                try:
                    shortest_distance = shortest_path_length(G, source=i, target=j, weight='weight')
                    matrix[i][j] = shortest_distance
                except NetworkXNoPath:  # 如果两个点之间没有路径，我们可以保持原值或设置为 np.inf 等
                    pass

    return matrix
def town_mds_frommat_path(town_mat,maxd,times=1):
    #将maxd*times之外的边用路径替换
    town_a={}
    count=0
    step=max(1,int(len(town_mat)/10))
    for town in town_mat:
        if count%step==0:
            print(count/len(town_mat))
        count+=1
        #print('--------')
        #print(town)
        
        mat=town_mat[town]
        mat1=compute_shortest_distances(mat, maxd*times)
        a=mds(mat1)
        #print(a)
        town_a[town]=a
    return town_a

def test_inf(cid_mat,inf):
    cid_mat1={}
    for cid in cid_mat:
        mat=cid_mat[cid]
        mat1=[]
        flag=0
        for row in mat:
            row1=[]
            for i in row:
                if i<=inf:
                    row1.append(0)
                else:
                    row1.append(i)
                    flag+=1
            mat1.append(row1)
        if flag>1:
            cid_mat1[cid]=mat1
    return cid_mat1
'''
实证——多尺度mds
'''
def calculate_ratio(temp):
    tot=0
    n=0
    for it in temp:
        d,x=it
        if d<=3:
            n+=x
        tot+=x
    print(n/tot)
    return n/tot
def get_remain_mat(cp_d,maxd0,maxd1):
    #将maxd0和maxd1之间的边输出为每个点的矩阵
    cid_cid1={}
    cid_set=set()
    count=0
    step=max(1,int(len(cp_d)/10))
    for cp in cp_d:
        if count%step==0:
            #print(count/len(cp_d))
            pass
        count+=1
        if maxd0<cp_d[cp]<=maxd1:
            cid_set|=set(cp)
            cid1,cid2=cp
            if cid1 not in cid_cid1:
                cid_cid1[cid1]=set([cid1])
            cid_cid1[cid1].add(cid2)
            if cid2 not in cid_cid1:
                cid_cid1[cid2]=set([cid2])
            cid_cid1[cid2].add(cid1)
    cid_list=list(cid_set)
    print(len(cid_list))
    cid_mat={}
    count=0
    step=max(1,int(len(cid_list)/20))
    for cid in cid_list:
        if count%step==0:
            #print(count/len(cid_list))
            pass
        count+=1
        temp=cid_cid1[cid]
        mat=[]
        for cida in temp:
            row=[]
            for cidb in temp:
                cp=(min(cida,cidb),max(cida,cidb))
                row.append(cp_d[cp])
            mat.append(row)
        cid_mat[cid]=mat
    return cid_mat
def step_multiscale(minratio,cp_d,maxd0,inf=100,change=0.1):
    maxd1=maxd0
    ratio=1
    while ratio>minratio and maxd1<inf:
        maxd1+=change
        maxd1=round(maxd1,2)
        cid_mat=get_remain_mat(cp_d,maxd0,maxd1)
        title=r'递归\multi_maxd0_'+str(maxd0)+'_maxd1_'+str(maxd1)
        cid_a=town_mds_frommat(cid_mat)
        output_town_a(cid_a,title+'.csv')
        highdim,temp=plot_town_a(cid_a,title+'.png',0.1,0.02)
        ratio=calculate_ratio(temp)
        print(maxd0,maxd1,ratio)
    return round(maxd1-change,2)
def multiscale_while(minratio,cp_d,maxd0,inf=100,change=0.1):
    res=[maxd0]
    maxd1=maxd0+change
    while maxd1>maxd0 and maxd1<inf:
        maxd1=step_multiscale(minratio,cp_d,maxd0)
        print('---------------------------------')
        print(maxd1,res)
        res.append(maxd1)
        maxd0=maxd1
        maxd1=maxd0+change
    return res
def plot_multiscale(res):
    #这个函数还不是读取文件的，不过res打印出来了
    x=list(range(len(res)))
    ax=plt.subplot()
    ax.set_xlabel('Layer',fontsize=16)
    ax.set_ylabel('Radius',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(x,res,'o-',markersize=3,linewidth=1,color='rosybrown')
    plt.tight_layout()
    plt.savefig('layer_radius.png',dpi=150)
    
    plt.show()


'''
理论证明——环节一，位序规模
'''
def covisit_rank(cover,ratio):
    #位序规模，是对重复覆盖率的位序规模
    ax=plt.subplot()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Cover rate')
    for cid in cover:
        if random.random()<=ratio:
            temp=[]
            for cid1 in cover:
                #x=len(cover[cid]&cover[cid1])/min(len(cover[cid]),len(cover[cid1]))
                x=len(cover[cid]&cover[cid1])/len(cover[cid])
                if x>0:
                    temp.append(x)
            if len(temp)>=10:
                temp.sort(reverse=True)
                #x=np.linspace(1,100,len(temp))
                x=list(range(len(temp)))
                plt.plot(x,temp,'-',linewidth=0.3,color='royalblue',alpha=0.05)
    plt.savefig('covisit rank.png',dpi=150)
    plt.show()
    return
def read_mat_sample(file,ratio,inf):
    #先读100inf的
    cid_rank={}
    with open(file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        header1=eval(header[0])
        tag_cid={}
        for i in range(len(header1)):
            if random.random()<=ratio:
                tag_cid[i]=header1[i]
        #print(tag_list)
        count=0
        for row in rd:
            if count in tag_cid:
                cid=tag_cid[count]
                cid_rank[cid]=[]
                row1=list(eval(row[0]))
                for num in row1:
                    if 0<num<inf:
                        cid_rank[cid].append(num)
            count+=1
    return cid_rank
def rescale_list(input_list):
    # 获取list0的最小值和最大值
    #coverrate的数值在1到100之间，取不到100，不用处理即可log，但要对一下斜率
    #主要要处理x的归一化，有些格子交互对象更多
    min_val = min(input_list)
    max_val = max(input_list)

    # 等比例放缩索引值
    list1 = [(i / (len(input_list) - 1) * 99 + 1) if len(input_list) > 1 else 50 for i in range(len(input_list))]
    
    # 线性放缩list0的每个元素值
    list2 = [(val - min_val) / (max_val - min_val) * 99 + 1 if max_val > min_val else 50 for val in input_list]

    return list1, list2

    
def plot_rank(cid_rank,title,minl=10):
    ax=plt.subplot()
    ax.set_xlabel('Normalized ranking',fontsize=16)
    ax.set_ylabel('Normalized distance',fontsize=16)
    for cid in cid_rank:
        list1=cid_rank[cid]
        list1.sort(reverse=True)
        if len(list1)>minl:
            x,y=rescale_list(list1)
            if y[0]!=y[-1]:
                plt.plot(x,y,'o-',linewidth=1,markersize=2,color='burlywood',alpha=0.003)
    plt.savefig(title,dpi=150)
    plt.show()
    return
def plot_rank_norescale(cid_rank,title,minl=10):
    ax=plt.subplot()
    ax.set_xlabel('normalized ranking')
    ax.set_ylabel('distance by cover rate')
    for cid in cid_rank:
        list1=cid_rank[cid]
        list1.sort(reverse=True)
        if len(list1)>minl:
            x=np.linspace(1,100,len(list1))
            y=list1
            plt.plot(x,y,'o-',linewidth=1,markersize=2,color='b',alpha=0.01)
    plt.savefig(title,dpi=150)
    plt.show()
    return
#位序规模回归，每个点的邻域单独回归，看参数的分布
def covisit_rank_xyoutput(cover,ratio):
    ax=plt.subplot()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Cover rate')
    cid_xy={}
    xlist,ylist=[],[]
    for cid in cover:
        if random.random()<=ratio:
            temp=[]
            for cid1 in cover:
                #x=len(cover[cid]&cover[cid1])/min(len(cover[cid]),len(cover[cid1]))
                x=len(cover[cid]&cover[cid1])/len(cover[cid])
                if 10**(-1.9)<=x:
                    temp.append(x)
            if len(temp)>=10:
                temp.sort(reverse=True)
                #x=np.linspace(1,100,len(temp))
                x=list(range(1,len(temp)+1))
                plt.plot(x,temp,'-',linewidth=0.3,color='royalblue',alpha=0.05)
                cid_xy[cid]=(x,temp)
                xlist+=x
                ylist+=temp
    plt.savefig('covisit rank.png',dpi=150)
    plt.show()
    return cid_xy,xlist,ylist
def reg_loglog(x,y,su=False):
    err=0
    x1,y1=[],[]
    for i in range(len(x)):
        if x[i]>0 and y[i]>0:
            x1.append(x[i])
            y1.append(y[i])
        else:
            err+=1
    #print('err',err)
    x=np.log(np.array(x1))
    y=np.log(np.array(y1))
    #x=np.log(np.array(x))
    #y=np.log(np.array(y))
    x1 = sm.add_constant(x)
    model = sm.OLS(y, x1)
    results = model.fit()
    if su:
        print(results.summary())
    a,b=results.params
    pa,pb=results.pvalues
    r2=results.rsquared
    ca,cb=results.conf_int()
    cb1,cb2=cb
    return a,b,pb,r2
def reg_each_cid(cid_xy):
    err=0
    cid_ab={}
    cid_r2={}
    cid_r2_p={}
    for cid in cid_xy:
        x,y=cid_xy[cid]
        a,b,pb,r2=reg_loglog(x,y)
        if pb>0.05:
            err+=1
        else:
            cid_ab[cid]=(a,b)
            cid_r2_p[cid]=r2
        cid_r2[cid]=r2
    print('err',err)
    print('ave r2',np.sum(list(cid_r2.values()))/len(cid_r2))
    print('ave r2 with sig p',np.sum(list(cid_r2_p.values()))/len(cid_r2_p))#显著部分的平均r2；从结果来看都是显著的
    return cid_ab,cid_r2

def plot_abr2(cid_ab,cid_r2):
    alist,blist,r2list=[],[],[]
    for cid in cid_ab:
        a,b=cid_ab[cid]
        alist.append(float(a))
        blist.append(float(b))
    for cid in cid_r2:
        r2=float(cid_r2[cid])
        if r2>0:
            r2list.append(r2)
    #print(r2list)
    print(np.average(r2list))
    print(np.average(alist))
    print(np.average(blist))
    plt.figure(figsize=(16,4))
    ax=plt.subplot(1,4,1)
    ax.set_xlabel('R square')
    ax.set_ylabel('n')
    plt.hist(r2list,bins=30)
    ax.text(0.01, 1.02,'a', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    
    ax=plt.subplot(1,4,2)
    ax.set_xlabel('a')
    ax.set_ylabel('n')
    plt.hist(alist,bins=30)
    ax.text(0.01, 1.02,'b', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    
    ax=plt.subplot(1,4,3)
    ax.set_xlabel('b')
    ax.set_ylabel('n')
    plt.hist(blist,bins=30)
    ax.text(0.01, 1.02,'c', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    
    ax=plt.subplot(1,4,4)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    alist1,blist1=alist.copy(),blist.copy()
    alist1.sort()
    blist1.sort()
    arange=alist1[-1]-alist1[0]
    brange=blist1[-1]-blist1[0]
    if arange>brange:
        ax.set_xlim(alist1[0],alist1[-1])
        ax.set_ylim((blist1[0]+blist1[-1])/2-arange/2,(blist1[0]+blist1[-1])/2+arange/2)
    else:
        ax.set_xlim((alist1[0]+alist1[-1])/2-brange/2,(alist1[0]+alist1[-1])/2+brange/2)
        ax.set_ylim(blist1[0],blist1[-1])
    
    #plt.axis('off')
    plt.plot(alist,blist,'o',markersize=1,color='royalblue',alpha=0.2)
    ax.text(0.01, 1.02,'b', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.savefig('reg_coverrate_rank_bycid.png',dpi=150)
    return
def output_cid_reg(cid_xy,cid_ab,title):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for cid in cid_xy:
            row=[cid,cid_xy[cid]]
            if cid in cid_ab:
                row.append(cid_ab[cid])
            wt.writerow(row)
    return

def drank_dcoverrate(ylist,step):
    k_rank={}
    maxy=ylist[0]
    for i in range(len(ylist)):
        key=int((maxy-ylist[i])/step)
        if key not in k_rank:
            k_rank[key]=[]
        k_rank[key].append(i)
    startlist=[]
    for k in k_rank:
        rank=k_rank[k]
        rank.sort()
        startlist.append(rank[0])
    return startlist
def getxy_plot_distri(res,k):
    dict1={}
    for d in res:
        key=int(log(d,10)*k)
        if key not in dict1:
            dict1[key]=0
        dict1[key]+=1
    temp=sorted(dict1.items(),key=lambda x:x[0])
    #print(temp)
    x,y=[],[]
    for it in temp:
        x.append(10**(it[0]/k))
        y.append(it[1]/len(res)/(10**((it[0]+1)/k)-10**(it[0]/k)))
    return x,y
def test_iid_changerank(cid_xy,k,rev=False,changecr=0.1,title='iid_changerate.png'):
    #分析对原始位序规模的coverrate，变化相同的dcoverrate，rank如何变化，以实现iid的中心极限定理
    #不用位序规模的分析
    #应该可以反过来，算每变化一个rank，coverrate的变化，如果是一个均值方差有限的iid即可
    
    size_res={}
    size_list=[(10,200),(200,300),(300,400),(400,500),(500,600),(600,1000),(10,1000)]
    for minn,maxn in size_list:
        
        dx,dy=[],[]
        for cid in cid_xy:
            x,y=cid_xy[cid]
            
                
            if minn<=len(y)<maxn:
                if rev:
                    y.sort()
                    y=1/np.array(y)
                    #y.sort(reverse=True)
                else:
                    y.sort(reverse=True)
                temp=drank_dcoverrate(y,changecr)
                for i in range(len(temp)-1):
                    dx.append(temp[i+1]-temp[i])
                    
                for i in range(len(y)-1):
                    if y[i]-y[i+1]>0:
                        dy.append(y[i]-y[i+1])
        size_res[(minn,maxn)]=(dx,dy)
    xn,yn=2,len(size_res)
    plt.figure(figsize=(3*xn,3*yn))
    
    for i in range(2*len(size_res)):
        
        ax=plt.subplot(yn,xn,i+1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if i%xn==0:
            ax.set_ylabel('Probability density')
        if xn==2:
            j=i//2
        else:
            j=i%len(size_list)
        size=size_list[j]
        dx,dy=size_res[size]
        
        if (xn==2 and i%2==0) or (yn==2 and i//xn==0):
            plt.ylim(10**(-3),10**3)
            plt.xlim(10**(-2.5),1)
            x,y=getxy_plot_distri(dy,k)
            if rev:
                ax.set_xlabel('Change of dist per ranking')
            else:
                ax.set_xlabel('Change of coverrate per ranking')
            plt.plot(x,y,'o',markersize=2,color='royalblue')
        else:
            plt.ylim(10**(-6),10**0)
            plt.xlim(1,10**2.2)
            x,y=getxy_plot_distri(dx,k)
            if rev:
                ax.set_xlabel('Change of ranking per '+str(changecr)+' dist')
            else:
                ax.set_xlabel('Change of ranking per '+str(changecr)+' coverrate')
            plt.plot(x,y,'o',markersize=2,color='royalblue')
        ax.text(0.01, 1.02,str(chr(96 + i+1)), transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    plt.tight_layout()
    plt.savefig(title,dpi=150)
    plt.show
            
    '''
    ax=plt.subplot()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.hist(dy,bins=100)
    '''
    return
#位序规模参数与size的关系，每个点的邻域算一个数据点
def reg_lin(x,y,su=True):
    x1 = sm.add_constant(x)
    model = sm.OLS(y, x1)
    results = model.fit()
    if su:
        print(results.summary())
    a,b=results.params
    pa,pb=results.pvalues
    r2=results.rsquared
    ca,cb=results.conf_int()
    cb1,cb2=cb
    return a,b,pb,r2
def reg_maxd_sizemustd(file):
    maxdlist,mu,std=[],[],[]
    with open(file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        header=next(rd)
        for row in rd:
            if float(row[0])>=4:
                maxdlist.append(float(row[0]))
                mu.append(float(row[2]))
                std.append(float(row[3]))
    plt.figure(figsize=(8,4))
    maxd1=np.log(np.array(maxdlist))
    ax=plt.subplot(1,2,1)
    ax.set_xlabel('log maxd')
    ax.set_ylabel('average log size')
    plt.plot(maxd1,mu,'o',markersize=2,color='royalblue')
    ax.text(0.01, 1.02,'a', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    a,b,pb,r2=reg_lin(maxd1,mu)
    
    std1=np.log(np.array(std))
    ax=plt.subplot(1,2,2)
    ax.set_xlabel('maxd')
    ax.set_ylabel('log std of log size')
    plt.plot(maxdlist,std1,'o',markersize=2,color='royalblue')
    ax.text(0.01, 1.02,'b', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    a,b,pb,r2=reg_lin(maxdlist,std1)
    
    plt.tight_layout()
    plt.savefig('maxd_mu_std.png',dpi=150)
    plt.show()
    return

#size的正态分布检验
def norm_test(data):
    d, p_value_ks = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    print(f"Kolmogorov-Smirnov检验: D = {d}, p-value = {p_value_ks}")
    #w, p_value_w = stats.shapiro(data)
    #print(f"Shapiro-Wilk检验: W = {w}, p-value = {p_value_w}")
    return d,p_value_ks
def size_syn(size,maxd=2):
    #因为maxd小的时候，会呈现半正态分布，因此需要变形
    #方法是找出可能的均值，再将大于均值的部分减去均值，做对数后进行正态检验（gpt说的，感觉不一定靠谱）
    ax=plt.subplot()
    ax.set_xlabel('log size')
    ax.set_ylabel('n')
    n, bins, patches = plt.hist(size, bins=15, edgecolor="k")
    #print(patches)
    #print(n)
    #print(bins)
    maxn=0
    maxb=0
    for i in range(len(n)):
        if n[i]>maxn:
            maxn=n[i]
            maxb=(bins[i]+bins[i+1])/2
    print(maxb)
    
    
    size1=[]
    for s in size:
        if s>=maxb:
            size1.append(s-maxb)
    norm_test(np.log(size1))
    '''
    maxb=0.717+1.298*np.log(maxd)
    sigma=np.e**(-0.526+0.033*np.log(maxd))
    mu=maxb
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

    # 计算正态分布的概率密度函数 (PDF)
    pdf = norm.pdf(x, mu, sigma)
    
    # 绘制正态分布曲线
    plt.plot(x, pdf*len(size)*(bins[1]-bins[0]), label=f'μ={mu}, σ={sigma}')
    plt.show()
    '''
    size2=[]
    for s in size:
        if s>=maxb:
            size2.append(s-maxb)
            size2.append(maxb-s)
    d,p=norm_test(size2)
    return maxb,p,np.std(size2, ddof=1)
def output_maxd_norm(maxd_res,title):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        wt.writerow(['maxd','norm test','','','half-norm test',])
        wt.writerow(['','','exp','std','','exp','std'])
        for maxd in maxd_res:
            row=[maxd]
            for num in maxd_res[maxd]:
                row.append("{:.2f}".format(num))
            wt.writerow(row)
    return
def plot_size(cid_mat):
    size=[]
    for cid in cid_mat:
        size.append(len(cid_mat[cid]))
    sizelog=np.log(np.array(size))
    #ax=plt.subplot()
    #ax.set_xlabel('log size')
    #ax.set_ylabel('n')
    #plt.hist(sizelog,bins=20)
    #plt.show()
    return sizelog
def get_maxd_size_distri(cp_d,cid_list):
    maxd_size={}
    for maxd in range(2,10):
        minn=30000
        ratio=1
        cid_mat=cut_mat_by_dis(cp_d,maxd,cid_list,minn,ratio,0)
        size=plot_size(cid_mat)
        maxd_size[maxd]=size
    return maxd_size
def plot_maxd_size_distri(maxd_size):
    plt.figure(figsize=(12,6))
    for i in range(8):
        ax=plt.subplot(2,4,i+1)
        maxd=i+2
        size=maxd_size[maxd]
        ax.set_xlabel('log size')
        ax.set_ylabel('n')
        n, bins, patches=plt.hist(maxd_size[maxd],bins=20)
        d,p0=norm_test(size)
        ax.text(max(bins)-2,max(n)-20,'p='+"{:.4f}".format(p0))
        ax.text(0.1, 1.02,str(chr(96 + i+1)), transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    plt.subplots_adjust()
    plt.tight_layout()
    plt.savefig('size_distri.png',dpi=150)
    plt.show()
    return
def plot_intext3b(size):
    #画正文图3b的子图，把maxd=7的正态hist与估计的正态分布曲线叠加
    ax=plt.subplot()
    ax.set_xlabel('log size')
    ax.set_ylabel('n')
    n, bins, patches = plt.hist(size, bins=15, edgecolor="k")
    mu=np.average(size)
    sigma=np.std(size)
    
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

    # 计算正态分布的概率密度函数 (PDF)
    pdf = norm.pdf(x, mu, sigma)
    mu=f"{mu:.2f}"
    sigma=f"{sigma:.2f}"
    # 绘制正态分布曲线
    plt.plot(x, pdf*len(size)*(bins[1]-bins[0]), label=f'μ={mu}, σ={sigma}')
    plt.legend()
    plt.savefig('main3b.png',dpi=150)
    plt.show()
    
    return 
'''
实证——单纯形数量
'''
def combination(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def matrix_to_graph(distance_matrix, threshold):
    G = nx.Graph()
    n = len(distance_matrix)
    for i in range(n):
        for j in range(i+1, n):  # 只遍历矩阵的上三角部分
            if distance_matrix[i][j] < threshold:
                G.add_edge(i, j)
    return G
def count_tetrahedra(G,dim=3):
    count = 0
    tot=0
    for nodes in combinations(G.nodes, dim+1):  # 遍历图中所有四个节点的组合
        #print(nodes)
        tot+=1
        if all(G.has_edge(nodes[i], nodes[j]) for i in range(dim) for j in range(i + 1, dim+1)):
            count += 1
    #print(count)
    return count,tot  # 每个四面体被计算了6次

def dfs_tetrahedra(G, u, v, visited, depth):
    if depth == 2 and G.has_edge(u, v):
        return 1
    if depth > 2:
        return 0
    
    count = 0
    visited[v] = True
    for w in G.neighbors(v):
        if not visited[w]:
            count += dfs_tetrahedra(G, u, w, visited, depth + 1)
    visited[v] = False
    return count

def count_tetrahedra_dfs(G):
    visited = {node: False for node in G.nodes()}
    total_count = sum(dfs_tetrahedra(G, u, u, visited, 0) for u in G.nodes())
    return total_count // 24  # 每个四面体被计算了6次

def find_4d_simplex(graph, start, depth, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    
    if depth == 0:
        return [visited]
    
    simplex = []
    for neighbor in graph[start]:
        if neighbor not in visited:
            simplex.extend(find_4d_simplex(graph, neighbor, depth-1, visited.copy()))
    
    return simplex

def count_unique_4d_simplex(graph,dim):
    simplex = []
    for node in graph:
        simplex.extend(find_4d_simplex(graph, node, dim))
    
    # Removing duplicates
    unique_simplex = set(tuple(sorted(s)) for s in simplex)
    
    # Filtering sets that aren't fully connected
    fully_connected_simplex = [s for s in unique_simplex if all((i, j) in graph.edges() for i in s for j in s if i != j)]
    
    return len(fully_connected_simplex)
def cut_mat_by_dis_sing(cp_d,maxd,cid_list,minn,ratio,mindim=10):
    cid_mat={}
    cid_sing={}
    count=0
    step=max(1,int(len(cid_list)/20))
    for cid in cid_list:
        if count%step==0:
            print(count/len(cid_list))
        count+=1
        cand=[]
        for cid1 in cid_list:
            cp=(min(cid,cid1),max(cid,cid1))
            if cp_d[cp]<=maxd:
                cand.append(cid1)
        if len(cand)>=minn or random.random()<=ratio:
            mat=[]
            for cid1 in cand:
                row=[]
                for cid2 in cand:
                    cp=(min(cid1,cid2),max(cid1,cid2))
                    row.append(cp_d[cp])
                mat.append(row)
            if len(mat)>=mindim:
                cid_mat[cid]=mat
            else:
                cid_sing[cid]=len(mat)
    return cid_mat,cid_sing
def count_tri(town_mat,maxd,mindim=10,deep=False,maxcal=5,mincal=3):
    #maxcal是最大计算单纯形维数，对应与maxcal+1个点的单纯形
    #计算聚类系数
    count=0
    step=max(1,int(len(town_mat)/20))
    town_n={}
    err=0
    for town in town_mat:
        if count%step==0:
            print(count/len(town_mat),err)
        count+=1
        mat=town_mat[town]
        if len(mat)>=mindim:#需要最小维数保证统计意义，否则分母很小的时候可能会随机性很大
            temp={}
            G=matrix_to_graph(mat, maxd)
            #print(G.number_of_edges())
            maxdim=min(maxcal+1,len(mat))
            for dim in range(mincal,maxcal+1):
                if dim<=maxdim:
                    if deep:
                        n=count_unique_4d_simplex(G,dim)
                        tot=combination(len(mat), dim+1)
                    else:
                        n,tot=count_tetrahedra(G,dim)
                    #print(n,tot)
                    if tot==0:
                        err+=1
                        temp[dim]=0
                    else:
                        temp[dim]=n/tot
                else:
                    temp[dim]=0
            town_n[town]=temp
            #print(temp)
        '''
        else:
            if len(mat)==3:
                if 0 in [mat[0][1], mat[0][2], mat[1][2]]:
                    town_n[town]={3:0,4:0,5:0}
            
            #town_n[town]={3:0,4:0,5:0}
        '''
    return town_n
def count_tri_dim(town_mat,maxd,maxdim,mindim,deep=False,maxcal=5,mincal=3):
    #maxcal是最大计算单纯形维数，对应与maxcal+1个点的单纯形
    #计数单纯形数量而非聚类系数，画si图用的
    count=0
    step=max(1,int(len(town_mat)/20))
    town_n={}
    err=0
    klist=[]
    for town in town_mat:
        if count%step==0:
            print(count/len(town_mat),err)
        count+=1
        mat=town_mat[town]
        if maxdim>len(mat)>=mindim:#需要最小维数保证统计意义，否则分母很小的时候可能会随机性很大
            temp={}
            G=matrix_to_graph(mat, maxd)
            k=len(G.edges())/(len(mat))*2
            klist.append(k)
            #print(G.number_of_edges())
            calmaxdim=min(maxcal+1,len(mat))
            for dim in range(mincal,maxcal+1):
                if dim<calmaxdim:
                    if deep:
                        n=count_unique_4d_simplex(G,dim)
                        tot=combination(len(mat), dim+1)
                    else:
                        n,tot=count_tetrahedra(G,dim)
                    #print(n,tot)
                    if tot==0:
                        err+=1
                        temp[dim]=0
                    else:
                        #temp[dim]=n/tot
                        temp[dim]=n
                else:
                    temp[dim]=0
            town_n[town]=temp
            #print(temp)
    print('average k:',np.average(klist))
    return town_n

def output_cid_n(cid_n,cid_sing,title):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for cid in cid_n:
            wt.writerow([cid,'dim>0',cid_n[cid]])
        for cid in cid_sing:
            wt.writerow([cid,'dim=0',''])
    return
def distri_xy(nlist,k):
    dict1={}
    nsamp=0
    for n in nlist:
        if n!=0:
            key=int(log(n,10)*k)
            if key not in dict1:
                dict1[key]=0
            dict1[key]+=1
            nsamp+=1
    temp=sorted(dict1.items(),key=lambda x:x[0])
    x,y=[],[]
    for it in temp:
        x.append(10**(it[0]/k))
        y.append(it[1]/len(nlist)/(10**((it[0]+1)/k)-10**(it[0]/k)))
    return x,y,nsamp
def plot_intext(file,title='intext_simplex.png',k=5,maxcal=5):
    #正文中图，包含整体聚类系数分布，正文就是maxd=2，子图标题abc
    dim_num={3:[],4:[],5:[]}
    nsing=0
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            if row[1]=='dim>0':
                temp=eval(row[2])
                for dim in temp:
                    dim_num[dim].append(temp[dim])
            else:
                nsing+=1
    plt.figure(figsize=((maxcal-2)*3.5,3))
    plt.subplots_adjust(wspace=0.5)
    subt={0:'a',1:'b',2:'c'}
    for i in range(3):
        dim=i+3
        list1=dim_num[dim]
        print('average num of simplex:',np.average(list1))
        ax=plt.subplot(1,maxcal-2,i+1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Clustering Coefficient')
        if i==0:
            ax.set_ylabel('Probability Density')
        
        x,y,nsamp=distri_xy(list1,k)
        ax.text(x[0],y[-1],'n='+str(nsamp)+', '+str(round(nsamp/2156*100,2))+'%')
        print(dim,nsamp,nsamp/2156)
        plt.plot(x,y,'o',markersize=2,color='royalblue')
        #ax.set_title(str(subt[i]),fontweight='bold')
        ax.text(0.01, 1.02,str(subt[i]), transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(title,dpi=150)
    return

def read_res(file):
    dim_num={3:[],4:[],5:[]}
    dim_nd={3:0,4:0,5:0}
    nsing=0
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            if row[1]=='dim>0':
                temp=eval(row[2])
                for dim in temp:
                    dim_num[dim].append(temp[dim])
                    if temp[dim]>0:
                        dim_nd[dim]+=1
            else:
                nsing+=1
    return dim_num,dim_nd
def plot_si(size_d_n,title,k):
    #按节点大小分组的单纯形数量分布，一个半径下的画在一张图里，因为整体算不动，聚类系数就不放了
    size_d_ave={}
    ytot=len(size_d_n)
    plt.figure(figsize=(3*3.5,ytot*3))
    
    ycount=0
    gcount=0
    for mindim,maxdim in [(7,10),(10,20),(20,30),(30,40)]:
        ycount+=1
        if mindim in size_d_n:
            d_n=size_d_n[mindim]
            size_d_ave[mindim]={}
            for d in d_n:
                gcount+=1
                nlist=d_n[d]
                x,y,nsamp=distri_xy(nlist,k)
                ax=plt.subplot(ytot,3,gcount)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('$nsimp$')
                if d==3:
                    ax.set_ylabel('Probability Density')
                
                ax.text(x[0],y[-1],'n='+str(nsamp)+'; average number='+str(round(np.average(nlist),1)))
                size_d_ave[mindim][d]=np.average(nlist)
                plt.plot(x,y,'o',markersize=2,color='royalblue')
                #ax.set_title(str(subt[i]),fontweight='bold')
                ax.text(0.01, 1.02,str(chr(96 + gcount)), transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(title,dpi=150)
    
    return size_d_ave
def output_maxd_size_ave(maxd_size_ave,title,maxdlist=[3,2.8,2.7,2.5,2.4,2.2,2]):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        wt.writerow(['maxd','size','dim3','dim4','dim5'])
        for maxd in maxdlist:
        #for maxd in [3]:
            for mindim,maxdim in [(7,10),(10,20),(20,30),(30,40)]:
                temp=maxd_size_ave[maxd].get(mindim,'-')
                row=[maxd,mindim]
                for dim in [3,4,5]:
                    
                    if temp=='-':
                        row+=['-']
                    else:
                        row.append(maxd_size_ave[maxd][mindim][dim])
                wt.writerow(row)
    return
'''
#理论证明——环节2排列组合
'''
def cut_mat_by_dis_cand(cp_d,maxd,cid_list,minn,ratio,maxn=10000):
    cid_mat={}
    cid_cand={}
    count=0
    step=max(1,int(len(cid_list)/20))
    for cid in cid_list:
        if count%step==0:
            #print(count/len(cid_list))
            pass
        count+=1
        cand=[]
        for cid1 in cid_list:
            cp=(min(cid,cid1),max(cid,cid1))
            if cp_d[cp]<=maxd:
                cand.append(cid1)
        if maxn>len(cand)>=minn or random.random()<=ratio:
            mat=[]
            for cid1 in cand:
                row=[]
                for cid2 in cand:
                    cp=(min(cid1,cid2),max(cid1,cid2))
                    row.append(cp_d[cp])
                mat.append(row)
            if len(mat)>2:
                cid_mat[cid]=mat
                cid_cand[cid]=cand
    return cid_mat,cid_cand
def analysis_node_fortheo(mat,cand,maxd,centercid):
    #为理论分析提供实证基础，单节点的邻域内分析
    cid_e={}
    cid_d={}
    for i in range(len(mat)):
        cid=cand[i]
        if cid !=centercid:
            cid_e[cid]=set()
            degree=0
            for w in mat[i]:
                if w<=maxd:
                    degree+=1
            if degree>=3:
                cid_d[cid]=degree-2#减掉自连接，以及与中心的连接
                
            for j in range(len(mat)):
                w=mat[i][j]
                if w<=maxd:
                    cid1=cand[j]
                    if cid1!=centercid and cid1!=cid:
                        cid_e[cid].add((cid,cid1))
                    
    return cid_e,cid_d

def com_analysis(mat,cand,cid_e,cid_d,mink):
    #分析节点度大于k的子网络中，组合连接的情况
    n=len(mat)
    level1tot={}
    level2tot={}
    level3tot={}
    level4tot={}
    level2count=0
    level3count=0
    for i in range(n):
        cida=cand[i]
        
        if cid_d.get(cida,0)>=mink:
            level1=[]
            level2=[]
            level3=[]
            level4=[]
            sg1temp=[]
            for edge in cid_e[cida]:#与A相连的点，有多少节点度大于阈值
                for cidb in edge:
                    if cidb!=cida and cid_d.get(cidb,0)>=mink:
                        sg1temp.append(cidb)
            if len(sg1temp)/cid_d[cidb]>0:
                level1.append(len(sg1temp)/cid_d[cida])
            else:
                level1.append(0)
            for cidb in sg1temp:
                
                sg2temp=[]
                for edge in cid_e[cidb]:
                    for cidc in edge:#与B相连的点，有多少是也与A相连的
                        if cidc not in [cida,cidb] and cid_d.get(cidc,0)>=mink and cidc in sg1temp:
                            sg2temp.append(cidc)
                if len(sg2temp)/cid_d[cidb]>0:
                    level2.append(len(sg2temp)/cid_d[cidb])
                else:
                    level2.append(0)
                #level2count+=len(sg1temp)*len(sg2temp)
                level2count+=len(sg2temp)
                for cidc in sg2temp:#与AB都相连的点，有多少与其他AB相连的点相连
                    sg3temp=[]
                    for edge in cid_e[cidc]:
                        for cidd in edge:
                            if cidd not in [cida,cidb,cidc] and cid_d.get(cidd,0)>=mink and cidd in sg2temp:
                                sg3temp.append(cidd)
                    level3count+=len(sg3temp)
                    if len(sg3temp)/cid_d[cidc]>0:
                        level3.append(len(sg3temp)/cid_d[cidc])
                    else:
                        level3.append(0)
                    for cidd in sg3temp:
                        sg4temp=[]
                        for edge in cid_e[cidd]:
                            for cide in edge:
                                if cide not in [cida,cidb,cidc,cidd] and cid_d.get(cide,0)>=mink and cide in sg3temp:
                                    sg4temp.append(cide)
                        if sg4temp!=[]:
                            level4.append(len(sg4temp)/cid_d[cidd])
                        else:
                            level4.append(0)
                        
                if cida not in level3tot:
                    level3tot[cida]=[]
                level3tot[cida]+=level3
                if cida not in level4tot:
                    level4tot[cida]=[]
                level4tot[cida]+=level4
            level2tot[cida]=level2
            level1tot[cida]=level1
            
    #print(np.average(level2tot))
    return level1tot,level2tot,level3tot,level4tot,level2count,level3count
def ratio_analysis(cid_mat,cid_cand,maxd,mink,title):
    #分析局域子图中，各层级内部连边比例的概率分布，每个cid的局域出一个数据点，所有cid合起来画分布
    #每个cid局域的数据点为内部所有连边比例的平均值
    #也可以把所有局域的合起来画分布
    level1rat,level2rat,level3rat,level4rat=[],[],[],[]
    level1all=[]
    level2all=[]
    level3all=[]
    level4all=[]
    size_rat={}#用于20230825的理论分析，见《20230824流形证明结果整理和写作工作记录》的最后部分
    #计算4个ratio与maxd和size的关系，此函数的输入已固定maxd，因此这里只要记录不同size的ratio即可
    #之后用于回归各size的平均ratio，此回归见20230823理论证明整合文件夹，此处仅输出maxd_size_ratio的文件
    #value为list
    
    for cid in cid_mat:
        level1tot=[]
        level2tot=[]
        level3tot=[]
        level4tot=[]
        cid_e,cid_d=analysis_node_fortheo(cid_mat[cid],cid_cand[cid],maxd,cid)
        
        level1,level2,level3,level4,level2count,level3count=com_analysis(cid_mat[cid],cid_cand[cid],cid_e,cid_d,mink)
        for cida in level1:
            level1tot+=level1[cida]
        for cida in level2:
            level2tot+=level2[cida]
        for cida in level3:
            level3tot+=level3[cida]
        for cida in level4:
            level4tot+=level4[cida]
        level1rat.append(np.average(level2tot))
        level2rat.append(np.average(level2tot))
        level3rat.append(np.average(level3tot))
        level4rat.append(np.average(level4tot))
        level1all+=level1tot
        level2all+=level2tot
        level3all+=level3tot
        level4all+=level4tot
        
    plt.figure(figsize=(6,6))
    ax=plt.subplot(2,2,1)
    ax.set_xlabel('$inratio( · ;layer_1)$')
    ax.set_ylabel('n')
    print(np.average(level1all))
    plt.hist(level1all,bins=10)
    ax.text(0.01, 1.02,'a', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    
    ax=plt.subplot(2,2,2)
    ax.set_xlabel('$inratio( · ;layer_2)$')
    #ax.set_ylabel('Number of such subgraphs')
    print(np.average(level2all))
    plt.hist(level2all,bins=10)
    ax.text(0.01, 1.02,'b', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    
    ax=plt.subplot(2,2,3)
    ax.set_xlabel('$inratio( · ;layer_3)$')
    ax.set_ylabel('n')
    print(np.average(level3all))
    plt.hist(level3all,bins=10)
    ax.text(0.01, 1.02,'c', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    
    ax=plt.subplot(2,2,4)
    ax.set_xlabel('$inratio( · ;layer_4)$')
    #ax.set_ylabel('Number of such subgraphs')
    print(np.average(level4all))
    plt.hist(level4all,bins=10)
    ax.text(0.01, 1.02,'d', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    plt.tight_layout()
    plt.savefig(title,dpi=150)
    plt.show()
    return np.average(level1all),np.average(level2all),np.average(level3all),\
np.average(level4all),level1all,level2all,level3all,level4all


def calculate_sumk(cid_mat,cid_cand,maxd,mink):
    #计算每个邻域中，所有大于mink的节点度的和
    cid_sumk={}
    for cid in cid_mat:
        cid_e,cid_d=analysis_node_fortheo(cid_mat[cid],cid_cand[cid],maxd,cid)
        
        #节点度大于阈值（实际上是节点度大于阈值+1，因为这里去掉了与中心的交互）的节点数量
        m=0
        
        klist=[]#节点度大于阈值的节点度组成的list
        for cid1 in cid_d:
            if cid_d[cid1]>=mink:
                m+=1
                temp=0
                for edge in cid_e[cid1]:
                    cida,cidb=edge
                    if cida in cid_d and cidb in cid_d and cid not in edge:
                        #这里分析时都把邻域的中心节点去掉，相应的，去掉中心的单纯形维数，在实际邻域中要+1
                        if cid_d[cida]>=mink and cid_d[cidb]>=mink:
                            temp+=1
                klist.append(temp)
                
        if klist!=[]:
            cid_sumk[cid]=np.sum(klist)
            
    return cid_sumk

def output_sumk(cid_sumk,title):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        for cid in cid_sumk:
            wt.writerow([cid,cid_sumk[cid]])
    return

#理论证明，sumk的分析
def read_sumk(file):
    klist=[]
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            klist.append(int(row[1]))
    return klist
def sumk_thre_analysis(thre1,thre2,window=30):
    maxd_size_pro={}
    maxd_data={}
    for maxd in [2,2.2,2.4,2.5,2.7,2.8,3]:
        
        n1,n2,sizelist=[],[],[]#将不同size的k合到一起，但按size排序，以便按邻近统计，做个滑动窗口
        for minn in range(7,80):
            
            maxn=minn+1
            mink=4
            file=r'sumk_maxd'+str(maxd)+'_mink'+str(mink)+'_minn'+str(minn)+'_maxn'+str(maxn)+'.csv'
            klist=read_sumk(file)
            for k in klist:
                if k>thre1:
                    n1.append(1)
                else:
                    n1.append(0)
                if k>thre2:
                    n2.append(1)
                else:
                    n2.append(0)
                sizelist.append(minn)
        maxd_data[maxd]=(n1,n2,sizelist)
        x,y1,y2=[],[],[]
        for i in range(len(n1)-window+1):
            n1temp=n1[i:i+window]
            n2temp=n2[i:i+window]
            stemp=sizelist[i:i+window]
            x.append(np.average(stemp))
            y1.append(np.average(n1temp))
            y2.append(np.average(n2temp))
        maxd_size_pro[maxd]=(x,y1,y2)
    return maxd_size_pro,maxd_data
def size_pro_reg_plot(maxd_size_pro,title='size_pro.png',maxdlist=[2,2.2,2.4,2.6,2.8,3],logx=False):
    wnum=int(len(maxdlist)/2)+1
    plt.figure(figsize=(4*wnum,8))
    
    for i in range(len(maxdlist)):
        maxd=maxdlist[i]
        
        ax=plt.subplot(2,wnum,i+1)
        if logx:
            ax.set_xscale('log')
        if i//wnum==1:
            ax.set_xlabel('size')
        if i%wnum==0:
            ax.set_ylabel('Proportion of D with simplices')
        x,y1,y2=maxd_size_pro[maxd]
        x=np.array(x)
        y1=np.array(y1)
        y2=np.array(y2)
        
        plt.plot(x,y2,'o',markersize=2,color='orange',label='Dim 3')
        plt.plot(x,y1,'o',markersize=2,color='royalblue',label='Dim 4')
        ax.text(0.01, 1.02,str(chr(96 + i+1)), transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
    plt.legend()
    plt.savefig(title,dpi=150)
    plt.show()
    return
def read_cid_sumk(file):
    cid_k={}
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            cid_k[int(row[0])]=int(row[1])
    return cid_k
def cut_pop(maxpop,cover):
    cidlist=[]
    for cid in cover:
        if len(cover[cid])<=maxpop:
            cidlist.append(cid)
    return cidlist
def cid_sumk_thre_analysis_cut(nodes_in_cut,thre1,thre2,cid_list,window=30,shell=True):
    maxd_size_pro={}
    maxd_data={}
    for maxd in [2,2.2,2.4,2.5,2.7,2.8,3]:
        
        n1,n2,sizelist=[],[],[]#将不同size的k合到一起，但按size排序，以便按邻近统计，做个滑动窗口
        #n1是四维的，n2是三维的
        for minn in range(7,80):
            
            maxn=minn+1
            mink=4
            file=r'sumk_maxd'+str(maxd)+'_mink'+str(mink)+'_minn'+str(minn)+'_maxn'+str(maxn)+'.csv'
            cid_k= read_cid_sumk(file)
            if shell:
                cidlist=nodes_in_cut
            else:
                cidlist=list(set(cid_list)-set(nodes_in_cut))
            
            #print(len(cidlist))
            for cid in cid_k:
                k=cid_k[cid]
                if cid in cidlist:
                    if k>thre1:
                        n1.append(1)
                    else:
                        n1.append(0)
                    if k>thre2:
                        n2.append(1)
                    else:
                        n2.append(0)
                    sizelist.append(minn)
        maxd_data[maxd]=(n1,n2,sizelist)
        x,y1,y2=[],[],[]
        for i in range(len(n1)-window+1):
            n1temp=n1[i:i+window]
            n2temp=n2[i:i+window]
            stemp=sizelist[i:i+window]
            x.append(np.average(stemp))
            y1.append(np.average(n1temp))
            y2.append(np.average(n2temp))
        maxd_size_pro[maxd]=(x,y1,y2)
    return maxd_size_pro,maxd_data
def regression_with_asym(x,y,maxd=2):
    # Input data
    #x = np.array([11.6, 11.9, 12.2, 12.5, 12.833333333333334, 13.133333333333333, 13.433333333333334, 13.766666666666667, 14.1, 14.433333333333334, 14.766666666666667, 15.1, 15.4, 15.7, 16.033333333333335, 16.366666666666667, 16.7, 17.033333333333335, 17.366666666666667, 17.733333333333334, 18.066666666666666, 18.4, 18.733333333333334, 19.1, 19.433333333333334, 19.866666666666667, 20.266666666666666, 20.766666666666666, 21.233333333333334, 21.7, 22.166666666666668, 22.666666666666668, 23.166666666666668, 23.666666666666668, 24.166666666666668, 24.7, 25.266666666666666, 25.8, 26.333333333333332, 26.933333333333334, 27.533333333333335, 28.2, 28.9, 29.633333333333333, 30.466666666666665, 31.4, 32.333333333333336, 33.3, 34.46666666666667, 35.8, 37.2, 38.733333333333334, 40.333333333333336, 42.0, 43.766666666666666])
    #y = np.array([0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.7, 0.7, 0.7333333333333333, 0.7333333333333333, 0.7666666666666667, 0.8, 0.8, 0.8, 0.8333333333333334, 0.8333333333333334, 0.8666666666666667, 0.8666666666666667, 0.8666666666666667, 0.8333333333333334, 0.8333333333333334, 0.8666666666666667, 0.8666666666666667, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8666666666666667, 0.9, 0.9, 0.9, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667, 0.9666666666666667])
    
    # 1. Estimate the asymptotic value from the tail of the data
    tail_fraction = 0.1  # Use the last 20% of data
    y_asym = np.mean(y[-int(len(y) * tail_fraction):])
    print(y_asym)
    # 2. Find the x_0 value
    threshold = 0.02  # A threshold to determine how close y should be to y_asym to be considered asymptotic
    indices_asym = np.where(np.abs(y - y_asym) < threshold)[0]
    x_0 = x[indices_asym[0]]
    #x_0=(3-maxd)/0.2*3+15
    print(x_0,indices_asym[0])
    
    
    # 3. Linear regression for x < x_0
    #x_linear = x[x < x_0].reshape(-1, 1)
    if indices_asym[0]>0:
        x_linear = x[:indices_asym[0]]
        x1=[]
        for temp in x_linear:
            x1.append([temp])
        y_linear = y[:indices_asym[0]]
        lin_reg = LinearRegression().fit(x1, y_linear)
        
        m = lin_reg.coef_[0]
        c = lin_reg.intercept_
        r2_score=lin_reg.score(x1, y_linear)
        print(r2_score)
    else:
        m, c, x_0, y_asym,r2_score='','',x_0, y_asym,''
    return m, c, x_0, y_asym,r2_score
def regression_size_pro(maxd_size_pro,title):
    maxd_regres={}
    for maxd in maxd_size_pro:
        print('-----maxd',maxd)
        maxd_regres[maxd]={}
        x,y1,y2=maxd_size_pro[maxd]
        m1,c1,x01,yas1,r21=regression_with_asym(x,y1)#四维
        maxd_regres[maxd]['dim4']=(m1,c1,x01,yas1,r21)
        m1,c1,x01,yas1,r21=regression_with_asym(x,y2)#三维
        maxd_regres[maxd]['dim3']=(m1,c1,x01,yas1,r21)
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        wt.writerow(['maxd','dim','slope','intercept','asymptotic position','asymptotic value','R squared for regression'])
        for maxd in maxd_regres:
            
            for dim in maxd_regres[maxd]:
                row=[maxd]
                row.append(dim)
                row+=list(maxd_regres[maxd][dim])
                wt.writerow(row)
    return
'''
理论证明——结果计算
'''
def probability_between_values(mu, sigma, x0, x1):
    return norm.cdf(x1, mu, sigma) - norm.cdf(x0, mu, sigma)
def read_sizepro(file=r'F:\流形研究\20230823理论证明整合\size_pro_pop5False_window30.csv'):
    maxd_dim_para={}
    with open(file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        for row in rd:
            maxd=float(row[0])
            if maxd not in maxd_dim_para:
                maxd_dim_para[maxd]={}
            dim=row[1]
            maxd_dim_para[maxd][dim]=(float(row[2]),float(row[3]),float(row[4]),float(row[5]))
    return maxd_dim_para
def estimate_simp_1sizepro(maxd,size0,size1,maxd_dim_para,N=2156):
    #计算maxd下，size在size0和size1之间（左闭右开）的邻域中有多少包含高维单纯形
    #---计算该size的邻域数量
    mu=0.717+1.298*log(maxd)
    logsigma=-0.526+0.033*maxd
    sigma=e**logsigma
    ndim3,ndim4=0,0
    for size in range(size0,size1):
        psize=probability_between_values(mu, sigma, log(size), log(size+1))
        n0=psize*N
        #print(n0)
        #---给定size和maxd，计算形成单纯形的概率
        alpha,beta,xasym,yasym=maxd_dim_para[maxd]['dim3']
        if size<xasym:
            ndim3+=n0*(beta+alpha*size)
        else:
            ndim3+=n0*yasym
        alpha,beta,xasym,yasym=maxd_dim_para[maxd]['dim4']
        if size<xasym:
            #print(size,beta+alpha*size)
            ndim4+=n0*(beta+alpha*size)
        else:
            ndim4+=n0*yasym
    return ndim3,ndim4
def output_est(title,maxd_dim_para,maxd_dim_para1='',opt='1sizepro',\
               maxdlist=[2,2.2,2.4,2.5,2.7,2.8,3],sizelist=[7,10,20,30,40]):
    maxd_size_n={}
    for maxd in maxdlist:
        maxd_size_n[maxd]={}
        for i in range(len(sizelist)-1):
            size0=sizelist[i]
            size1=sizelist[i+1]
            if opt=='1sizepro':
                ndim3,ndim4=estimate_simp_1sizepro(maxd,size0,size1,maxd_dim_para)
            else:
                pass
            maxd_size_n[maxd][size0]=(ndim3,ndim4)
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        wt.writerow(['maxd','size','dim3','dim4'])
        for maxd in maxdlist:
            for size in sizelist[:-1]:
                ndim3,ndim4=maxd_size_n[maxd][size]
                wt.writerow([maxd,size,ndim3,ndim4])
    return maxd_size_n
def est_compare_emp():
    # Load data
    est_1sizepro_df = pd.read_csv("est_1sizepro.csv")
    maxd_size_nd_df = pd.read_csv("maxd_size_nd.csv")
    
    # Merge dataframes based on the first two columns
    merged_df = pd.merge(est_1sizepro_df, maxd_size_nd_df, on=maxd_size_nd_df.columns[0:2].tolist())
    
    # Filter out rows with '-' in either dataframe
    filtered_df = merged_df[(merged_df[merged_df.columns[4]] != '-') & (merged_df[merged_df.columns[5]] != '-')]
    
    # Convert to float
    filtered_df[filtered_df.columns[2]] = filtered_df[filtered_df.columns[2]].astype(float)
    filtered_df[filtered_df.columns[3]] = filtered_df[filtered_df.columns[3]].astype(float)
    
    # Perform linear regression for column 3
    X_col3 = filtered_df[filtered_df.columns[4]].values.reshape(-1, 1)
    y_col3 = filtered_df[filtered_df.columns[2]].values
    lm_col3 = LinearRegression().fit(X_col3, y_col3)
    r2_col3 = r2_score(y_col3, lm_col3.predict(X_col3))
    
    # Perform linear regression for column 4
    X_col4 = filtered_df[filtered_df.columns[5]].values.reshape(-1, 1)
    y_col4 = filtered_df[filtered_df.columns[3]].values
    lm_col4 = LinearRegression().fit(X_col4, y_col4)
    r2_col4 = r2_score(y_col4, lm_col4.predict(X_col4))
    
    # Plot scatter plot for column 3
    plt.figure(figsize=(9, 4))
    
    plt.subplot(1, 2, 1)
    #print(list(y_col3))
    x3=[]
    for x in list(X_col3):
        x3.append(float(x[0]))
    plt.scatter(x3, list(y_col3), color='royalblue')
    #plt.plot(X_col3, y_col3,'o', color='blue')
    plt.plot(x3, lm_col3.predict(X_col3), color='orange')
    #plt.title(f"Regression for dim3 ($R^2$ = {r2_col3:.2f})")
    plt.plot([],[],'o',color='white',label=f"Regression for dim3 ($R^2$ = {r2_col3:.2f})")
    plt.xlabel('Num of D with dim3 simplex')
    plt.ylabel('Estimated num of D with dim3 simplex')
    plt.legend()
    
    # Plot scatter plot for column 4
    plt.subplot(1, 2, 2)
    x4=[]
    for x in list(X_col4):
        x4.append(float(x[0]))
    plt.scatter(x4, list(y_col4), color='royalblue')
    plt.plot(x4, lm_col4.predict(X_col4), color='orange')
    plt.plot([],[],'o',color='white',label=f"Regression for dim4 ($R^2$ = {r2_col4:.2f})")
    #plt.title(f"Regression for dim4 ($R^2$ = {r2_col4:.2f})")
    plt.xlabel('Num of D with dim4 simplex')
    plt.ylabel('Estimated num of D with dim4 simplex')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('compare_est.png',dpi=150)
    plt.show()
    
    r2_col3, r2_col4
    return
'''
维数计算方法比较
'''
'''
比较单纯形计数和mds
读取单纯形分析结果文件，
每个maxd，
定义出现至少k_tri/比例r_tri的邻域中心，为单纯形高维（先做至少1个
mds结果，邻域的特征根达到最大特征根的r_root，判别维数
分为四块，单纯形高维mds高维，类推
'''

def read_tri_res(file):
    cid_dim={}#读取以cid为中心的邻域内的各维度单纯形数量，
    #理论上来说，同一maxd，不同size领域的中心之间是没有交集的
    nsing=0
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            if row[1]=='dim>0':
                cid=int(row[0])
                cid_dim[cid]={}
                temp=eval(row[2])
                for dim in temp:
                    cid_dim[cid][dim]=temp[dim]
                    
            else:
                nsing+=1
    return cid_dim
def test_tri_size_inter(maxd):
    size_cid={}
    for minn,maxn in [(7,10),(10,20),(20,30),(30,40)]:
        file=r'F:\流形研究\20230815流形存在性-三角数量2\numtridim_maxd'+str(maxd)+'_mindim'+str(minn)+'_maxdim'+str(maxn)+'_maxcal5.csv'
        try:
            with open(file,'r') as f:
                rd=csv.reader(f)
        except:
            pass
        else:
            cid_dim=read_tri_res(file)
            size_cid[minn]=set(cid_dim.keys())
    for size in size_cid:
        for size1 in size_cid:
            if size!=size1:
                print(size,size1,size_cid[size]&size_cid[size1])
    return
def tri_dim(dim_n,ktri,rtri,aven):
    if ktri!='':
        if dim_n[3]<ktri:
            return 2
        else:
            if dim_n[4]<ktri:
                return 3
            else:
                if dim_n[5]<ktri:
                    return 4
                else:
                    return 5
    else:
        if dim_n[3]<rtri*combination(aven, 4):
            return 2
        else:
            if dim_n[4]<rtri*combination(aven, 5):
                return 3
            else:
                if dim_n[5]<rtri*combination(aven, 6):
                    return 4
                else:
                    return 5
        
def mds_dim(alist,rmds):
    dim=0
    while alist[dim]>0 and alist[dim]/alist[0]>rmds:
        dim+=1
    return dim
def compare_single(dim_n,alist,ktri,rtri,aven,rmds,low):
    dimtri=tri_dim(dim_n,ktri,rtri,aven)
    dimmds=mds_dim(alist,rmds)
    #print(dimmds)
    if dimtri<=low and dimmds<=low:
        return 'tril_mdsl'
    if dimtri>low and dimmds<=low:
        return 'trih_mdsl'
    if dimtri<=low and dimmds>low:
        return 'tril_mdsh'
    if dimtri>low and dimmds>low:
        return 'trih_mdsh'
        
def compare_mds_tri(maxd,ktri,rtri,rmds,low=3):
    #mds的doublemaxd的结果维数太高了，只能用普通mds的分析结果了
    #读取mds和tri的结果，将tri结果合并
    #根据tri和参数ktri计算每个cid的tri维数
    #根据mds和参数rmds计算每个cid的mds维数
    opt='mdsnew_'
    minn=30000
    ratio=1
    file=opt+'maxd'+str(maxd)+'_minn'+str(minn)+'_ratio'+str(ratio)+'mds.csv'
    cid_a=read_town_a(file)
    res={'tril_mdsl':0,'trih_mdsl':0,'tril_mdsh':0,'trih_mdsh':0}
    tot=0
    for minn,maxn in [(1,7),(7,10),(10,20),(20,30),(30,40)]:
        aven=max(int(minn/2+maxn/2),6)
        
        file=r'F:\流形研究\20230815流形存在性-三角数量2\numtridim_maxd'+str(maxd)+'_mindim'+str(minn)+'_maxdim'+str(maxn)+'_maxcal5.csv'
        try:
            with open(file,'r') as f:
                rd=csv.reader(f)
        except:
            pass
        else:
            cid_dim=read_tri_res(file)
            for cid in cid_dim:
                dim_n=cid_dim[cid]
                if cid in cid_a:
                    alist=cid_a[cid]
                    
                    tot+=1
                    temp=compare_single(dim_n,alist,ktri,rtri,aven,rmds,low)
                    res[temp]+=1
    res1={}
    for key in res:
        
        res1[key]=res[key]/tot
    return res,res1
def output_compare(title,ktri,rtri,rmds):
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        wt.writerow(['maxd','judged low dim by both methods','judged low dim by MDS','judged low dim by simplices','judged high dim by both methods'])
        for maxd in [3,2.8,2.7,2.5,2.4,2.2,2]:
            res,res1=compare_mds_tri(maxd,ktri,rtri,rmds,low=3)
            print(maxd,res1)
            if ktri=='':
                wt.writerow([maxd]+list(res1.values()))
            else:
                wt.writerow([maxd]+list(res.values()))
    return




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
    
    scale=6
    
    #-----选址流形
    
    file=r'F:\流形研究\20230812taz+流形修改\visit_coverrate_6_cover.csv'
    file=r'visit_coverrate_6_cover.csv'
    file=r'E:\流形研究\20230831代码交接\20230831选址流形交付\cover_cleantraj_scale150_visit.csv'
    cover=read_cover(file)
    '''
    minf=5
    opt='minf5_coverrate_inf100_'
    
    plot_pop_sne(cover,scale,opt,True,100,minf)#设置交互量下限后，重新输出矩阵，以便位序规模分析
    #***更改城市需要传入ynum参数
    
    opt='sub_coverrate_'
    res,lscover,mat,nodes=plot_pop_sne(cover,scale,opt,True)
    
    opt='visit_coverrate_'
    res,lscover,mat,nodes=plot_pop_sne(cover,scale,opt,True)
    '''
    #-----选址流形变形
    
    locfile=r'F:\流形研究\20230831代码交接\20230831选址流形交付\results\ls_clean_con_home_subc1_subu1_cho300_scale900_visit.csv'
    locc=read_locc(locfile)
    manifoldfile='visit_coverrate_6_manifold.csv'
    manifoldfile=r'E:\流形研究\20230831代码交接\20230831选址流形交付\results1\ls_manifold_coverrate_subc1_subu1_scale900.csv'
    #manifoldfile=r'F:\流形研究\20230812taz+流形修改\minf5_coverrate_inf100_6_manifold.csv'
    res,nodes=read_manifold(manifoldfile)
    
    
    #df,transres,nodes1=manifold_transby_pop(res,nodes,cover)
    #plot_with_locc(locc,nodes1,transres,cover,r'F:\流形研究\20230925选址流形人口密度变形测试结果\trans_900_locc_1.png',color='cover')
    #df,log_transres,nodes1=manifold_transby_pop(res,nodes,cover,'log')
    #plot_with_locc(locc,nodes1,log_transres,cover,r'F:\流形研究\20230925选址流形人口密度变形测试结果\trans_900_locc_log.png',color='cover')
    #df,sqrt_transres,nodes1=manifold_transby_pop(res,nodes,cover,'sqrt')
    #plot_with_locc(locc,nodes1,sqrt_transres,cover,r'F:\流形研究\20230925选址流形人口密度变形测试结果\trans_900_locc_sqrt.png',color='cover')
    
    #test_res,test_nodes,test_cover=create_test_trans()
    #test_df,test_transres,test_nodes1=manifold_transby_pop(test_res,test_nodes,test_cover)
    #plot_with_locc([],test_nodes,test_transres,test_cover,r'F:\流形研究\20230925选址流形人口密度变形测试结果\test.png',color='cover')
    
    xmin, xmax, ymin, ymax =113.67561783007596,114.60880792079337,\
    22.28129833936937,22.852485545898546#深圳最大最小经纬度
    ynumls=int(424/6)+1
    #map_res=map_coords(nodes1,ynumls)
    #map_df,map_transres,map_nodes1=manifold_transby_pop(map_res,nodes1,cover)
    #plot_with_locc(locc,nodes1,map_transres,cover,r'F:\流形研究\20230925选址流形人口密度变形测试结果\trans_map_900_locc_1.png',color='cover')
    #df,sqrt_map_transres,nodes1=manifold_transby_pop(map_res,nodes1,cover,'sqrt')
    #plot_with_locc(locc,nodes1,sqrt_map_transres,cover,r'F:\流形研究\20230925选址流形人口密度变形测试结果\trans_map_900_locc_sqrt.png',color='cover')
    #plot_with_locc(locc,nodes1,map_res,cover,r'F:\流形研究\20230925选址流形人口密度变形测试结果\map_900_locc.png',color='cover')
    
    
    #-----均匀性检验
    '''
    title=r'F:\流形研究\20230925选址流形人口密度变形测试结果\trans_900_locc_R.png'
    calculate_r_(transres,nodes1,locc,title)
    title=r'F:\流形研究\20230925选址流形人口密度变形测试结果\trans_map_900_locc_R.png'
    calculate_r_(map_transres,nodes1,locc,title)
    '''
    #calculate_r_(sqrt_transres,nodes1,locc,title)
    #calculate_r_(sqrt_map_transres,nodes1,locc,title)
    '''
    #-----动图可视化
    title=r'F:\流形研究\20230926动图\test.png'
    gap=1
    bre=5
    cutdist=0.1
    #manifold_gif(cover,nodes,gap,bre,res,title,cutdist)
    merge_gif()
    '''
    '''
    #-----mds
    file=r'visit_coverrate_6_mat.csv'
    cp_d=read_cp(file)
    cid_list=read_cid_list(file)
    k=5
    plot_distri(cp_d,k,'distribution of coverrate.png')
    
    opt='visit_6_coverrate_'
    opt='visit_6_coverrate_path_'
    opt='testinf_'
    for maxd in [1.8]:
    #for maxd in [3,2.8,2.7,2.5,2.4,2.2,2]:#mds不需要按minn，maxn分开，只要读取后对到cid邻域上即可
    #for maxd in [0.001,0.002,0.003,0.006,0.01]:
        print(maxd)
        minn=30000
        ratio=1
        times=0.1
        cid_mat=cut_mat_by_dis(cp_d,maxd,cid_list,minn,ratio)
        print('cid_mat ready')
        if opt.find('path')!=-1:
            print('path')
            cid_a=town_mds_frommat_path(cid_mat,maxd,5)
        elif opt.find('testinf')!=-1:
            cid_mat=test_inf(cid_mat,10)
            cid_a=town_mds_frommat(cid_mat,True)
        else:
            cid_a=town_mds_frommat(cid_mat,True)
            
        output_town_a(cid_a,opt+'maxd'+str(maxd)+'_minn'+str(minn)+'_ratio'+str(ratio)+'mds.csv')
        cid_a=read_town_a(opt+'maxd'+str(maxd)+'_minn'+str(minn)+'_ratio'+str(ratio)+'mds.csv')
        plot_town_a(cid_a,opt+'maxd'+str(maxd)+'_minn'+str(minn)+'_ratio'+str(ratio)+'_times'+str(times)+'mds.png',times)
    '''
    '''
    #-----多尺度mds
    file=r'visit_coverrate_6_mat.csv'
    cp_d=read_cp(file)
    cid_list=read_cid_list(file)
    minratio,maxd0=0.8,3.8
    res=multiscale_while(minratio,cp_d,maxd0,10)
    plot_multiscale(res)
    '''
    '''
    #-----位序规模
    '''
    '''
    #---计算位序规模，对coverrate而非距离，双对数轴
    #cover=read_cover(r'visit_coverrate_6_cover.csv')
    minf=5
    ynumls=int(ynum/scale)+1
    #lscover=get_large_scale_cover(cover,scale,ynumls)
    ratio=1
    covisit_rank(lscover,ratio)#分析交互量（即重复覆盖率，而非其倒数）的位序规模
    '''
    '''
    #---计算位序规模，对距离，即流形学习的输入矩阵，双对数轴
    file='minf5_coverrate_inf100_6_mat.csv'
    ratio=0.3
    inf=100
    cid_rank=read_mat_sample(file,ratio,inf)
    title='dist_rank.png'
    plot_rank(cid_rank,title,minl=30)
    '''
    '''
    #---位序规模回归
    file=r'F:\流形研究\20230812taz+流形修改\visit_coverrate_6_cover.csv'
    
    cover=read_cover(file)
    
    ratio=1
    cid_xy,xlist,ylist=covisit_rank_xyoutput(cover,ratio)
    cid_ab,cid_r2=reg_each_cid(cid_xy)
    
    plot_abr2(cid_ab,cid_r2)
    maxd=9
    
    #---中心极限定理解释的分析，coverrate和rank的变化独立同分布
    
    test_iid_changerank(cid_xy,10,False,0.1,'iid_changerate.png')
    test_iid_changerank(cid_xy,10,True,1,'iid_changedist.png')
    '''
    '''
    #---size的正态性检验
    file=r'F:\流形研究\20230812taz+流形修改\visit_coverrate_6_mat.csv'
    cp_d=read_cp(file)
    cid_list=read_cid_list(file)
    
    maxd_res1={}
    for maxd in range(20,90):
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
    '''
    '''
    #---画main 3b
    maxd_res1={}
    for maxd in [7]:
        minn=30000
        ratio=1
        cid_mat=cut_mat_by_dis(cp_d,maxd,cid_list,minn,ratio,0)
        size=plot_size(cid_mat)
        plot_intext3b(size)
    
    #---对size均值和标准差对maxd进行回归
    file='maxd_norm.csv'
    reg_maxd_sizemustd(file)
    
    #-----画size分布
    maxd_size=get_maxd_size_distri(cp_d,cid_list)
    plot_maxd_size_distri(maxd_size)
    '''
    
    
    '''
    
    #-----单纯形
    file=r'F:\流形研究\20230812taz+流形修改\visit_coverrate_6_mat.csv'
    #cid_list=read_cid_list(file)
    #cp_d=read_cp(file)
    mindim=7#因为会计算到5维单纯形，为了保证统计意义，至少矩阵是8*8的，这样1个5维单纯形时就是C8-2
    #后面改变了计算最大维数就可以降低
    minn=30000
    maxcal=5
    mincal=3
    
    #给定maxd，数单纯形
    for maxd in [2]:
        
        cid_mat,cid_sing=cut_mat_by_dis_sing(cp_d,maxd,cid_list,minn,1,mindim)
        print(len(cid_mat))#该维数下满足的邻域数量
        cid_n=count_tri(cid_mat,maxd,mindim,True,maxcal,mincal)
        print(len(cid_n))#确认一下前面的邻域筛选对不对
        
        
        file='tri_maxd'+str(maxd)+'_mindim'+str(mindim)+'_maxcal'+str(maxcal)+'.csv'
        output_cid_n(cid_n,[],file)#cid_sing是之前0维需要另外记录的情况，
        #但现在我们设置了mindim以保证统计意义，所以不需要另外记录了，
        k=3
    file='tri_maxd2_mindim7_maxcal5.csv'
    plot_intext(file,title='intext_simplex.png',k=5)
    '''
    '''
    #读取分析结果，画图
    maxcal=5
    mincal=3
    maxd_size_ave={}
    maxd_size_nd={}
    for maxd in [3,2.8,2.7,2.5,2.4,2.2,2]:
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
                size_d_n[mindim]=dim_num
                size_d_nd[mindim]=dim_nd
        maxd_size_nd[maxd]=size_d_nd
    
        
        k=3
        title='si_numtri_k'+str(k)+'_maxd'+str(maxd)+'_mindim'+str(mindim)+'_maxdim'+str(maxdim)+'.png'
        size_ave=plot_si(size_d_n,title,k)#不把single加进去
        maxd_size_ave[maxd]=size_ave
    
    #print(maxd_size_ave)
    title='maxd_size_ave.csv'
    output_maxd_size_ave(maxd_size_ave,title)
    title='maxd_size_nd.csv'
    output_maxd_size_ave(maxd_size_nd,title)
    '''
    #-----理论证明，排列组合内部连边比例
    '''
    #---计算inratio及其平均
    maxd_minn_ratio={}
    layer_ratio={1:[],2:[],3:[],4:[]}
    #for maxd in [3]:
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
            
            
            #---对每个邻域取平均，后以邻域为单位分析
            layer_ratio[1].append(l1)
            layer_ratio[2].append(l2)
            layer_ratio[3].append(l3)
            layer_ratio[4].append(l4)
            if maxd not in maxd_minn_ratio:
                maxd_minn_ratio[maxd]={}
            maxd_minn_ratio[maxd][minn]=(l1,l2,l3,l4)
    '''
            
    
    '''
    #---计算不同size的sumk，用以后续理论证明
    for maxd in [2,2.2,2.4,2.5,2.7,2.8,3]:
        for minn in range(7,80):
        #for minn in range(7,40):
            maxn=minn+1
            print(maxd,minn)
        
            ratio=0
        
            cid_mat,cid_cand=cut_mat_by_dis_cand(cp_d,maxd,cid_list,minn,ratio,maxn)
            
            mink=4
            #-----计算给定size，给定maxd后，每个邻域中大于mink节点的度总和，用以输出分布
            cid_sumk=calculate_sumk(cid_mat,cid_cand,maxd,mink)
            title='sumk_maxd'+str(maxd)+'_mink'+str(mink)+'_minn'+str(minn)+'_maxn'+str(maxn)+'.csv'
            output_sumk(cid_sumk,title)
    '''
    '''
    #---sumk分析
    thre1,thre2=17.451440494252715,5.250561153723304
    maxdlist=[2,2.2,2.4,2.5,2.7,2.8,3]
    maxd_size_pro,maxd_data=sumk_thre_analysis(thre1,thre2,30)
    
    '''
    '''
    #---sumk回归
    file=r'F:\流形研究\20230812taz+流形修改\visit_coverrate_6_cover.csv'
    cover=read_cover(file)
    maxpop=5
    poplist=cut_pop(maxpop,cover)
    thre1,thre2=17.451440494252715,5.250561153723304
    shell=False
    #shell=True
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
    '''
    '''
    #-----理论证明——结果计算
    maxd_dim_para=read_sizepro('size_pro_pop5False_window30.csv')
    title=r'est_1sizepro.csv'
    maxd_size_n=output_est(title,maxd_dim_para)
    '''
    '''
    #---理论证明估算单纯形邻域数量结果对实证的回归
    est_compare_emp()
    '''
    #-----维数计算方式比较
    '''
    maxd=2
    ktri=''
    
    rtri=0.01
    rmds=0.2
    res,res1=compare_mds_tri(maxd,ktri,rtri,rmds,low=3)
    title='compare_ktri'+str(ktri)+'_rtri'+str(rtri)+'_rmds'+str(rmds)+'.csv'
    output_compare(title,ktri,rtri,rmds)
    '''