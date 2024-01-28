"""
最近邻分析，画格网统计面积，计算平均最短距离
"""

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import csv
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
                
    return np.array(res),nodes

def read_locc(file):
    #read the result of location choice
    locc=[]
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            locc.append(int(row[0]))
    return locc

def get_xy(cid,ynum=424,scale=1):
    ynum = int(np.ceil(ynum/scale))
    xid=int((cid-1)/ynum)
    yid=cid-xid*ynum-1
    return xid,yid

def calc_da(X):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    # plt.hist(distances.T[1],bins=30)
    # plt.show()
    d_a = np.average(distances.T[1])
    return d_a

def xy2gid(X:np.ndarray,d,Xlocc = None,ax=None):
    """在最小外包矩形左小角开始绘制渔网，覆盖所有点。给定点的坐标，计算所在渔网的坐标

    Args:
        X (ndarray): 点坐标
        d (double): 格网边长
        Xlocc (ndarray): 选址点
    Returns:
        gid (ndarray): 每个格点所在的格网id
    """
    x, y = X.T[0],X.T[1]
    xmin,xmax = x.min(),x.max()
    ymin,ymax = y.min(),y.max()
    
    xnum = int((xmax - xmin) / d) + 1
    ynum = int((ymax - ymin) / d) + 1
    # 0,0表示左下角格子
    xx = ((x - xmin) / d).astype(int)
    yy = ((y - ymin) / d).astype(int)
    
    if ax is not None:
        for i,xxx in enumerate(xx):
            # Create a Rectangle patch
            rect = patches.Rectangle((xmin + d*xx[i], ymin + d*yy[i]), d, d, linewidth=1, edgecolor='lightblue', facecolor='lightblue',alpha=0.2)
            # Add the patch to the Axes
            ax.add_patch(rect)
        ax.scatter(x,y)
        if Xlocc is not None:
            ax.scatter(Xlocc.T[0],Xlocc.T[1],c="r",s=70)
        # plt.vlines([d * i + xmin for i in range(xnum+1)],ymin,ymin + d * ynum)
        # plt.hlines([d * i + ymin for i in range(ynum+1)],xmin,xmin + d * xnum)
        ax = plt.gca()
        ax.set_aspect('equal')
        # plt.show()
    
    return yy * xnum + xx

def calc_area(X,d=None,Xlocc=None,ax=None):
    """画边长为d的渔网，计算面积

    Args:
        X (ndarray): 流形点
        d (float, optional): 渔网边长，默认为平均最短距离. Defaults to None.
        Xlocc (ndarray, optional): 选址点. Defaults to None.

    Returns:
        A (float): 面积
    """
    if d is None:
        d = calc_da(X)
    gid = xy2gid(X,d,Xlocc,ax=ax)
    print("grid count:",len(np.unique(gid)))
    print("grid scale = ",d)
    A = d * d * len(np.unique(gid))
    return A

def calc_R(X,X_locc,d=None,title=""):
    fig,ax = plt.subplots(figsize=(36,36))
    da = calc_da(X_locc)
    de = 1 / (2 * np.sqrt(len(X_locc) / calc_area(X,d=d,Xlocc=X_locc,ax=ax)))
    print("loc count: {}".format(len(X_locc)))
    print("da = {}, de = {}".format(da,de))
    R = da / de
    ax.set_title("R = {}".format(R))
    fig.savefig(title)
    return R

if __name__ == "__main__":
    # 参数
    subc = 400
    subu = 2
    method = ["covisit","coverrate"][0]
    print("subc:{}  subu:{}\nmethod:{}\n".format(subc,subu,method))
    
    # 读取流形坐标、选址结果
    res,nodes=read_manifold('ls_manifold{}_subc{}_subu{}_scale900.csv'.format("_"+method if method == "coverrate" else "",subc,subu))
    locc=read_locc('ls_clean_greedy_nodup_home_subc{}_subu{}_cho300_scale900_visit.csv'.format(subc,subu))
    
    # 准备数据
    embedding = pd.DataFrame({"cid":nodes,"x":res.T[0],"y":res.T[1]})
    embedding["x_orig"] = embedding["cid"].apply(lambda x:get_xy(x,scale=6)[0])
    embedding["y_orig"] = embedding["cid"].apply(lambda x:get_xy(x,scale=6)[1])

    # 流形坐标均匀性
    X = embedding[["x","y"]].to_numpy()
    X_locc = embedding.loc[embedding.cid.isin(locc),["x","y"]].to_numpy()
    daX = calc_da(X_locc)
    print("R(by all average nearest distance):{}\n".format(calc_R(X,X_locc,
                                                            title="manifold_R_{}_subc{}_subu{}.png".format(method,subc,subu))))
    print("R(by locc average nearest distance):{}\n".format(calc_R(X,X_locc,d=daX,
                                                            title="manifold_R_{}_subc{}_subu{}_d{}.png".format(method,subc,subu,daX))))
    
    # 地图坐标均匀性
    X_orig = embedding[["x_orig","y_orig"]].to_numpy()
    X_locc_orig = embedding.loc[embedding.cid.isin(locc),["x_orig","y_orig"]].to_numpy()
    daX_orig = calc_da(X_locc_orig)
    print("R(by all average nearest distance):{}\n".format(calc_R(X_orig,X_locc_orig,
                                                            title="map_R_{}_subc{}_subu{}.png".format(method,subc,subu))))
    print("R(by locc average nearest distance):{}\n".format(calc_R(X_orig,X_locc_orig,d=daX_orig,
                                                            title="map_R_{}_subc{}_subu{}_d{}.png".format(method,subc,subu,daX_orig))))
