# %%
from sklearn.manifold import Isomap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functools
from sklearn.metrics.pairwise import pairwise_distances
import statsmodels.api as sm
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# %% [markdown]
# 网格类

# %%
class Grid:
    def __init__(self,row_num,xmax,ymax,origin_cid,xmin=0,ymin=0):
        self.row_num = row_num # 行增量
        self.xmin = xmin 
        self.ymin = ymin                
        self.xmax = xmax # 一行格子数量-1
        self.ymax = ymax # 行数-1
        self.origin_cid = origin_cid       
    
    def xy_to_cid(self,x,y):
        return self.origin_cid + x + self.row_num * y
    
    def cid_to_xy(self,cid):
        x = (cid - self.origin_cid) % self.row_num + self.xmin
        y = int((cid - self.origin_cid) / self.row_num) + self.ymin
        return x, y
    
    def merged_cid(self,cid0,order):
        x,y = self.cid_to_xy(cid0)
        newx = int(x / order)
        newy = int(y / order)
        return newx + int((self.xmax-self.xmin) / order + 1) * newy
    


# %% [markdown]
# 原始网格

# %%
grid_0 = Grid(714,217,216,76622)


# %%
# ttcs所在的12级fnid编号
ttcs_fnid = 153145
grid_0.merged_cid(ttcs_fnid,18)

# %% [markdown]
# 用来进行流形学习的网格

# %%
# 用于学习的统计单元
grid = Grid(13,12,12,0)

# 构建网格fnid到交互矩阵索引列的字典
cid2id = {} 
k=0

for j in range(grid.ymax + 1):
    for i in range(grid.xmax +1):
        cid2id[grid.xy_to_cid(i,j)] = k
        k += 1

# 索引列到网格fnid
id2cid = {v:k for k,v in cid2id.items()}

# %%
ttcs_id = cid2id[grid_0.merged_cid(ttcs_fnid,18)]
ttcs_id

# %% [markdown]
# 核心代码

# %%
def calc_ttcs_radiation(cid2id,val="person"):
    """读取当晚驻留人员住址，生成驻留测度，基于栅格，计算合并到高阶格网单元后的单点交互量矩阵

    Args:
        cid2id (dict): {统计单元:交互矩阵index}
        val (str, optional): 测度 Defaults to "person".

    Returns:
        radiation (ndarray): 单点交互矩阵
    """
    ttcs_fnid = 153145
    radiation = np.zeros((len(cid2id),len(cid2id)))
    # 读取驻留住址
    ttcs_stay = pd.read_csv("ttcs_stay.csv")
    # ttcs_stay是按O、D、驻留时间进行groupby的，按驻留时间groupby的目的是便于找到一组OD的最大驻留时间
    ttcs_stay_sum = ttcs_stay.groupby("home_fnid").sum().reset_index()[["home_fnid","stay_time","person"]]
    ttcs_stay_sum.rename(columns={"home_fnid":"fnid"},inplace=True)
    ttcs_stay_max = ttcs_stay.groupby("home_fnid").max().reset_index()[["home_fnid","stay_time"]]
    ttcs_stay_max.rename(columns={"home_fnid":"fnid","stay_time":"stay_time_max"},inplace=True)
    df = pd.merge(left=ttcs_stay_sum,right=ttcs_stay_max,on="fnid")
    
    home = df["fnid"].to_numpy()
    interaction = df[val].to_numpy()
    # 构建单点交互矩阵
    for i,fnid in enumerate(home):
        x,y = grid_0.cid_to_xy(fnid)
        if 0<=x<=217 and 0<=y<=216 and fnid>=76622 and fnid != ttcs_fnid: # 六环范围，去除自己前往自己
        # if grid_0.merged_cid(fnid,18) in cid2id.keys(): # 这个判断句不确定对不对
            radiation[cid2id[grid_0.merged_cid(ttcs_fnid,18)]][cid2id[grid_0.merged_cid(fnid,18)]] += interaction[i]
            radiation[cid2id[grid_0.merged_cid(fnid,18)]][cid2id[grid_0.merged_cid(ttcs_fnid,18)]] += interaction[i]
    # for ii in range(len(cid2id)): # 对角元归零
    #     radiation[ii][ii] = 0
    return radiation

def calc_neighbor_mat(cid2id):
    """根据统计单元格网制作邻接矩阵

    Args:
        cid2id (dict): {统计单元:矩阵id}

    Returns:
        neighbor_mat (ndarray): 邻接矩阵
    """
    neighbor_mat = np.zeros((len(cid2id),len(cid2id)))
    for cid in cid2id.keys():
        x,y = grid.cid_to_xy(cid)
        # 如果不是边缘，则xy坐标相邻的设为1
        if x != grid.xmax:
            neighbor_mat[cid2id[grid.xy_to_cid(x,y)]][cid2id[grid.xy_to_cid(x+1,y)]] = 1
        if y != grid.ymax:
            neighbor_mat[cid2id[grid.xy_to_cid(x,y)]][cid2id[grid.xy_to_cid(x,y+1)]] = 1
        if x != 0:
            neighbor_mat[cid2id[grid.xy_to_cid(x,y)]][cid2id[grid.xy_to_cid(x-1,y)]] = 1
        if y != 0:
            neighbor_mat[cid2id[grid.xy_to_cid(x,y)]][cid2id[grid.xy_to_cid(x,y-1)]] = 1
    return neighbor_mat
        

def calc_distance_mat(radiation,neighbor,b):
    """基于单点交互量和邻接矩阵计算合成交互量，构建距离

    Args:
        radiation (ndarray): 单点交互量
        neighbor (ndarray): 邻接矩阵
        b (double): 合成参数

    Returns:
        eff_dis (ndarray): 有效距离矩阵
    """
    # 合成交互量矩阵
    syc_interaction = radiation + b * neighbor
    # 等效距离
    eff_dis = 1 - np.log(syc_interaction / syc_interaction.max()+0.0000000000000001) # 防止无穷大,交互为0距离为1+36.841361487904734
    return eff_dis

def manifold_embedding(dist_mat,nc,r,save_path="embedding.csv"):
    """流形嵌入，返回统计单元各个点坐标

    Args:
        dist_mat (ndarray): 距离矩阵
        nc (int): 嵌入维度数
        r (double): 搜索距离
        save_path (str, optional): 保存路径. Defaults to "embedding.csv".

    Returns:
        embedding (ndarray): 嵌入坐标,shape: len(cid2id) * nc
    """
    # 流形学习嵌入
    isomap = Isomap(n_components=nc,n_neighbors = None,radius=r,metric='precomputed')
    embedding = isomap.fit_transform(dist_mat)
    print(isomap.reconstruction_error())
    
    # 生成样本索引列
    index_col = np.arange(dist_mat.shape[0]).reshape(-1, 1)
    cid_col = np.vectorize(lambda x: id2cid[x])(index_col)
    # 将index_col和embedding数组按列合并
    result = np.hstack((cid_col, embedding))
    col_names = ['x{}'.format(i+1) for i in range(nc)]
    # 将结果转换为pandas DataFrame
    df = pd.DataFrame(result, columns=['index']+col_names)
    # 将结果保存为csv文件
    df.to_csv(save_path, index=False)
    return embedding

def correlation(x1,x2):
    results = sm.OLS(x1,x2).fit()
    print(results.summary())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    sns.boxplot(x=x1, y=x2, whis=[0, 100], width=.6, palette="vlag", ax=ax1) # 箱线图
    sns.regplot(x=x1, y=x2, ax=ax2,scatter_kws={'alpha':0.2},fit_reg=False) # 散点图和OLS直线
    fig.tight_layout()
    fig.show()

def evaluation(dist_mat,embedding,cases):
    # 1 distance-interaction scatter, 和reconstruction_error有关
    embedding_dis = pairwise_distances(embedding,metric="euclidean")
    ttcs_fnid = 153145
    ttcs_id = cid2id[grid_0.merged_cid(ttcs_fnid,18)]
    embedding_radiation_arr = embedding_dis[ttcs_id]
    orig_dis_arr = dist_mat[ttcs_id]
    correlation(orig_dis_arr,embedding_radiation_arr)
    # 2 同心圆，要求病例和距离负相关，病例和方位角无关
    # distance - cases
    correlation(embedding_radiation_arr,cases)

# %% [markdown]
# 验证集，即病例cases

# %%
def get_cases():
    cases = np.zeros(len(cid2id))
    df = pd.read_csv("cases_g1.csv")
    df.rename(columns={"cases_g1":"cases"},inplace=True)
    home = df["fnid"].to_numpy()
    fnid_cases = df["cases"].to_numpy()
    # cases也是基于栅格的，要按统计单元统计
    for i,fnid in enumerate(home):
        x,y = grid_0.cid_to_xy(fnid)
        if 0<=x<=217 and 0<=y<=216 and fnid>=76622: # 六环范围
        # if grid_0.merged_cid(fnid,18) in cid2id.keys():
            cases[cid2id[grid_0.merged_cid(fnid,18)]] += fnid_cases[i]
    return cases


cases = get_cases() # ndarray

# %% [markdown]
# 单次rb指数回归

# %%
def ra_reg(b,r):
    radiation = calc_ttcs_radiation(cid2id)
    neighbor = calc_neighbor_mat(cid2id)
    dist_mat = calc_distance_mat(radiation,neighbor, b)
    embedding = manifold_embedding(dist_mat,2,r)

    plt.figure(figsize=(10,10))
    plt.scatter(embedding[:, 0], embedding[:, 1], 
                c=functools.reduce(lambda x,y:x+y,[[i/(grid.ymax+1)]*(grid.xmax+1) for i in range(grid.ymax+1)]), cmap='viridis',
                s=10*cases+0.2
                )
    plt.axis("equal")
    plt.xlim([-20,35])
    plt.ylim([-20,35])

    plt.annotate("b={}\r={}".format(b,r),(0.7,0.8),xycoords="figure fraction")
    # plt.savefig("case_diffusion/r={}_b={}.png".format(r,b))
    plt.show()

    ttcs_fnid = 153145
    center_idx = cid2id[grid_0.merged_cid(ttcs_fnid,18)]
    center_coord = embedding[center_idx]
    distance = pairwise_distances(embedding, embedding[center_idx].reshape(1, -1)).flatten()
    angle = np.arctan2(embedding[:, 1] - center_coord[1], embedding[:, 0] - center_coord[0])
    cases_nottcs = np.delete(cases,center_idx)
    distance = np.delete(distance,center_idx)
    angle = np.delete(angle,center_idx)

    # 将数据转换为pandas DataFrame格式
    data = pd.DataFrame({'value': cases_nottcs, 'distance': distance, 'angle': angle})

    # 使用线性回归模型，同时控制方位角
    model = ols('value ~ distance + angle', data).fit()

    # 进行方差分析
    anova_table = sm.stats.anova_lm(model, typ=2)

    # 打印结果
    print(anova_table)
    print(model.summary())

    fig,ax = plt.subplots(2,2,figsize=(12,12))
    sns.scatterplot(x=distance,y=angle,c=np.log(cases_nottcs+1),ax=ax[0][0])
    ax[0][0].set_xlabel("distance")
    ax[0][0].set_ylabel("angle")
    sns.scatterplot(x=cases_nottcs,y=angle,ax=ax[0][1],alpha=0.15)
    ax[0][1].set_xlabel("cases")
    ax[0][1].set_ylabel("angle")
    sns.scatterplot(x=distance,y=cases_nottcs,ax=ax[1][0],alpha=0.15)
    ax[1][0].set_xlabel("distance")
    ax[1][0].set_ylabel("cases")
    noze = np.nonzero(cases_nottcs)
    sns.regplot(x=distance[noze], y=np.log(cases_nottcs[noze]), ax=ax[1][1],scatter_kws={'alpha':0.2},fit_reg=True)
    ax[1][1].set_xlabel("distance")
    ax[1][1].set_ylabel("log(cases)")
    plt.tight_layout()
    plt.show()
    
    # 非零回归
    data_noze = pd.DataFrame({'value': np.log(cases_nottcs[noze]), 'distance': distance[noze], 'angle': angle[noze]})
    model_n = ols('value ~ distance', data_noze).fit()
    print(model_n.rsquared)
    print(model_n.summary())
    model_2 = ols('value ~ distance + angle', data_noze).fit()
    print(model_2.summary())


# %% [markdown]
# 测地线距离回归

# %%
def geodesic_reg(b,r,no_zero=True):
    epsilon = 1e-16
    radiation = calc_ttcs_radiation(cid2id)
    neighbor = calc_neighbor_mat(cid2id)
    r_c1 = 1-np.log(b/(b+radiation.max())+1e-16) + epsilon # 邻接距离
    r_c2 = 1-np.log(1/(b+radiation.max())+1e-16) + epsilon # 最大交互距离
    dist_mat = calc_distance_mat(radiation,neighbor, b) # 等效距离
    print("b = {}, r_c1 = {}, r_c2 = {}".format(b,r_c1,r_c2))
    isomap = Isomap(n_components=2,n_neighbors = None,radius=r,metric='precomputed')
    geodesic_dist = isomap.fit(dist_mat).dist_matrix_ # 测地线距离
    
    ttcs_fnid = 153145
    center_idx = cid2id[grid_0.merged_cid(ttcs_fnid,18)]
    distance = geodesic_dist[center_idx]
    # 去除ttcs所在统计单元
    cases_nottcs = np.delete(cases,center_idx) 
    distance = np.delete(distance,center_idx)
    if no_zero:
        noze = np.nonzero(cases_nottcs)
        cases_nottcs = cases_nottcs[noze]
        distance = distance[noze]
    # 反演合成交互量
    # data_noze = pd.DataFrame({'value': cases_nottcs, 'distance': (radiation.max())*(np.exp(1-distance)-1e-16)}) # 这里注意一下radiation也是矩阵
    data_noze = pd.DataFrame({'value': cases_nottcs, 'distance': ((radiation + b * neighbor).max())*(np.exp(1-distance)-1e-16)}) # 这里注意一下radiation也是矩阵
    model_n = ols('value ~ distance - 1', data_noze).fit()
    print(model_n.rsquared)
    print(model_n.summary())


# %%
ra_reg(700,7.345)

# %%
geodesic_reg(100,6.76)

# %%



