import sklearn.manifold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi,cos,ceil
from shapely.wkt import loads
import functools
from sklearn.metrics.pairwise import pairwise_distances
import statsmodels.api as sm
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.interpolate import griddata
import networkx as nx
import os

class Grid:
    def __init__(self,row_num,xmax,ymax,origin_cid,xmin=0,ymin=0):
        """格网对象，研究区域是一张格网内部的一个矩形选区

        Args:
            row_num (int): 整个格网每行格子数，即研究区域内，相邻两行格子之间的cid差
            xmax (int): 研究区一行格子数量-1，即最大x索引
            ymax (int): 研究区行数-1，即最大y索引
            origin_cid (int): 原点(0,0)对应的格网cid
            xmin (int, optional): 原点x索引. Defaults to 0.
            ymin (int, optional): 原点y索引. Defaults to 0.
        """
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
    
class PropagationManifold:
    """流形嵌入前的所有步骤
    """
    def __init__(self,scale,input_grid:Grid,center_fnid,data_path="ttcs_stay.csv",val="person",save_folder=".") -> None:
        self.scale = scale
        self.center_fnid = center_fnid
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder = save_folder
        
        row_num = ceil((input_grid.xmax+1)/scale)
        col_num = ceil((input_grid.ymax+1)/scale)
        grid = Grid(row_num,row_num-1,col_num-1,0)
        self.grid = grid

        # 构建网格fnid到交互矩阵索引列的字典
        cid2id = {} 
        k=0
        for j in range(self.grid.ymax + 1):
            for i in range(self.grid.xmax +1):
                cid2id[self.grid.xy_to_cid(i,j)] = k
                k += 1
        # 索引列到网格fnid
        self.cid2id = cid2id
        self.id2cid = {v:k for k,v in cid2id.items()}
        self.center_idx = cid2id[input_grid.merged_cid(self.center_fnid,scale)]
        
        # 读取驻留住址，构建单点交互矩阵
        ttcs_stay = pd.read_csv(data_path)
        # ttcs_stay是按O、D、驻留时间进行groupby的，按驻留时间groupby的目的是便于找到一组OD的最大驻留时间
        ttcs_stay_sum = ttcs_stay.groupby("home_fnid").sum().reset_index()[["home_fnid","stay_time","person"]]#第一列固定的天堂超市的id扔掉了
        ttcs_stay_sum.rename(columns={"home_fnid":"fnid"},inplace=True)
        ttcs_stay_max = ttcs_stay.groupby("home_fnid").max().reset_index()[["home_fnid","stay_time"]]#最大驻留时长
        ttcs_stay_max.rename(columns={"home_fnid":"fnid","stay_time":"stay_time_max"},inplace=True)
        df = pd.merge(left=ttcs_stay_sum,right=ttcs_stay_max,on="fnid")
        
        home = df["fnid"].to_numpy()
        interaction = df[val].to_numpy()
        radiation = np.zeros((len(cid2id),len(cid2id)))

        # 构建单点交互矩阵
        for i,fnid in enumerate(home):
            x,y = input_grid.cid_to_xy(fnid)
            if 0<=x<=input_grid.xmax and 0<=y<=input_grid.ymax and fnid>=input_grid.origin_cid and fnid != center_fnid: # 六环范围，去除自己前往自己
            # if grid.merged_cid(fnid,scale) in cid2id.keys(): # 这个判断句不确定对不对
                radiation[self.center_idx][cid2id[input_grid.merged_cid(fnid,scale)]] += interaction[i]
                radiation[cid2id[input_grid.merged_cid(fnid,scale)]][self.center_idx] += interaction[i]
        # for ii in range(len(cid2id)): # 对角元归零
        self.radiation = radiation
        
        # 制作邻接矩阵
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
        self.neighbor = neighbor_mat
        
        # 构造验证集
        cases = np.zeros(len(cid2id))
        df = pd.read_csv("cases_g1.csv")
        df.rename(columns={"cases_g1":"cases"},inplace=True)
        home = df["fnid"].to_numpy()
        fnid_cases = df["cases"].to_numpy()
        # cases也是基于栅格的，要按统计单元统计
        for i,fnid in enumerate(home):
            x,y = input_grid.cid_to_xy(fnid)
            if 0<=x<=input_grid.xmax and 0<=y<=input_grid.ymax and fnid>=input_grid.origin_cid: # 六环范围
            # if input_grid.merged_cid(fnid,scale) in cid2id.keys():
                cases[cid2id[input_grid.merged_cid(fnid,scale)]] += fnid_cases[i]
        self.cases = cases
    
    # 合成等效交互量    
    def calc_distance_mat(self,b):
        self.b = b
        # 合成交互量矩阵
        syc_interaction = self.radiation + b * self.neighbor
        # 等效距离
        eff_dis = 1 - np.log(syc_interaction / syc_interaction.max()+1e-16) # 防止无穷大,交互为0距离为1+36.841361487904734
        np.save("{}/syc_interaction_scale{}_b{}.npy".format(self.save_folder,self.scale,b),syc_interaction)
        np.save("{}/eff_distance_scale{}_b{}.npy".format(self.save_folder,self.scale,b),eff_dis)
        self.syc_interaction = syc_interaction
        self.eff_distance = eff_dis
    
    # 可视化（地图）
    def map_visualize(self,ax=None,figsize=(10,10),annotate_br=True):
        noax = ax is None
        map_location = np.zeros((len(self.cid2id),2))
        for cid,id in self.cid2id.items():
            x,y = self.grid.cid_to_xy(cid)
            map_location[id][0] = x
            map_location[id][1] = y
        if noax:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        ax.scatter(map_location[:, 0], map_location[:, 1], 
                    c=functools.reduce(lambda x,y:x+y,[[i/(self.grid.ymax+1)]*(self.grid.xmax+1) for i in range(self.grid.ymax+1)]), cmap='viridis',
                    s=10*self.cases+5
                    )
        ax.set_aspect("equal")
        ax.set_box_aspect(1)
        if noax:
            plt.savefig("{}/mapvis_scale{}.png".format(self.save_folder,self.scale))
            plt.show()
            
    # 更新嵌入参数
    def update(self,**kwarg):
        pass
    
    # 输出实证数据
    def cases_arr(self):
        return np.delete(self.cases,self.center_idx)
    
    def radiation_arr(self):
        return np.delete(self.radiation[self.center_idx],self.center_idx)
    
    def syc_interaction_arr(self):
        return np.delete(self.syc_interaction[self.center_idx],self.center_idx)
    
    def eff_distance_arr(self):
        return np.delete(self.eff_distance[self.center_idx],self.center_idx)
    
    def map_distance_arr(self):
        map_dist = np.zeros(len(self.cid2id))
        x_center,y_center = self.grid.cid_to_xy(self.center_idx)
        for cid,id in self.cid2id.items():
            x,y = self.grid.cid_to_xy(cid)
            map_dist[id] = np.sqrt((x-x_center)**2+(y-y_center)**2)
        return np.delete(map_dist,self.center_idx)
    
    def map_angle_arr(self):
        map_location = np.zeros((len(self.cid2id),2))
        x_center,y_center = self.grid.cid_to_xy(self.center_idx)
        for cid,id in self.cid2id.items():
            x,y = self.grid.cid_to_xy(cid)
            map_location[id][0] = x
            map_location[id][1] = y
        angle = np.arctan2(map_location[:, 1] - y_center, map_location[:, 0] - x_center)
        return np.delete(angle,self.center_idx)
    
    def manifold_distance_arr(self):
        distance = pairwise_distances(self.embedding, self.embedding[self.center_idx].reshape(1, -1)).flatten()
        return np.delete(distance,self.center_idx)
    
    def manifold_angle_arr(self):
        center_coord = self.embedding[self.center_idx]
        angle = np.arctan2(self.embedding[:, 1] - center_coord[1], self.embedding[:, 0] - center_coord[0])
        return np.delete(angle,self.center_idx)
            
    def inverse_interaction_arr(self,distance_arr):
        return self.syc_interaction.max()*(np.exp(1-distance_arr)-1e-16)
    
    def data_df(self) -> pd.DataFrame:
        pass
        
class IsomapPropagationManifold(PropagationManifold):
    def __init__(self, scale, input_grid: Grid, center_fnid, data_path="ttcs_stay.csv", val="person", save_folder=".") -> None:
        super().__init__(scale, input_grid, center_fnid, data_path, val, save_folder)
        self.method = "Isomap"
        self.nc = 2
    
    def set_nc(self,nc):
        self.nc = nc
        
    def calc_distance_mat(self,b):
        super().calc_distance_mat(b)
        # 记录Isomap特有的参量
        epsilon = 1e-16
        self.r_c1 = 1-np.log(b/self.syc_interaction.max()+1e-16) + epsilon # 邻接距离
        self.r_c2 = 1-np.log(1/self.syc_interaction.max()+1e-16) + epsilon # 最大交互距离
        
    def manifold_embedding(self,r):
        self.r = r
        manifold = sklearn.manifold.Isomap(n_components=self.nc,metric="precomputed",n_neighbors=None,radius=r)
        embedding = manifold.fit_transform(self.eff_distance)
        np.save("{}/embedding_{}_nc={}_b={}_r={}.npy".format(self.save_folder,
                                                           self.method,
                                                           self.nc,
                                                           self.b,
                                                           r),embedding)
        self.geodesic_distance = manifold.fit(self.eff_distance).dist_matrix_
        self.embedding = embedding
    
    # 更新r，b，nc参数
    def update(self,r,b,nc=2):
        self.set_nc(nc)
        self.calc_distance_mat(b)
        self.manifold_embedding(r)
    
    # 可视化
    def visualize(self,ax=None,figsize=(10,10),annotate_br=True):
        noax = ax is None
        if noax:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        ax.scatter(self.embedding[:, 0], self.embedding[:, 1], 
                    c=functools.reduce(lambda x,y:x+y,[[i/(self.grid.ymax+1)]*(self.grid.xmax+1) for i in range(self.grid.ymax+1)]), cmap='viridis',
                    s=10*self.cases+5
                    )
        ax.set_aspect("equal")
        ax.set_box_aspect(1)
        if annotate_br:
            ax.annotate("b={}\nr={:.3f}".format(self.b,self.r),(0.7,0.8),xycoords="axes fraction")
        if noax:
            plt.savefig("{}/manvis_{}_r={}_b={}.png".format(self.save_folder,self.method,self.r,self.b))
            plt.show()

    
    # 输出实证结果    
    def geodesic_distance_arr(self):
        return np.delete(self.geodesic_distance[self.center_idx],self.center_idx) # type: ignore
    
    def data_df(self):    
        # 因变量-病例数
        cases = self.cases_arr()
        
        # baseline-自然扩散距离、单点交互、反演地图交互
        map_dist = self.map_distance_arr()
        inv_map_inter = self.inverse_interaction_arr(map_dist)

        # 等效距离、流形距离、测地线距离
        eff_dist = self.eff_distance_arr()
        man_dist = self.manifold_distance_arr()
        gde_dist = self.geodesic_distance_arr()
        
        # 单点交互、合成交互、反演流形交互、反演测地线交互
        rad_inter = self.radiation_arr()
        syc_inter = self.syc_interaction_arr()
        inv_man_inter = self.inverse_interaction_arr(man_dist)
        inv_gde_inter = self.inverse_interaction_arr(gde_dist)
        
        data = pd.DataFrame({"cases":cases,
                            "map_distance":map_dist,
                            "radiation_interaction":rad_inter,
                            "effective_distance":eff_dist,
                            "manifold_distance":man_dist,
                            "geodesic_distance":gde_dist,
                            "synthetic_interaction":syc_inter,
                            "inverse_map_interaction":inv_map_inter,
                            "inverse_manifold_interaction":inv_man_inter,
                            "inverse_geodesic_interaction":inv_gde_inter})
        data_log1p = data.apply(np.log1p)
        data_log = data.applymap(lambda x:np.log(x) if x > 0 else np.nan)
        data = pd.concat([data,data_log.add_prefix("log_"),data_log1p.add_prefix("log1p_")],axis=1)
        return data
    
    def data_noze_df(self):
        noze = np.nonzero(self.cases_arr())
        data = self.data_df()
        return data.iloc[noze].copy()
    
class TSNEPropagationManifold(PropagationManifold):
    def __init__(self, scale, input_grid: Grid, center_fnid, data_path="ttcs_stay.csv", val="person", save_folder=".") -> None:
        super().__init__(scale, input_grid, center_fnid, data_path, val, save_folder)
        self.method = "TSNE"
        self.nc = 2
        
        # 地图距离
        map_distance = np.zeros((len(self.cid2id),len(self.cid2id)))
        for cid1,id1 in self.cid2id.items():
            x1,y1 = self.grid.cid_to_xy(cid1)
            for cid2,id2 in self.cid2id.items():
                x2,y2 = self.grid.cid_to_xy(cid2)
                if id1 != id2:
                    map_distance[id1][id2] = map_distance[id2][id1] = np.sqrt((x1-x2)**2+(y1-y2)**2)
        self.map_distance = map_distance
        
    
    def set_nc(self,nc):
        self.nc = nc
    
    def calc_distance_mat(self, b, t=36):
        super().calc_distance_mat(b)
        eff_distance = self.eff_distance
        G = nx.Graph()
        num_nodes = eff_distance.shape[0]
        G.add_nodes_from(range(num_nodes))

        # 构建图的边
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if eff_distance[i, j] <= t:
                    G.add_edge(i, j, weight=eff_distance[i, j])

        # 计算节点之间的最短路径
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))

        # 创建测地线距离矩阵
        geodesic_distance_mat = np.copy(eff_distance)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if eff_distance[i, j] > t:
                    geodesic_distance = shortest_paths[i][j]
                    geodesic_distance_mat[i, j] = geodesic_distance
                    geodesic_distance_mat[j, i] = geodesic_distance

        self.geodesic_distance =  geodesic_distance_mat
        
    def manifold_embedding(self,p):
        self.p = p
        manifold = sklearn.manifold.TSNE(n_components=self.nc,metric="precomputed",init="random",learning_rate="auto",
                                         random_state=111,perplexity=p)
        # 这里用了测地线距离生成流形
        embedding = manifold.fit_transform(self.geodesic_distance)
        np.save("{}/embedding_{}_nc={}_b={}_p={}.npy".format(self.save_folder,
                                                           self.method,
                                                           self.nc,
                                                           self.b,
                                                           p),embedding)
        self.embedding = embedding
    
    # 更新参数
    def update(self,b,p,nc=2):
        self.set_nc(nc)
        self.calc_distance_mat(b)
        self.manifold_embedding(p)
    
    # 可视化
    def visualize(self,ax=None,figsize=(10,10),annotate=True):
        noax = ax is None
        if noax:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        ax.scatter(self.embedding[:, 0], self.embedding[:, 1], 
                    c=functools.reduce(lambda x,y:x+y,[[i/(self.grid.ymax+1)]*(self.grid.xmax+1) for i in range(self.grid.ymax+1)]), cmap='viridis',
                    s=10*self.cases+5
                    )
        ax.set_aspect("equal")
        ax.set_box_aspect(1)
        if annotate:
            ax.annotate("b={}\np={:.3f}".format(self.b,self.p),(0.7,0.8),xycoords="axes fraction")
        if noax:
            plt.show()
            plt.savefig("{}/manvis_{}_p={}_b={}.png".format(self.save_folder,self.method,self.p,self.b))
    
    # 输出实证结果
    def data_df(self):    
        # 因变量-病例数
        cases = self.cases_arr()
        
        # baseline-自然扩散距离、单点交互、反演地图交互
        map_dist = self.map_distance_arr()
        inv_map_inter = self.inverse_interaction_arr(map_dist)

        # 等效距离、流形距离
        eff_dist = self.eff_distance_arr()
        man_dist = self.manifold_distance_arr()
        
        # 单点交互、合成交互、反演流形交互
        rad_inter = self.radiation_arr()
        syc_inter = self.syc_interaction_arr()
        inv_man_inter = self.inverse_interaction_arr(man_dist)
        
        data = pd.DataFrame({"cases":cases,
                            "map_distance":map_dist,
                            "radiation_interaction":rad_inter,
                            "effective_distance":eff_dist,
                            "manifold_distance":man_dist,
                            "synthetic_interaction":syc_inter,
                            "inverse_map_interaction":inv_map_inter,
                            "inverse_manifold_interaction":inv_man_inter})
        data_log1p = data.apply(np.log1p)
        data_log = data.applymap(lambda x:np.log(x) if x > 0 else np.nan)
        data = pd.concat([data,data_log.add_prefix("log_"),data_log1p.add_prefix("log1p_")],axis=1)
        return data
    
    def data_noze_df(self):
        noze = np.nonzero(self.cases_arr())
        data = self.data_df()
        return data.iloc[noze].copy()
            
        
#####################################################################################################    
############## 分析算法 ##############################################################################

def r_search(gmanifold:IsomapPropagationManifold,b):
    gmanifold.calc_distance_mat(b)
    r_c1 = gmanifold.r_c1
    r_c2 = gmanifold.r_c2
        
    # 1.等效距离分布直方图
    fig, ax = plt.subplots()
    sns.histplot(gmanifold.eff_distance_arr(),bins=50,ax=ax) # type: ignore
    ax.axvline(x=r_c1,c="r")
    ax.axvline(x=r_c2,c="r",linestyle="dashdot")
    plt.show()
    
    # 2.10步设置r可视化
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(24,10))
    for i, ax in enumerate(axes.flat):
        r = r_c1 + i*(r_c2-r_c1)/10 if i < 11 else 38
        gmanifold.manifold_embedding(r)
        gmanifold.visualize(ax=ax)
    fig.text(0.1, 0.9, "Interaction = radiation + b * neighbor, b={}, rc1={:.3f}, rc2={:.3f} ".format(b,r_c1,r_c2), fontsize=16, fontweight='bold')

    plt.savefig("{}/Isomap_rsearch_b={}".format(gmanifold.save_folder,gmanifold.b))
    plt.show()
    
def analyze_manifold(gmanifold:PropagationManifold,**kwargs):
    gmanifold.update(**kwargs)
    
    # 1.可视化
    gmanifold.visualize()
    
    # 2.散点，病例~距离，病例~交互
    
    data = gmanifold.data_df()
    #data_noze = gmanifold.data_noze_df()
    
    fig,axs = plt.subplots(2,4,figsize=(12,6))
    sns.scatterplot(x="map_distance",y="cases",data=data,alpha=0.2,ax=axs[0][0])
    sns.scatterplot(x="effective_distance",y="cases",data=data,alpha=0.2,ax=axs[0][1])
    sns.scatterplot(x="manifold_distance",y="cases",data=data,alpha=0.2,ax=axs[0][2])
    axs[0][3].remove()
    sns.scatterplot(x="inverse_map_interaction",y="cases",data=data,alpha=0.2,ax=axs[1][0])
    sns.scatterplot(x="synthetic_interaction",y="cases",data=data,alpha=0.2,ax=axs[1][1])
    sns.scatterplot(x="inverse_manifold_interaction",y="cases",data=data,alpha=0.2,ax=axs[1][2])
    sns.scatterplot(x="radiation_interaction",y="cases",data=data,alpha=0.2,ax=axs[1][3])
    
    plt.tight_layout()
    plt.show()
    
    fig,axs = plt.subplots(2,4,figsize=(12,6))
    axs[0][0].set(xscale="log")
    sns.scatterplot(x="map_distance",y="log1p_cases",data=data,alpha=0.2,ax=axs[0][0])
    axs[0][1].set(xscale="log")
    sns.scatterplot(x="effective_distance",y="log1p_cases",data=data,alpha=0.2,ax=axs[0][1])
    axs[0][2].set(xscale="log")
    sns.scatterplot(x="manifold_distance",y="log1p_cases",data=data,alpha=0.2,ax=axs[0][2])
    axs[0][3].remove()
    axs[1][0].set(xscale="log")
    sns.scatterplot(x="inverse_map_interaction",y="log1p_cases",data=data,alpha=0.2,ax=axs[1][0])
    axs[1][1].set(xscale="log")
    sns.scatterplot(x="synthetic_interaction",y="log1p_cases",data=data,alpha=0.2,ax=axs[1][1])
    axs[1][2].set(xscale="log")
    sns.scatterplot(x="inverse_manifold_interaction",y="log1p_cases",data=data,alpha=0.2,ax=axs[1][2])
    axs[1][3].set(xscale="log")
    sns.scatterplot(x="radiation_interaction",y="log1p_cases",data=data,alpha=0.2,ax=axs[1][2])
    plt.tight_layout()
    plt.show()

    
def analyze_isomap_manifold(gmanifold:IsomapPropagationManifold,b,r):
    gmanifold.update(b=b,r=r)
    
    # 1.可视化
    gmanifold.visualize()
    
    # 2.散点，病例~距离，病例~交互
    
    data = gmanifold.data_df()
    #data_noze = gmanifold.data_noze_df()
    
    fig,axs = plt.subplots(2,5,figsize=(15,6))
    sns.scatterplot(x="map_distance",y="cases",data=data,alpha=0.2,ax=axs[0][0])
    sns.scatterplot(x="effective_distance",y="cases",data=data,alpha=0.2,ax=axs[0][1])
    sns.scatterplot(x="manifold_distance",y="cases",data=data,alpha=0.2,ax=axs[0][2])
    sns.scatterplot(x="geodesic_distance",y="cases",data=data,alpha=0.2,ax=axs[0][3])
    axs[0][4].remove()
    sns.scatterplot(x="inverse_map_interaction",y="cases",data=data,alpha=0.2,ax=axs[1][0])
    sns.scatterplot(x="synthetic_interaction",y="cases",data=data,alpha=0.2,ax=axs[1][1])
    sns.scatterplot(x="inverse_manifold_interaction",y="cases",data=data,alpha=0.2,ax=axs[1][2])
    sns.scatterplot(x="inverse_geodesic_interaction",y="cases",data=data,alpha=0.2,ax=axs[1][3])
    sns.scatterplot(x="radiation_interaction",y="cases",data=data,alpha=0.2,ax=axs[1][4])
    plt.tight_layout()
    plt.show()
    
    fig,axs = plt.subplots(2,5,figsize=(15,6))
    axs[0][0].set(xscale="log")
    sns.scatterplot(x="map_distance",y="log1p_cases",data=data,alpha=0.2,ax=axs[0][0])
    axs[0][1].set(xscale="log")
    sns.scatterplot(x="effective_distance",y="log1p_cases",data=data,alpha=0.2,ax=axs[0][1])
    axs[0][2].set(xscale="log")
    sns.scatterplot(x="manifold_distance",y="log1p_cases",data=data,alpha=0.2,ax=axs[0][2])
    axs[0][3].set(xscale="log")
    sns.scatterplot(x="geodesic_distance",y="log1p_cases",data=data,alpha=0.2,ax=axs[0][3])
    axs[0][4].remove()
    axs[1][0].set(xscale="log")
    sns.scatterplot(x="inverse_map_interaction",y="log1p_cases",data=data,alpha=0.2,ax=axs[1][0])
    axs[1][1].set(xscale="log")
    sns.scatterplot(x="synthetic_interaction",y="log1p_cases",data=data,alpha=0.2,ax=axs[1][1])
    axs[1][2].set(xscale="log")
    sns.scatterplot(x="inverse_manifold_interaction",y="log1p_cases",data=data,alpha=0.2,ax=axs[1][2])
    axs[1][3].set(xscale="log")
    sns.scatterplot(x="inverse_geodesic_interaction",y="log1p_cases",data=data,alpha=0.2,ax=axs[1][3])
    axs[1][4].set(xscale="log")
    sns.scatterplot(x="radiation_interaction",y="log1p_cases",data=data,alpha=0.2,ax=axs[1][1])
    plt.tight_layout()
    plt.show()

def isotropy_test(gmanifold:PropagationManifold,ols_formula,merge_threshold=3,**kwargs):
    if kwargs:
        gmanifold.update(**kwargs)

    # 1.数据分组、回归
    data = gmanifold.data_df()
    # 分组合并函数
    def merge_classes(arr, threshold):
        unique_classes, class_counts = np.unique(arr, return_counts=True)
        num_classes = len(unique_classes)

        while num_classes > 1:
            sorted_indices = np.argsort(class_counts)  # 按类别数量升序排序的索引
            min_count_idx = sorted_indices[0]  # 数量最少的类别索引
            min_count = class_counts[min_count_idx]  # 数量最少的类别数量

            if np.min(class_counts) >= threshold:
                break
            
            # 检查左侧和右侧的类别数量，选择要合并的类别索引
            if min_count_idx == 0:  # 最左侧的类别
                merge_idx = min_count_idx + 1
            elif min_count_idx == num_classes - 1:  # 最右侧的类别
                merge_idx = min_count_idx - 1
            else:
                left_count = class_counts[min_count_idx - 1]
                right_count = class_counts[min_count_idx + 1]
                if left_count <= right_count:
                    merge_idx = min_count_idx - 1
                else:
                    merge_idx = min_count_idx + 1

            # 合并类别
            arr[arr == unique_classes[min_count_idx]] = unique_classes[merge_idx]

            # 更新类别数量
            class_counts[merge_idx] += min_count
            class_counts = np.delete(class_counts, min_count_idx)
            unique_classes = np.delete(unique_classes, min_count_idx)

            num_classes -= 1
        return arr

    # 分成8组
    if "map" in ols_formula:
        angle = gmanifold.map_angle_arr()
    else:
        angle = gmanifold.manifold_angle_arr()
    group = np.floor((angle+np.pi) / np.pi * 4)
    group[group == 8] = 7
    if merge_threshold is None:
        merge_threshold = 3
    g = merge_classes(group,threshold=merge_threshold)
    # 筛选同组角度，进行回归，得到回归参数和置信区间
    result_df = pd.DataFrame(columns=['Group', 'Coefficient', 'CI_lower', 'CI_upper'])
    categories = np.unique(g)
    # 针对每个类别进行回归分析
    for Group in categories:
        df = data[g == Group]

        # 构建并拟合线性回归模型
        model = ols(ols_formula,df)
        results = model.fit()
        
        # 提取回归系数和置信区间
        coef = results.params.iloc[1]
        ci = results.conf_int(alpha=0.05).iloc[1]  # 参数的95% 置信区间
        intercept = results.params.iloc[0]
        ci_inter = results.conf_int(alpha=0.05).iloc[0]  # 截距的95% 置信区间
        # 将结果添加到DataFrame中
        result_df = result_df.append({'Group': Group, 'Coefficient': coef, 
                                      'Intercept': intercept,
                                    'CI_lower': ci[0], 'CI_upper': ci[1], 
                                    'CI_inter_lower':ci_inter[0],
                                    'CI_inter_upper':ci_inter[1]}, ignore_index=True)

    # 全局回归
    model = ols(ols_formula,data)
    model_res = model.fit()
    resid = model_res.resid
    print(model_res.summary())
    # 第一幅散点用到的数据
    if "map" in ols_formula:
        manifold_distance = gmanifold.map_distance_arr()
    else:
        manifold_distance = gmanifold.manifold_distance_arr()
    cases_est = data[ols_formula[:ols_formula.find("~")].rstrip()]
    
    # 2.画图
    fig,axs = plt.subplots(2,2,figsize=(10,8))

    # 2.1.散点图
    all_groups = [i for i in range(8)]
    marker_styles = [
        "o",
        "X",
        (4, 0, 45),
        "P",
        (4, 0, 0),
        (4, 1, 0),
        "^",
        (4, 1, 45),]
    markers = {group: style for group, style in zip(all_groups, marker_styles)}
    for ii in range(9):
        axs[0,0].axhline(ii * np.pi/4 -np.pi, color='gray', linestyle=':')
    palette = sns.color_palette("Set1", n_colors=len(np.unique(g)))

    # 创建一个颜色字典，将每个类别映射到对应的颜色
    color_dict = {gid: palette[i] for i,gid in enumerate(np.unique(g))}
    edge_colors = [color_dict[val] for val in g]
    sns.scatterplot(x=manifold_distance,y=angle,hue=cases_est,palette="Spectral",style=g,markers=markers,s=60,ax=axs[0,0])
    axs[0,0].set_title(ols_formula[:ols_formula.find("~")].rstrip())
    axs[0,0].set_xlabel("Distance")
    axs[0,0].set_ylabel("Angle")
    axs[0,0].set_yticks([i * np.pi / 2-np.pi for i in range(5)],
                        labels=["$-\pi$","$-\pi/2$","0","$\pi/2$","$\pi$"])
    

    # 2.2. 分组残差图
    for gg in all_groups:
        idx = (g==gg)
        sns.swarmplot(x=g[idx],y=resid[idx],order=all_groups,marker=markers[gg],ax=axs[0,1])
    axs[0,1].axhline(0, color='gray', linestyle='--')
    axs[0,1].set_xlabel('Group')
    axs[0,1].set_ylabel('Residuals')
    axs[0,1].set_xticks([i for i in range(8)],labels=[i for i in range(8)])
    axs[0,1].set_title('Global Regression Residuals by Group')

    
    # 2.3. 分组回归参数置信区间
    axs[1,0].errorbar(result_df['Group'], result_df['Coefficient'],
                yerr=(result_df['Coefficient'] - result_df['CI_lower'],
                      result_df['CI_upper'] - result_df['Coefficient']),
                fmt='o', capsize=5, label='Coefficient')
    axs[1,0].axhline(model_res.params[1], color='gray', linestyle='--')
    ci_n = model_res.conf_int(alpha=0.05)
    axs[1,0].axhline(ci_n[1][1], color='gray', linestyle=':')
    axs[1,0].axhline(ci_n[0][1], color='gray', linestyle=':')

    axs[1,0].set_xlabel('Group')
    axs[1,0].set_ylabel('Coefficient')
    axs[1,0].set_title('Group Regression Coefficients with 95% CI')
    axs[1,0].legend()
    
    # 2.4. 分组回归截距置信区间
    axs[1,1].errorbar(result_df['Group'], result_df['Intercept'],
                yerr=(result_df['Intercept'] - result_df['CI_inter_lower'],
                      result_df['CI_inter_upper'] - result_df['Intercept']),
                fmt='o', capsize=5, label='Intercept')
    axs[1,1].axhline(model_res.params[0], color='gray', linestyle='--')
    ci_n = model_res.conf_int(alpha=0.05)
    axs[1,1].axhline(ci_n[1][0], color='gray', linestyle=':')
    axs[1,1].axhline(ci_n[0][0], color='gray', linestyle=':')

    axs[1,1].set_xlabel('Group')
    axs[1,1].set_ylabel('Intercept')
    axs[1,1].set_title('Group Regression Coefficients with 95% CI')
    axs[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    


def br_heatmap(gmanifold:IsomapPropagationManifold,blist,steps,ols_function="",ols_function_list=None,fig_size=(),title="",xlabels=[],ylabels=[],auto_vrange=False):
    mats = []
    if ols_function_list is not None:
        for f in ols_function_list:
            mats.append(np.zeros((len(blist),steps+2)))
    else:
        ols_function_list = [ols_function]
        mats.append(np.zeros((len(blist),steps+2)))
    for j,b in enumerate(blist):
        gmanifold.calc_distance_mat(b)
        r_c1 = gmanifold.r_c1
        r_c2 = gmanifold.r_c2
        print("b = {}, r_c1 = {}, r_c2 = {}".format(b,r_c1,r_c2))
        
        for i in range(steps+2):
            r = r_c1 + i*(r_c2-r_c1)/steps if i <= steps + 1 else 38
            gmanifold.manifold_embedding(r=r)
            data = gmanifold.data_df()
            data_noze = gmanifold.data_noze_df()
                        
            for id,ofc in enumerate(ols_function_list):
                if "log_cases" in ofc:
                    model_n = ols(ofc, data_noze).fit()
                else:
                    model_n = ols(ofc,data).fit()
                mats[id][j][i] = model_n.rsquared

    col = ["step {}".format(i) for i in range(steps + 1)] + ["MDS"]
    for idx,mat in enumerate(mats):
        df = pd.DataFrame(data=mat,columns=col,index=blist)
        fig,ax = plt.subplots(figsize=fig_size if fig_size else (15,8))
        if auto_vrange:
            sns.heatmap(df,annot=True,fmt=".4f",ax=ax)
        else:
            sns.heatmap(df,annot=True,fmt=".4f",ax=ax,vmin=0,vmax=1,cmap="rainbow")
        if not title:
            ax.set_title("$R^2$ of {}".format(ols_function_list[idx]))
        else:
            ax.set_title(title)
        ax.set_xlabel("r")
        ax.set_ylabel("b")
        if xlabels:
            ax.set_xticklabels(xlabels)
        if ylabels:
            ax.set_yticklabels(ylabels)
        fig.tight_layout()
        # fig.savefig("case_diffusion/br_heatmap.png")
        

if __name__ == "__main__":
    input_grid = Grid(714,217,216,76622)
    beijing_manifold = IsomapPropagationManifold(scale=18,input_grid=input_grid,center_fnid=153245)
    