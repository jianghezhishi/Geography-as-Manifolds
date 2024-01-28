import datetime
from gurobipy import *
from loc_choice import get_xy, output_cover, get_subcover, greedy_nodup,\
    gurobi_cover, output_cover_res_by_rank, calculate_score, read_cover, output_res_int
import csv
from large_scale_covisit_manifold import *
from math import pi, cos, log
import numpy as np
import pandas as pd
#import folium
from shapely.geometry import shape, mapping
from shapely.geometry import Polygon
from shapely.wkt import dumps, loads
import json
from covisit_locck_sne import cut_cover_with_locc, read_locc
from covisit_sne import reverse, covisit_withhome_from_cover, sne_mat, output_res
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import seaborn as sns
import geopandas as gpd
from distribute_analysis import *

# 网格类
class LocationGrid:
    def __init__(self, xmin, xmax, ymin, ymax, gridscale) -> None:
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        self.gridscale = gridscale
        r = 6371*1000
        ymid = (ymin+ymax)/2
        r1 = r*cos(ymid/180*pi)
        self.xgap = gridscale/r1/pi*180
        self.ygap = gridscale/r/pi*180
        self.xnum = int((xmax-xmin)/self.xgap)+1
        self.ynum = int((ymax-ymin)/self.ygap)+1


def get_xy(cid, ynum=424, scale=1):
    ynum = int(np.ceil(ynum/scale))
    xid = int((cid-1)/ynum)
    yid = cid-xid*ynum-1
    return xid, yid


def get_wkt(cid, xmin=113.67561783007596, xgap=0.0014608354755808878,
            ymax=22.852485545898546, ygap=0.0013489824088780957, ynum=424, scale=1):
    xid, yid = get_xy(cid, scale=scale)
    x0 = xmin+xid*xgap*scale
    x1 = x0+xgap*scale
    y1 = ymax-yid*ygap*scale
    y0 = y1-ygap*scale
    wkt = 'POLYGON(('+str(x0)+' '+str(y0)+','+str(x1)+' '+str(y0)+','\
        + str(x1)+' '+str(y1)+','+str(x0)+' ' + \
        str(y1)+','+str(x0)+' '+str(y0)+'))'
    return wkt


# 集合覆盖选址，全覆盖，最小化需要门店数量，所以计算tot之后按从大到小取tot个门店
def gurobi_cover_mink(cover, con=True):
    """gurobi选址

    Args:
        cover (dict): 轨迹覆盖
        con (bool, optional): True连续近似，False整数规划. Defaults to True.

    Returns:
        x(dict): {cid:choice}每个格子被选址的情况，连续就是浮点
    """
    uid_cid = {}
    for cid in cover:
        for uid in cover[cid]:
            if uid in uid_cid:
                uid_cid[uid].add(cid)
            else:
                uid_cid[uid] = set([cid])
    print(len(uid_cid))
    uid_list = list(uid_cid.keys())
    cid_list = list(cover.keys())
    mat = {}
    for uid in uid_list:
        for cid in uid_cid[uid]:
            mat[(uid, cid)] = 1
    print('DATA LOADED')
    print(datetime.datetime.now())

    m = Model('COVER')
    if con:
        x = m.addVars(cid_list, vtype=GRB.CONTINUOUS, name='x')
        y = m.addVars(uid_list, vtype=GRB.CONTINUOUS, name='y')
        m.addConstrs(x[cid] <= 1 for cid in cid_list)
        m.addConstrs(x[cid] >= 0 for cid in cid_list)
        m.addConstrs(y[uid] >= 0 for uid in uid_list)
        m.addConstrs(y[uid] <= 1 for uid in uid_list)
        xy = m.addVars(mat.keys(), vtype=GRB.CONTINUOUS, name='xy')
    else:
        x = m.addVars(cid_list, vtype=GRB.BINARY, name='x')
        y = m.addVars(uid_list, vtype=GRB.BINARY, name='y')
        xy = m.addVars(mat.keys(), vtype=GRB.BINARY, name='xy')

    cost = m.addVar(vtype=GRB.CONTINUOUS, name='cost')

    m.setObjective(cost, GRB.MINIMIZE)
    m.addConstrs(xy[uid, cid] <= mat[(uid, cid)]*x[cid] for (uid, cid) in mat)
    m.addConstrs(y[uid] <= xy.sum(uid, '*') for uid in uid_list)
    m.addConstrs(y[uid] >= 0.99 for uid in uid_list)
    m.addConstr(cost == x.sum())

    print('PROBLEM FORMULATED')
    print(datetime.datetime.now())

    m.optimize()
    print('SOLVED')
    print(datetime.datetime.now())
    x = m.getAttr('x', x)

    tot = 0
    for cid in x:
        tot += x[cid]
    tot = int(tot)
    print(tot)
    return tot, x

# 把cover反过来，以uid为key
def reverse_bipartite(cover):
    user_cover = {}
    for cid in cover:
        for user in cover[cid]:
            user_cover[user] = user_cover.get(user, set()) | {cid}
    return user_cover

# 用户剪枝，剪掉前往不同格子过少的用户
def subuser(cover, k=2):
    user_cover = reverse_bipartite(cover)
    sub_user_cover = get_subcover(user_cover, k)
    # print(len(sub_user_cover))
    subuser_cover = reverse_bipartite(sub_user_cover)
    return subuser_cover


def plot_with_locc(locc, nodes, res, cover, title, color='cover'):
    """把选址画在流形上

    Args:
        locc (list): 选址结果
        nodes (list): 流形学习嵌入的节点列表
        res (ndarray): embedding坐标
        cover (dict): {cid:set(uid)}轨迹覆盖，用于画人口热力
        title (str): 输出标题
        color (str, optional): 节点颜色选项，设置颜色按照方位还是人口来可视化。cover则按人口热力画，lat按纬度，lon按经度. Defaults to 'cover'.
    # 修复了之前colorbar不随数值改变的问题
    """
    print('start')
    x_list, y_list, locc_x, locc_y = [], [], [], []  # 输入数据
    x0, x1, y0, y1 = 100, -100, 100, -100
    c = []  # 颜色
    maxnum, minnum = -100, 100000  # 颜色最大最小值
    for i in range(len(nodes)):  # 读取数据
        nod = nodes[i]
        lng, lat = get_xy(nod)
        co = len(cover[nod])
        if color == 'cover':
            num = log(co, 10)
            # num=co
            c.append(num)
        elif color == 'lng':
            c.append(lng)
            num = lng
        elif color == 'lat':
            c.append(lat)
            num = lat
        else:
            c.append(0)
            num = 0
        if num < minnum:
            minnum = num
        if num > maxnum:
            maxnum = num
        xy = res[i]
        x, y = xy[0], xy[1]
        x_list.append(x)
        y_list.append(y)
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
        if nod in locc:
            locc_x.append(x)
            locc_y.append(y)
    print(len(locc_x))

    # 计算合理的figsize
    xs, ys = x1-x0, y1-y0
    scale = 5/xs
    plt.figure(figsize=(6*2, ys*scale*2))

    cmp = plt.cm.get_cmap("Blues")
    norm = colors.Normalize()
    norm.autoscale(c)
    plt.scatter(x_list, y_list, s=10, c=c, cmap=cmp, edgecolors='none')
    plt.scatter(locc_x, locc_y, marker='o', edgecolors=(
        0.9, 0.1, 0.1, 0.7), s=20)  # 圈出选址结果
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmp),
                 label='Logarithm of population based on base 10')
    plt.clim(minnum, maxnum)

    plt.savefig(title, dpi=150)
    plt.show()
    return


# 选址流形接口
class LocationManifold:
    def __init__(self, grid: LocationGrid, cover_path, scale=1, subc=1, subu=1, save_folder=".") -> None:
        self.__grid = grid
        self.scale = scale
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder = save_folder
        self.__cover = read_cover(cover_path)
        self.gridls = LocationGrid(
            grid.xmin, grid.xmax, grid.ymin, grid.ymax, scale*grid.gridscale)
        self.subc = subc
        self.subu = subu
        coverls_path = '{}/coverls_cleantraj_subc{}_subu{}_scale{}_visit.csv'.format(
            save_folder, subc, subu, scale * grid.gridscale)

        if os.path.exists(coverls_path):
            self.coverls = read_cover(coverls_path)
        else:
            self.coverls = get_large_scale_cover(
                self.__cover, scale, int(np.ceil(grid.ynum / scale)))
            # 用户和格网剪枝
            if subu != 1:
                self.coverls = subuser(self.coverls, k=subu)
            if subc != 1:
                self.coverls = get_subcover(self.coverls, k=subc)
            output_cover(self.coverls, coverls_path)

    def set_params(self, subc=1, subu=1, scale=None):
        if scale is not None:
            self.scale = scale
        self.gridls = LocationGrid(self.__grid.xmin, self.__grid.xmax,
                                   self.__grid.ymin, self.__grid.ymax, self.scale*self.__grid.gridscale)
        self.subc = subc
        self.subu = subu
        coverls_path = '{}/coverls_cleantraj_subc{}_subu{}_scale{}_visit.csv'.format(
            self.save_folder, self.subc, self.subu, self.gridls.gridscale)
        if os.path.exists(coverls_path):
            self.coverls = read_cover(coverls_path)
        else:
            # 重置尺度
            self.coverls = get_large_scale_cover(
                self.__cover, self.scale, int(np.ceil(self.__grid.ynum / self.scale)))
            # 用户和格网剪枝
            if subu != 1:
                self.coverls = subuser(self.coverls, k=subu)
            if subc != 1:
                self.coverls = get_subcover(self.coverls, k=subc)
            output_cover(self.coverls, coverls_path)

    def location_choice(self, locc_method="con", k=300):
        self.locc_method = locc_method
        # 整数规划
        if locc_method == "int":
            title = '{}/ls_clean_int_home_subc{}_subu{}_cho{}_scale{}_visit.csv'.format(
                self.save_folder, self.subc, self.subu, k, self.gridls.gridscale)
            if os.path.exists(title):
                self.locc = read_locc(title)
            else:
                res_list, x = gurobi_cover(self.coverls, k, False)  # 2156备选
                self.locc = output_cover_res_by_rank(x, k, title)
            self.locc_method += "_k{}".format(k)
        # 连续近似
        if locc_method == "con":
            title = '{}/ls_clean_con_home_subc{}_subu{}_cho{}_scale{}_visit.csv'.format(
                self.save_folder, self.subc, self.subu, k, self.gridls.gridscale)
            if os.path.exists(title):
                self.locc = read_locc(title)
            else:
                res_list, x = gurobi_cover(self.coverls, k)  # 2156备选
                self.locc = output_cover_res_by_rank(x, k, title)
            self.locc_method += "_k{}".format(k)
        # 无重复贪心
        if locc_method == "greedy":
            title = '{}/ls_clean_greedy_nodup_home_subc{}_subu{}_cho{}_scale{}_visit.csv'.format(
                self.save_folder, self.subc, self.subu, k, self.gridls.gridscale)
            if os.path.exists(title):
                self.locc = read_locc(title)
            else:
                res = greedy_nodup(self.coverls, k)
                output_res_int(res, title)
                self.locc = res
            self.locc_method += "_k{}".format(k)
        # 全覆盖
        if locc_method == "allcover":
            title = '{}/ls_allcover_subc{}_subu{}_scale{}.csv'.format(
                self.save_folder, self.subc, self.subu, self.gridls.gridscale)
            if os.path.exists(title):
                self.locc = read_locc(title)
            else:
                tot, x = gurobi_cover_mink(self.coverls)
                self.locc = output_cover_res_by_rank(x, tot, title)
        # 全覆盖整数规划
        if locc_method == "allcover_int":
            title = '{}/ls_allcover_int_subc{}_subu{}_scale{}.csv'.format(
                self.save_folder, self.subc, self.subu, self.gridls.gridscale)
            if os.path.exists(title):
                self.locc = read_locc(title)
            else:
                tot, x = gurobi_cover_mink(self.coverls, False)
                self.locc = output_cover_res_by_rank(x, tot, title)

    def calc_interaction(self, interaction_method):
        self.interaction_method = interaction_method
        title = "{}/mat_{}_subc{}_subu{}_scale{}.npz".format(
            self.save_folder, self.interaction_method, self.subc, self.subu, self.gridls.gridscale)
        if os.path.exists(title):
            matfile = np.load(title)
            self.mat, self.nodes = matfile["mat"].tolist(
            ), matfile["nodes"].tolist()
        else:
            if interaction_method == "covisit":
                self.mat, self.nodes = covisit_withhome_from_cover(
                    self.coverls)
            elif interaction_method == "coverrate":
                self.mat, self.nodes = mat_by_coverrate(self.coverls)
            np.savez(title, mat=self.mat, nodes=self.nodes)

    def manifold_embedding(self):
        title = '{}/ls_manifold_{}_subc{}_subu{}_scale{}.csv'.format(
            self.save_folder, self.interaction_method, self.subc, self.subu, self.gridls.gridscale)
        if os.path.exists(title):
            self.embedding, self.nodes = read_manifold(title)
        else:
            self.embedding = sne_mat(self.mat)
            output_res(self.embedding, self.nodes, title)

    def update(self, interaction_method):
        self.calc_interaction(interaction_method)
        self.manifold_embedding()

    def visualize(self):
        title = '{}/ls_manifold_{}_{}_subc{}_subu{}_scale{}.png'.format(
            self.save_folder, self.interaction_method, self.locc_method, self.subc, self.subu, self.gridls.gridscale)
        plot_with_locc(self.locc, self.nodes, self.embedding,
                       self.coverls, title, color="cover")
    '''
    def map_visualize(self):#folium可交互地图
        m = folium.Map(location=[22.64, 114.04], zoom_start=10)
        for cid in self.nodes:
            # print(cid)
            polygon = loads(get_wkt(cid,
                                    xmin=self.__grid.xmin,
                                    xgap=self.__grid.xgap,
                                    ymax=self.__grid.ymax,
                                    ygap=self.__grid.ygap,
                                    ynum=self.__grid.ynum,
                                    scale=self.scale))
            xlst, ylst = polygon.exterior.xy
            lst = [[y, x] for x, y in zip(xlst, ylst)]
            fill_opacity = 1 if cid in self.locc else 0.2
            folium.Polygon(lst, fill_color='red',
                           fill_opacity=fill_opacity).add_to(m)
        return m
    '''
    def data_df(self):
        embedding = pd.DataFrame(
            {"cid": self.nodes, "x": self.embedding.T[0], "y": self.embedding.T[1]})
        embedding["x_orig"] = embedding["cid"].apply(
            lambda x: get_xy(x, ynum=self.__grid.ynum, scale=self.scale)[0])
        embedding["y_orig"] = embedding["cid"].apply(
            lambda x: get_xy(x, ynum=self.__grid.ynum, scale=self.scale)[1])
        embedding["locc"] = embedding["cid"].apply(
            lambda x: 1 if x in self.locc else 0)
        embedding["wkt"] = embedding["cid"].apply(get_wkt, scale=self.scale)
        return embedding


###########################################################################################
def geometry_intersect(gmanifold: LocationManifold, shp_file, attribute_name="NAME"):
    shapefile = gpd.read_file(shp_file)
    embedding = gmanifold.data_df()
    embedding["wkt"] = embedding["cid"].apply(get_wkt, scale=gmanifold.scale)

    # 创建一个空列来存储相交多边形的class信息
    embedding['intersects_class'] = ''

    # 将WKT多边形转换为几何对象
    embedding['geometry'] = embedding['wkt'].apply(loads)

    # 对每个数据多边形进行遍历
    for index, row in embedding.iterrows():
        # 检查数据多边形是否与任何SHP多边形相交
        intersects = shapefile.intersects(row['geometry'])

        # 如果有相交的多边形，则获取其class信息并添加到DataFrame中
        if intersects.any():
            intersecting_classes = shapefile.loc[intersects, attribute_name]
            embedding.at[index, 'intersects_class'] = list(intersecting_classes)[
                0]  # ','.join(intersecting_classes)

    return embedding


def distribute_analysis(gmanifold: LocationManifold):#均匀性分析，从distribute_analysis文件调用均匀性分析函数
    embedding = gmanifold.data_df()
    locc = gmanifold.locc
    # 流形坐标均匀性
    X = embedding[["x", "y"]].to_numpy()
    X_locc = embedding.loc[embedding.cid.isin(locc), ["x", "y"]].to_numpy()
    daX = calc_da(X_locc)
    folder, method, subc, subu = gmanifold.save_folder, gmanifold.interaction_method, gmanifold.subc, gmanifold.subu
    print("subc:{}  subu:{}\nmethod:{}\n".format(subc, subu, method))

    print("R(by all average nearest distance):{}\n".format(calc_R(X, X_locc,
                                                                  title="{}/manifold_R_{}_subc{}_subu{}.png".format(folder, method, subc, subu))))
    print("R(by locc average nearest distance):{}\n".format(calc_R(X, X_locc, d=daX,
                                                            title="{}/manifold_R_{}_subc{}_subu{}_d{}.png".format(folder, method, subc, subu, daX))))

    # 地图坐标均匀性
    X_orig = embedding[["x_orig", "y_orig"]].to_numpy()
    X_locc_orig = embedding.loc[embedding.cid.isin(
        locc), ["x_orig", "y_orig"]].to_numpy()
    daX_orig = calc_da(X_locc_orig)
    # calc_R会画图
    print("R(by all average nearest distance):{}\n".format(calc_R(X_orig, X_locc_orig,
                                                                  title="{}/map_R_{}_subc{}_subu{}.png".format(folder, method, subc, subu))))
    print("R(by locc average nearest distance):{}\n".format(calc_R(X_orig, X_locc_orig, d=daX_orig,
                                                            title="{}/map_R_{}_subc{}_subu{}_d{}.png".format(folder, method, subc, subu, daX_orig))))


def map_visualization(gmanifold: LocationManifold, shp_file, attribute_name="NAME"):
    embedding = geometry_intersect(gmanifold, shp_file, attribute_name)
    x_orig_range = embedding["x_orig"].max()-embedding["x_orig"].min()
    y_orig_range = embedding["y_orig"].max()-embedding["y_orig"].min()
    select_longitudes = [embedding["x_orig"].min(
    ) + int(i * x_orig_range/20) for i in range(20)]
    select_latitudes = [embedding["y_orig"].min(
    ) + int(i * y_orig_range/20) for i in range(20)]

    fig, ax = plt.subplots(2, 2, figsize=(36, 24))
    ax[0, 0].invert_yaxis()
    ax[0, 1].invert_yaxis()

    # cmap1 = sns.color_palette("flare_r", as_cmap=True)
    cmap1 = cm.get_cmap("spring")

    sns.scatterplot(x=embedding["x_orig"], y=embedding["y_orig"],
                    palette="pastel", hue=embedding["intersects_class"], ax=ax[0, 0])
    ax[0, 0].legend([], [], frameon=False)

    for i, long in enumerate(select_longitudes):
        longitude = embedding[embedding["x_orig"]
                              == long].sort_values("y_orig")
        ax[0, 0].plot(longitude["x_orig"], longitude["y_orig"], c=cmap1(i/20))
    ax[0, 0].set_title("longitudes")
    plt.colorbar(cm.ScalarMappable(cmap=cmap1), ax=ax[0, 0])

    # cmap2 = sns.color_palette("crest_r", as_cmap=True)
    cmap2 = cm.get_cmap("winter")

    sns.scatterplot(x=embedding["x_orig"], y=embedding["y_orig"],
                    palette="pastel", hue=embedding["intersects_class"], ax=ax[0, 1])
    ax[0, 1].legend([], [], frameon=False)
    for i, lat in enumerate(select_latitudes):
        latitudes = embedding[embedding["y_orig"] == lat].sort_values("x_orig")
        ax[0, 1].plot(latitudes["x_orig"], latitudes["y_orig"], c=cmap2(i/20))
    ax[0, 1].set_title("latitudes")
    plt.colorbar(cm.ScalarMappable(cmap=cmap2), ax=ax[0, 1])

    cmap2 = cm.get_cmap("spring")

    sns.scatterplot(x=-embedding["y"].to_numpy(), y=embedding["x"],
                    palette="pastel", hue=embedding["intersects_class"], ax=ax[1, 0])
    ax[1, 0].legend([], [], frameon=False)

    for i, long in enumerate(select_longitudes):
        longitude = embedding[embedding["x_orig"]
                              == long].sort_values("y_orig")
        ax[1, 0].plot(-longitude["y"].to_numpy(),
                      longitude["x"], c=cmap1(i/20))
    ax[1, 0].set_title("longitudes")
    plt.colorbar(cm.ScalarMappable(cmap=cmap1), ax=ax[1, 0])

    # cmap2 = sns.color_palette("crest_r", as_cmap=True)
    cmap2 = cm.get_cmap("winter")

    sns.scatterplot(x=-embedding["y"].to_numpy(), y=embedding["x"],
                    palette="pastel", hue=embedding["intersects_class"], ax=ax[1, 1])
    ax[1, 1].legend([], [], frameon=False)
    for i, lat in enumerate(select_latitudes):
        latitudes = embedding[embedding["y_orig"] == lat].sort_values("x_orig")
        ax[1, 1].plot(-latitudes["y"].to_numpy(),
                      latitudes["x"], c=cmap2(i/20))
    ax[1, 1].set_title("latitudes")
    plt.colorbar(cm.ScalarMappable(cmap=cmap2), ax=ax[1, 1])

    plt.savefig("{}/map_manifold_{}_subc{}_subu{}_scale{}.png".format(gmanifold.save_folder,
                                                                      gmanifold.interaction_method,
                                                                      gmanifold.subc,
                                                                      gmanifold.subu,
                                                                      gmanifold.gridls.gridscale))


def map_visualization_latlon(gmanifold: LocationManifold, shp_file, attribute_name="NAME", t1=4, t2=15):
    # 经纬线同时画上去，t1是原始地图断开阈值，即经纬线横跨空白多少个需要断开，t2是画成虚线的阈值，按流形距离计算
    embedding = geometry_intersect(gmanifold, shp_file, attribute_name)
    x_orig_range = embedding["x_orig"].max()-embedding["x_orig"].min()
    y_orig_range = embedding["y_orig"].max()-embedding["y_orig"].min()
    select_longitudes = [embedding["x_orig"].min(
    ) + int(i * x_orig_range/20) for i in range(20)]
    select_latitudes = [embedding["y_orig"].min(
    ) + int(i * y_orig_range/20) for i in range(20)]

    fig, ax = plt.subplots(1, 2, figsize=(30, 12))
    # ax[0].invert_yaxis()
    ax[0].invert_yaxis()

    # cmap1 = sns.color_palette("flare_r", as_cmap=True)

    sns.scatterplot(x=embedding["x_orig"], y=embedding["y_orig"],
                    palette="pastel", hue=embedding["intersects_class"], ax=ax[0])
    ax[0].legend([], [], frameon=False)

    cmap1 = cm.get_cmap("spring")
    cmap2 = cm.get_cmap("winter")
    # cmap1 = sns.color_palette("flare_r", as_cmap=True)
    # cmap2 = sns.color_palette("crest_r", as_cmap=True)

    lon_breaks = []
    lat_breaks = []
    for j, long in enumerate(select_longitudes):
        lon_breaks.append([])
        longitude = embedding[embedding["x_orig"]
                              == long].sort_values("y_orig")
        x, y = longitude["x_orig"].to_numpy(), longitude["y_orig"].to_numpy()
        for i in range(len(x) - 1):
            distance = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
            if distance > t1:
                # ax[0].plot([x[i], x[i+1]], [y[i], y[i+1]], linestyle=':', c=cmap1(j/20))
                lon_breaks[j].append(i)
            else:
                ax[0].plot([x[i], x[i+1]], [y[i], y[i+1]],
                           linestyle='-',  c=cmap1(j/20))
    ax[0].set_title("map")

    for j, lat in enumerate(select_latitudes):
        lat_breaks.append([])
        latitudes = embedding[embedding["y_orig"] == lat].sort_values("x_orig")
        x, y = latitudes["x_orig"].to_numpy(), latitudes["y_orig"].to_numpy()
        for i in range(len(x) - 1):
            distance = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
            if distance > t1:
                # ax[0].plot([x[i], x[i+1]], [y[i], y[i+1]], linestyle=':', c=cmap2(j/20))
                lat_breaks[j].append(i)
            else:
                ax[0].plot([x[i], x[i+1]], [y[i], y[i+1]],
                           linestyle='-',  c=cmap2(j/20))

    sns.scatterplot(x=-embedding["y"].to_numpy(), y=embedding["x"],
                    palette="pastel", hue=embedding["intersects_class"], ax=ax[1])
    ax[1].legend([], [], frameon=False)

    for j, long in enumerate(select_longitudes):
        longitude = embedding[embedding["x_orig"]
                              == long].sort_values("y_orig")
        x, y = -longitude["y"].to_numpy(), longitude["x"].to_numpy()
        for i in range(len(x) - 1):
            if i in lon_breaks[j]:
                continue
            distance = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
            if distance > t2:
                ax[1].plot([x[i], x[i+1]], [y[i], y[i+1]],
                           linestyle=':', c=cmap1(j/20))
            else:
                ax[1].plot([x[i], x[i+1]], [y[i], y[i+1]],
                           linestyle='-',  c=cmap1(j/20))
    ax[1].set_title("manifolds")

    for j, lat in enumerate(select_latitudes):
        latitudes = embedding[embedding["y_orig"] == lat].sort_values("x_orig")
        x, y = -latitudes["y"].to_numpy(), latitudes["x"].to_numpy()
        for i in range(len(x) - 1):
            if i in lat_breaks[j]:
                continue
            distance = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
            if distance > t2:
                ax[1].plot([x[i], x[i+1]], [y[i], y[i+1]],
                           linestyle=':', c=cmap2(j/20))
            else:
                ax[1].plot([x[i], x[i+1]], [y[i], y[i+1]],
                           linestyle='-',  c=cmap2(j/20))

    plt.savefig("{}/map_manifold_latlon_{}_subc{}_subu{}_scale{}.png".format(gmanifold.save_folder,
                                                                             gmanifold.interaction_method,
                                                                             gmanifold.subc,
                                                                             gmanifold.subu,
                                                                             gmanifold.gridls.gridscale))
