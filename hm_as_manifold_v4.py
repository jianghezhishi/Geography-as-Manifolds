'''
将回归改为对第6个点，而非整个邻域
'''
import csv
import numpy as np
import statsmodels.api as sm
from math import cos, pi
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from vedo import Points, show, Grid, Plotter, Line, Mesh
import vedo
import matplotlib
from scipy.optimize import minimize
import networkx as nx
from sklearn.manifold import MDS, cons_smacof
import os
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind

csv.field_size_limit(1024 * 1024 * 100)
plt.rcParams['font.size']=16
def get_tri(cp_lam,cid_s):
    #找出连边组成的所有三角形
    tris=set()
    step=int(len(cp_lam)/100+1)
    count=0
    for cp in cp_lam:
        if count%step==0:
            print(count/step)
        count+=1
        cid1,cid2=cp
        for cid3 in cid_s:
            if cid3!=cid1 and cid3!=cid2 and \
                    ((cid1,cid3) in cp_lam or (cid3,cid1) in cp_lam) and\
                    ((cid2,cid3) in cp_lam or (cid3,cid2) in cp_lam):
                tri=[cid1,cid2,cid3]
                tri.sort()
                tri=tuple(tri)
                tris.add(tri)
    print('tris ready',len(tris))
    return tris
def cal_normlam(cid1,cid2,cp_lam,cid_s, cid_k, sigmaind,kappaind):
    sisj = ((cid_s[cid2] * cid_s[cid1])**sigmaind  * (cid_k[cid2] * cid_k[cid1]) **kappaind)
    lam=cp_lam.get((cid2, cid1), 0) + cp_lam.get((cid1, cid2), 0)
    return lam/sisj
def cal_rad(cp_lam, cid_s, cid_k, sigmaind,kappaind, ep,tiv=0.01,title=r'E:\流形研究\20250317代码整理\tivcurve.npy'):
    # 计算距离函数中的R参数，依据是最大归一化交互量（即交互/节点度幂次）
    # 还有两种可能性，分别是按照三角不等式，以及按照双曲

    # 基于使所有距离大于0的R
    R = -1000
    for cp in cp_lam:
        cid1, cid2 = cp
        norml = np.log(cal_normlam(cid1,cid2,cp_lam,cid_s, cid_k, sigmaind,kappaind))
        if R < norml:
            R = norml
    R1=R
    print('R by nonneg',R1)
    # 基于三角不等式的R，参数tiv表示允许多少比例三角不等式不成立
    tris=list(get_tri(cp_lam,cid_s))
    tri_Rthre={}
    step=int((len(tris)/50)+1)
    count=0
    for tri in tris:
        if count%step==0:
            print(count/step)
        count+=1
        tri=tuple(tri)
        cid1,cid2,cid3=tri
        edg1=-np.log(cal_normlam(cid1,cid2,cp_lam,cid_s, cid_k, sigmaind,kappaind))
        edg2=-np.log(cal_normlam(cid1,cid3,cp_lam,cid_s, cid_k, sigmaind,kappaind))
        edg3=-np.log(cal_normlam(cid3,cid2,cp_lam,cid_s, cid_k, sigmaind,kappaind))
        edgs=[edg1,edg2,edg3]
        edgs.sort(reverse=True)
        tri_Rthre[tri]=edgs[0]-(edgs[1]+edgs[2])
    stemp=sorted(tri_Rthre.items(),key=lambda x:x[1],reverse=True)
    R2=stemp[int(tiv*len(stemp))][1]
    print('R by tri', R2)
    data=[]
    for i in stemp:
        data.append(i[1])
    np.save(title,np.array(data))


    R=max(R1,R2)
    print('R', R)
    return R


def cover_to_rad(cp_lam,cid_s,cid_k,title, R=1, inf=100, ep=0.1, sigmaind=1,kappaind=1, top=6, option='reg', \
                  figtitle=r'E:\流形研究\20250110基于回归的距离\temp.png', show=False,tiv=0.01,\
                 titletiv=r'E:\流形研究\20250317代码整理\tivcurve.npy'):
    # 将交互矩阵转化为距离，基于回归计算参数sigmaind和kappaind，即隐藏节点强度和隐藏节点度的指数
    # minpop用于给格子剪枝
    # R是双曲距离中的半径，设定为"auto"时调用cal_rad自动计算
    # maxd是用于计算流量阈值的参数，只有流量/质量幂大于阈值，才保留节点对，即给节点对剪枝
    # inf是无交互的填充值
    # ep是防止距离为负的参数，将距离过大的部分设置为R-log(e**R-ep)
    # alphaind是双曲加权论文中alpha决定的系数，即距离是归一化交互取对数后的线性
    # top表示knn中的邻域大小
    # opt用以选择初始化回归、迭代回归或最终回归alphaind
    # file是输入数据
    # cover的数据形式，在daas数据提取的三元组上也可以实现，可以以cover为统一的输入
    # figtitle是回归散点可视化存储路径
    maxd=inf

    nodes = list(cid_s.keys())
    #print(sigmaind,kappaind,cid_s)

    if option == 'initreg':  # 初始回归，不清楚应该给两个系数如何赋值，只能把所有数据输入
        xs = []
        ys = []  # 回归有一个问题，在参数未知的时候，不知道哪些数据是top
        for i in range(len(nodes)):
            cid = nodes[i]
            for j in range(len(nodes)):
                cid1 = nodes[j]
                if cid != cid1:
                    lam = cp_lam.get((cid, cid1), 0) + cp_lam.get((cid1, cid), 0)
                    #并非双向加和定义，数据是无向od（重复覆盖），但前面计算时节省时间和内存只算了上三角
                    if lam != 0:
                        xs.append([np.log(cid_s[cid] * cid_s[cid1]), np.log(cid_k[cid] * cid_k[cid1])])
                        #xs.append(np.log([cid_s[cid] * cid_s[cid1]/cid_k[cid] / cid_k[cid1]]))
                        ys.append(np.log(lam))
        #print(xs)
        xs = np.array(xs)
        X = sm.add_constant(xs)
        #print(X)
        model = sm.OLS(ys, X)
        results = model.fit()
        print(results.params)
        print(results.summary())
        y_fitted = results.fittedvalues
        plt.plot(y_fitted, ys, 'o', markersize=1, color=(1, 0, 0, 0.1))
        plt.savefig(figtitle, dpi=150)
        if show:
            plt.show()
        return round(results.params[1],4),round(results.params[2],4)
        #return results.params[1],results.params[2]
    elif option == 'reg':
        xs = []
        ys = []
        for i in range(len(nodes)):
            cid = nodes[i]
            temp = {}
            for j in range(len(nodes)):
                cid1 = nodes[j]
                if cid != cid1:
                    lam = cp_lam.get((cid, cid1), 0) + cp_lam.get((cid1, cid), 0)
                    if lam != 0:
                        temp[cid1] = ((cid_s[cid] * cid_s[cid1])**sigmaind * (
                                cid_k[cid] * cid_k[cid1])**kappaind)  / lam
            stemp = sorted(temp.items(), key=lambda x: x[1])
            for _ in [min(len(stemp)-1, top-2)]:  # 这里取top以内的所有节点，因此不完全是半径，也是为了照顾小邻域的情况
                item = stemp[_]
                cid1 = item[0]
                lam = cp_lam.get((cid, cid1), 0) + cp_lam.get((cid1, cid), 0)
                xs.append([np.log(cid_s[cid] * cid_s[cid1]), np.log(cid_k[cid] * cid_k[cid1])])
                ys.append(np.log(lam))
        xs = np.array(xs)
        X = sm.add_constant(xs)
        model = sm.OLS(ys, X)
        results = model.fit()
        print(results.params)
        print(results.summary())
        y_fitted = results.fittedvalues
        plt.plot(y_fitted, ys, 'o', markersize=1, color=(1, 0, 0, 0.1))
        plt.savefig(figtitle, dpi=150)
        if show:
            plt.show()
        return round(results.params[1],4),round(results.params[2],4)
    else:
        countbel0 = 0
        if R == 'auto':
            R1 = cal_rad(cp_lam, cid_s, cid_k, sigmaind,kappaind, ep,tiv,titletiv)
        else:
            R1=R
        ep=np.e**(R1-1)
        thre = np.exp(-maxd+R1)
        with open(title, 'w', newline='') as f:
            wt = csv.writer(f)
            wt.writerow([nodes])
            for i in range(len(nodes)):
                cid = nodes[i]
                row = []
                for j in range(len(nodes)):
                    cid1 = nodes[j]
                    if cid == cid1:
                        row.append(0)
                    else:
                        lam = cp_lam.get((cid, cid1), 0) + cp_lam.get((cid1, cid), 0)
                        sisj = ((cid_s[cid] * cid_s[cid1])**sigmaind* (cid_k[cid] * cid_k[cid1])**kappaind)
                        if np.e ** R1 - ep >= lam / sisj > thre:
                            row.append(R1 + np.log(sisj / lam))
                        elif lam / sisj > np.e ** R1 - ep:
                            #当交互量过大，使该定义下会出现负距离；这一问题在自动R定义下不会出现
                            # print(cid,cid1,lam,cid_s[cid],cid_s[cid1],cid_k[cid],cid_k[cid1])
                            row.append(R1 - np.log(np.e ** R1 - ep))
                            countbel0 += 1
                        else:#当交互量过小，thre是按maxd计算的，小于thre将导致距离大于maxd
                            row.append(inf)
                wt.writerow([row])
        print(countbel0)
        return R1



def read_mat(file, maxd, sigma=6, dmax=4):
    mask = []
    mat = []
    with open(file, 'r') as f:
        rd = csv.reader(f)
        header = next(rd)
        nodes = eval(header[0])
        count = 0
        for row in rd:
            cid0 = nodes[count]
            temp = eval(row[0])
            mat.append(temp)
            if sigma != None:
                dict0 = {}
                for i in range(len(temp)):
                    dict0[i] = temp[i]
                stemp = sorted(dict0.items(), key=lambda x: x[1])
                for i in range(sigma):
                    if 0 < stemp[i][1] < maxd:
                        cid1 = nodes[stemp[i][0]]
                        mask.append((cid0, cid1))
            else:
                for i in range(len(temp)):
                    if 0 < temp[i] < dmax:
                        cid1 = nodes[i]
                        mask.append((cid0, cid1))
            count += 1
    return nodes, mat, mask



def output_res(res, nodes, title):
    with open(title, 'w', newline='') as f:
        wt = csv.writer(f)
        for i in range(len(nodes)):
            wt.writerow([nodes[i], res[i]])
    return


def read_manifold(file, nc=2, opt=1):
    res, nodes = [], []
    with open(file, 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            temp = row[1][1:-1]
            try:
                cid = int(row[0])#这里需要解决，有些数据的cid是字符串，有些是int的问题，因为读取矩阵时使用eval读取nodes，因此当cid是int时需要在这里也按int读取
            except:
                cid=row[0]
            else:
                pass
            if opt == 1:
                xy = []
                flag = 0
                for it in temp.split(' '):
                    try:
                        num = float(it)
                    except:
                        pass
                    else:
                        flag += 1
                        xy.append(num)
                if flag == nc:
                    nodes.append(cid)
                    res.append(xy)
                else:
                    print('err')
            else:
                nodes.append(cid)
                res.append(eval(row[1]))
    return res, nodes

def find_intersections(x, y1, y2):
    x = np.array(x)
    y1 = np.array(y1)
    y2 = np.array(y2)

    # 两条折线的差值
    diff = y1 - y2

    intersections = []
    for i in range(len(x) - 1):
        if diff[i] * diff[i + 1] < 0:  # 存在交点（符号变化）
            # 线性插值计算交点位置
            x0, x1_ = x[i], x[i + 1]
            y0, y1_ = diff[i], diff[i + 1]
            x_intersect = x0 - y0 * (x1_ - x0) / (y1_ - y0)

            # 插值算出对应的 y 值（可选）
            y_intersect = y1[i] + (y1[i + 1] - y1[i]) * (x_intersect - x0) / (x1_ - x0)

            intersections.append((x_intersect, y_intersect))

    if not intersections:
        return None  # 无交点
    else:
        # 返回最右边的交点
        return max(intersections, key=lambda pt: pt[0])

def topo_fit(nodes, mask, res, title, title1, part=False, win=10, show=True,temptitle=r'ita/topo.npy'):
    print(len(nodes), len(res))
    if not os.path.exists(temptitle):
        print('making topo data')
        if part == True:
            nodes1, res1 = [], []
            for i in range(len(nodes)):
                cid = nodes[i]
                xy = res[i]
                if xy[0] < -25 and xy[1] < 0:
                    nodes1.append(cid)
                    res1.append(xy)
            nodes = nodes1
            res = res1
        elif isinstance(part, list):
            nodes1, res1 = [], []
            for i in range(len(nodes)):
                cid = nodes[i]
                xy = res[i]
                if cid in part:
                    nodes1.append(cid)
                    res1.append(xy)
            nodes = nodes1
            res = res1
        else:
            pass
        dists = []
        mask1 = set()
        for cp in mask:
            cid, cid1 = cp
            if cid != cid1 and cid in nodes and cid1 in nodes:
                mask1.add((min(cid, cid1), max(cid, cid1)))
        cid_res = {}
        for i in range(len(nodes)):
            cid_res[nodes[i]] = res[i]
        for cp in mask1:
            cid1, cid2 = cp
            if cid1 in nodes and cid2 in nodes:
                c1 = cid_res[cid1]
                c2 = cid_res[cid2]
                dist = np.sqrt(np.sum(np.square(np.array(c1) - np.array(c2))))
                dists.append(dist)
        plt.figure()
        plt.hist(dists, bins=30)
        plt.savefig(title, dpi=150)
        if show:
            plt.show()
        dists.sort()
        print(len(dists))
        win = dists[min(int(len(dists) * 0.99),len(dists)-1)]

        # 画两条曲线，一条是随着地理距离增大，被包含的拓扑边的比例，另一条是随着地理距离增大，包含边中非拓扑边的比例
        mat1 = cdist(np.array(res), np.array(res))
        np.fill_diagonal(mat1, 2 * win)
        keys = set()
        dlist = []
        topolist = []
        idx = np.where(mat1 <= win)
        print(idx)
        step = int(len(idx[0]) / 20)
        for i in range(len(idx[0])):  # 改造成数组？跑了一下不是这一步慢，是画曲线的部分慢，已经做了优化
            if i % step == 0:
                print(i / step)
            id1 = idx[0][i]
            id2 = idx[1][i]
            if id1 < id2:
                cid1 = nodes[id1]
                cid2 = nodes[id2]
                dist = mat1[id1][id2]
                keys.add(dist)
                dlist.append(dist)
                topolist.append((min(cid1, cid2), max(cid1, cid2)) in mask1)

        keys = list(keys)
        keys.sort()
        dlist = np.array(dlist)
        topolist = np.array(topolist)
        tot = np.count_nonzero(topolist == True)

        print(tot, len(mask1), len(keys))
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        x = []
        # step=max(int(len(keys)/50),1)
        step = keys[-1] / 50
        temp = 0
        for i in range(len(keys)):
            d = keys[i]
            # if i%step==0:
            if d / step > temp:
                temp += 1
                print(d, d / step)
                if 0 < d <= win:
                    x.append(d)
                    idx = np.where(dlist <= d)
                    data = topolist[idx]
                    y1.append(np.count_nonzero(data == True) / tot)
                    y2.append(np.count_nonzero(data == True) / len(data))
                    y3.append(len(data))
                    y4.append(np.count_nonzero(data == True))
        data=[x,y1,y2,y3,y4]
        np.save(temptitle,np.array(data))
    else:
        print('reading topo data')
    data=np.load(temptitle)
    x,y1,y2,y3,y4=data[0],data[1],data[2],data[3],data[4]
    plt.figure(figsize=(7, 6))
    # plt.figure(figsize=(5, 10))
    # ax = plt.subplot(211)
    ax = plt.subplot()
    ax.set_ylim(-0.3,1.1)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.plot(x, y1, '-', color=(0.22,0.29,0.51), linewidth=3,label='Cumulative proportion of edges covered')
    ax.plot(x, y2, '-', color=(0.98,0.71,0.17), linewidth=3, label='Cumulative proportion of correct edges')
    plt.xlabel('Distances by the embedding')
    plt.legend(loc='lower right')

    # ax = plt.subplot(212)
    # ax.plot(x, y3, '-', color='r', label='cumulative number of node pairs')
    # ax.plot(x, y4, '-', color='b', label='cumulative number of correct links')
    # plt.legend()

    plt.savefig(title1, dpi=300)
    if show:
        plt.show()

    return find_intersections(x, y1, y2)


def pearson_plot_part(res, nodes, part, mask, mat, title, opt=1, maxd=10, show=True):
    cid_res = {}
    for i in range(len(res)):
        cid_res[part[i]] = res[i]
    x, y = [], []
    tag = dict(zip(nodes, list(range(len(nodes)))))
    if opt == 1:

        # 这种情况下，只对mat中每行前6且在inf内的输入，找出这些节点对在嵌入里的距离
        for cp in mask:
            cid, cid1 = cp
            if cid in part and cid1 in part:
                d0 = mat[tag[cid]][tag[cid1]]
                if d0 < maxd:
                    x.append(d0)
                    y.append(np.linalg.norm(np.array(cid_res[cid]) - np.array(cid_res[cid1])))

    print(len(x))
    r = pearsonr(x, y)
    print(r[0], r[1])
    plt.figure(figsize=(6, 6))
    # plt.xlim((0,6))
    # plt.ylim((0,6))
    plt.plot(x, y, 'o', markersize=5, color=(0.22,0.29,0.51, max(0.01,min(1,1000/len(x)))))
    plt.xlabel('Distances by network weights')
    plt.ylabel('Distances by the embedding')
    plt.savefig(title, dpi=300)
    if show:
        plt.show()
    return r[0]


def local_boundary(mat, cid, k, inf, nc, tc):
    row = mat[cid]
    temp = dict(zip(range(len(mat)), row))
    temp = sorted(temp.items(), key=lambda x: x[1])
    cands = []
    flag = 0
    for i in range(k):
        if temp[i][1] < inf:
            cands.append(temp[i][0])
        else:
            flag += 1
    if len(cands) > 1:
        cands = np.array(cands)  # mds降维并不需要矩阵行列数大于维数，所以所有邻域都可以做
        mattemp = mat[cands.reshape(-1, 1), cands]
        # print(mattemp)
        embedding = MDS(n_components=nc, dissimilarity='precomputed')
        X_transformed = embedding.fit_transform(mattemp)
        cen = np.mean(X_transformed, axis=0)
        x0 = X_transformed[0]
        tao = 0
        for j in range(len(cands)):
            xj = X_transformed[j]
            num1, num2 = 0, 0
            for k in range(len(cands)):
                x = X_transformed[k]
                if np.dot((x0 - xj), (x - x0).T) > 0:
                    num1 += 1
                else:
                    num2 += 1
            if num2 == 0:
                tao += 1
            else:
                if num1 / num2 <= tc:
                    tao += 1
        return np.linalg.norm(cen - X_transformed[0]), tao / len(cands)
    else:
        return inf, 1


def boundary(mat, k, nb, tc, inf=10, nc=2):
    # 使用knn邻域，目标输出边缘点数量nb
    boundary1, boundary2 = set(), set()
    temp1, temp2 = {}, {}
    for cid in range(len(mat)):
        dist, tao = local_boundary(mat, cid, k, inf, nc, tc)
        temp1[cid] = dist
        temp2[cid] = tao
    # print(temp)
    temp1 = sorted(temp1.items(), key=lambda x: x[1], reverse=True)
    temp2 = sorted(temp2.items(), key=lambda x: x[1], reverse=True)
    for i in range(nb):
        boundary1.add(temp1[i][0])
        boundary2.add(temp2[i][0])
    return boundary1 & boundary2
    # return boundary2


def mat_to_g(mat, mask, nodes, inf):
    # 修改为只将top6的输入，即mask的边
    cid_tag = dict(zip(nodes, range(len(nodes))))
    mask1 = []
    for cp in mask:
        c1, c2 = cp
        mask1.append((cid_tag[c1], cid_tag[c2]))

    g = nx.Graph()
    g.add_nodes_from(range(len(mat)))
    for i in range(len(mat) - 1):
        for j in range(i + 1, len(mat)):
            if mat[i][j] < inf and ((i, j) in mask1 or (j, i) in mask1):
                g.add_weighted_edges_from([(i, j, mat[i][j])])

    return g, mask1


def calculate_path(g):
    paths = dict(nx.all_pairs_dijkstra_path(g))
    lengths = dict(nx.all_pairs_dijkstra_path_length(g))
    return paths, lengths


def output_path(paths, lengths, title1, title2):
    with open(title1, 'w', newline='') as f:
        wt = csv.writer(f)
        for i in paths:
            wt.writerow([i, paths[i]])
    with open(title2, 'w', newline='') as f:
        wt = csv.writer(f)
        for i in lengths:
            wt.writerow([i, lengths[i]])
    return


def read_path(file1, file2):
    paths, lengths = {}, {}
    with open(file1, 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            paths[int(row[0])] = eval(row[1])
    with open(file2, 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            lengths[int(row[0])] = eval(row[1])
    return paths, lengths


def get_cons_geod(paths, lengths, bound, inf, thre, w0, title, mask1):
    # 获取consistent的测地线，满足两个条件之一，不与边界相交，或两端到边界距离大于测地线长度
    bound1 = set(bound)
    cid_b_dist = {}
    for cid in paths:
        temp = []
        for b in bound:
            if b in lengths[cid]:
                temp.append(lengths[cid][b])
        temp.sort()
        if len(temp) == 0:
            cid_b_dist[cid] = inf
        else:
            cid_b_dist[cid] = temp[0]
    print('dist to boundary ready')

    cps = np.zeros((len(paths), len(paths)))

    for i in range(len(paths)):
        if i % 100 == 0:
            print(i)
        for j in range(len(paths)):
            if len(set(paths[i].get(j, bound1)) & bound1) == 0:
                cps[i][j] = 1
            elif (i, j) in mask1 or (j, i) in mask1:
                cps[i][j] = 1
            else:
                if lengths[i].get(j, inf * 3) <= cid_b_dist[i] + cid_b_dist[j]:
                    cps[i][j] = 1
                elif lengths[i].get(j, inf * 3) <= thre:
                    cps[i][j] = w0
    print('ready')
    np.save(title, cps)
    print('saved')
    return cps


def weight_mds_single(niter, kiter, mat, wmat, nc, init):
    res = []
    sres = []
    for i in range(niter):
        X_transformed, s = cons_smacof(np.array(mat), np.array(wmat), \
                                       n_components=nc, init=init, eps=10 ** (-50), max_iter=1, n_init=1)
        init = X_transformed
        sres.append(s)
        if i >= niter - kiter:
            res.append(X_transformed)

    print(s)
    print('emb ready')
    return res, sres


def RRE_iter(niter, kiter, mat, wmat, nc, init):
    # 迭代n次smacof，将最后k次做最小二乘，得到新的初始化
    res, sres = weight_mds_single(niter, kiter, mat, wmat, nc, init)
    matrices = []
    for i in range(len(res) - 1):
        matrices.append(res[i + 1] - res[i])

    def frobenius_norm(c, matrices):
        weighted_sum = sum(c[i] * matrices[i] for i in range(len(c)))
        return np.linalg.norm(weighted_sum, 'fro')

    def constraint(c):
        return np.sum(c) - 1

    c0 = np.ones(len(matrices)) / len(matrices)

    # 定义约束条件
    con = {'type': 'eq', 'fun': constraint}

    # 使用scipy的minimize函数来求解
    result = minimize(frobenius_norm, c0, args=(matrices,), constraints=con)

    # 输出优化结果
    print("最优系数:", result.x)
    print("最小化的 Frobenius 范数:", result.fun)

    x = sum(result.x[i] * res[i + 1] for i in range(len(matrices)))
    return sres, x


def smacof_by_RRE(niter, kiter, N, lengths, weight, nc, inf, path, init=False):
    mat = np.ones((len(lengths), len(lengths)))
    mat = mat * inf
    for i in lengths:
        for j in lengths[i]:
            mat[i][j] = lengths[i][j]
    print('mat ready')
    if isinstance(init, np.ndarray):
        initemb = init
    else:
        if init == True:
            embedding = MDS(n_components=nc, dissimilarity='precomputed')
            initemb = embedding.fit_transform(mat)

        else:
            initemb = None
    print('init ready')
    wmat = np.array(weight)
    slist = []
    for _ in range(N):
        sres, x = RRE_iter(niter, kiter, mat, wmat, nc, initemb)
        slist += sres
        np.save(r'{path}/x_N{_}.npy'.format(path=path, _=_), x)
        initemb = x
        print('-----------', _, slist[-1])
    np.save(r'{path}/slist.npy'.format(path=path), np.array(slist))
    return x, slist


def plot_s_curve(file, title, show=True):
    # 可视化smacof+RRE过程中stress的变化曲线
    slist = np.load(file).tolist()
    plt.figure(figsize=(10, 3))
    plt.plot(range(len(slist)), slist, 'o-', markersize=5, linewidth=3)
    plt.savefig(title, dpi=150)
    if show:
        plt.show()
    return



def tcie_para(R, perp, perpsne, bnc, nc, inf, k, nb, tc, thre, w0, niter, kiter, N, perptest=6, \
              show=False, matfile=None, dir0=r'E:\流形研究\20250305tcie_mask',emb=True):
    # R，计算距离矩阵时的参数
    # perp决定mask大小
    # perpsne，用于画边界图的tsne嵌入，使用的perp参数
    # bnc，边缘点探测所用维数
    # nc，维数
    # inf，距离矩阵中无连接部分的赋值
    # k，识别边界时的局部大小
    # nb，边界点数量
    # tc，基于方向的边界识别中的比例阈值
    # thre，计算权重矩阵时，路径小于该阈值的被赋予较低权重
    # w0，权重矩阵中的较低权重
    # niter，每轮rre进行的总smacof轮数
    # kiter，rre用于插值的轮数
    # N，rre轮数
    # perptest，用于拓扑和距离嵌入分析的perp
    # 调参主要针对niter之前的部分，即边界识别的部分，因为后面基本都是让smacof收敛，只要收敛就行，影响不大
    dirs = r'{dir0}\R{R}_bnc{bnc}_inf{inf}_k{k}_nb{nb}_tc{tc}'. \
        format(dir0=dir0, R=R, bnc=bnc, inf=inf, k=k, nb=nb, tc=tc)
    embdir = r'{dir}\nc{nc}_perp{perp}_thre{thre}_w0{w0}_niter{niter}_kiter{kiter}_N{N}'. \
        format(dir=dirs, nc=nc, perp=perp, thre=thre, w0=w0, niter=niter, kiter=kiter, N=N)
    embtitle = r'{dir}\res.csv'.format(dir=embdir)
    if matfile == None:
        matfile = r'E:\流形研究\20250101基于加权双曲的距离\triphi_R{R}.csv'.format(R=R)
    nodes, mat, mask = read_mat(matfile, maxd=inf, sigma=perp, dmax=None)
    #nodes, mat, mask = read_mat(matfile, maxd=10+R, sigma=perp, dmax=None)
    # #20250617这是旧版，似乎不太合理，所以改为maxd=inf
    mat = np.array(mat)
    print('mat read')
    if emb:

        if not os.path.exists(dirs):
            os.makedirs(dirs)
        print('dirs created')
        boundname = r'{dir}\bound.npy'.format(dir=dirs)
        if not os.path.exists(boundname):
            print('making bounds')
            bound = list(boundary(mat, k, nb, tc, inf, bnc))
            bound = np.array(bound)
            print('bound ready')
            # print(bound)

            np.save(boundname, bound)
            print('bound saved')



        else:
            bound = np.load(boundname)
            print('bound read')
        print('bound ready')

        pathname = r'{dir}\perp{perp}_paths.csv'.format(dir=dirs, perp=perp)
        lengthname = r'{dir}\perp{perp}_lengths.csv'.format(dir=dirs, perp=perp)
        if not os.path.exists(pathname):
            print('making path and length')
            g, mask1 = mat_to_g(mat, mask, nodes, inf)
            paths, lengths = calculate_path(g)
            print('path and length ready')
            output_path(paths, lengths, pathname, lengthname)
        else:
            paths, lengths = read_path(pathname, lengthname)

            cid_tag = dict(zip(nodes, range(len(nodes))))
            mask1 = []
            for cp in mask:
                c1, c2 = cp
                mask1.append((cid_tag[c1], cid_tag[c2]))
            print('bound, paths, lengths read')

        wname = r'{dir}\wmat_perp{perp}_thre{thre}_w0{w0}.npy'. \
            format(dir=dirs, perp=perp, thre=thre, w0=w0)
        if not os.path.exists(wname):
            print('making wmat')
            wmat = get_cons_geod(paths, lengths, bound, inf, thre, w0, wname, mask1)

        else:
            print('reading wmat')
            wmat = np.load(wname)

        print('cons geod ready')

        if not os.path.exists(embdir):
            os.makedirs(embdir)
            print('embedding')
            x, slist = smacof_by_RRE(niter, kiter, N, lengths, wmat, nc, inf, embdir, init=True)
        else:
            print('reading embedding')
            slist = np.load(r'{dir}\slist.npy'.format(dir=embdir))
            x = np.load(r'{dir}\x_N{N}.npy'.format(dir=embdir, N=N - 1))
        print('emb ready')
        plot_s_curve(r'{dir}\slist.npy'.format(dir=embdir), r'{dir}\slist.png'.format(dir=embdir), show)


        output_res(x, nodes, embtitle)

    res, nodes = read_manifold(embtitle, nc=nc, opt=1)

    if perp != perptest:
        nodes, mat, mask = read_mat(matfile, maxd=10 + R, sigma=perptest, dmax=None)

    curvetitle = r'{dir}\curve_perptest{perptest}.png'.format(dir=embdir, perptest=perptest)
    xinter,yinter=topo_fit(nodes, mask, res, r'{dir}\hist_perptest{perptest}.png'.format(dir=embdir, perptest=perptest), \
             curvetitle, part=None, win=30, show=show,temptitle=r'{dir}\topo.npy'.format(dir=embdir))


    pearstitle = r'{dir}\pears_perptest{perptest}.png'.format(dir=embdir, perptest=perptest)
    r = pearson_plot_part(res, nodes, nodes, mask, mat, pearstitle, opt=1, maxd=R + 20, show=show)

    print(r, R, perp, perpsne, nc, inf, k, nb, tc, thre, w0, niter, kiter, N)
    title = r'{dir}\para.csv'.format(dir=embdir)
    with open(title, 'w', newline='') as f:
        wt = csv.writer(f)
        row1 = [r, R, perp, perpsne, nc, inf, k, nb, tc, thre, w0, niter, kiter, N]
        row0 = ['pears'] + 'R,perp,perpsne,nc,inf,k,nb,tc,thre,w0,niter,kiter,N'.split(',')
        arr = np.array([row0, row1]).T.tolist()
        for i in arr:
            wt.writerow([i])
    return r,yinter


def auto_mat(file,title, R='auto', errthre=0.1, inf=100, ep=0.1, top=6, \
             figtitle=r'E:\流形研究\20250317代码整理\temp.png',minlam=5,tiv=0.01,\
             titletiv=r'E:\流形研究\20250317代码整理\tivcurve.npy'):
    #file是od矩阵文件名，第一行第一列是节点id组成的list，之后每行第一列是od矩阵对应行的list，第二列是位置的tuple
    #title是输出的距离矩阵文件名
    #errthre是迭代回归时的误差阈值，当新一轮迭代带来的参数改变比例小于该阈值时停止
    #inf是无连接的赋值
    #ep用于控制过强的交互距离也非负，在R=auto时该参数不会发挥作用
    #top是邻域大小
    #minlam给交互做剪枝，初始流量小于该参数的边被剪枝

    cp_lam = {}
    cid_k = {}  # 根据cp算
    cid_s = {}  # 不是人口，是边权加和
    with open(file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        nodes=eval(header[0])
        count=0
        for row in rd:
            cid0=nodes[count]
            temp=eval(row[0])
            for i in range(len(temp)):
                if i>count and temp[i]>=minlam:
                    cp_lam[(cid0,nodes[i])]=temp[i]
            count += 1

    print('lam ready')
    for cid in nodes:
        k = 0
        s = 0
        for cid1 in nodes:
            if (cid, cid1) in cp_lam or (cid1, cid) in cp_lam:
                k += 1
                s += cp_lam.get((cid, cid1), 0) + cp_lam.get((cid1, cid), 0)
        if k != 0:
            # if s < k:
            #     print('err, strength less than degree')
            # else:
            #     cid_k[cid] = k
            #     cid_s[cid] = s
            cid_k[cid] = k
            cid_s[cid] = s
    print('s and k ready',len(cp_lam),len(cid_s),len(cid_k))

    sigmaind,kappaind = cover_to_rad(cp_lam,cid_s,cid_k,\
                            title, R='auto', inf=inf, ep=ep, \
                            sigmaind=1,kappaind=1, top=top, option='initreg', \
                            figtitle=r'E:\流形研究\20250317代码整理\temp.png')
    err = 1
    while err > errthre:
        sigmaind1,kappaind1 = cover_to_rad(cp_lam,cid_s,cid_k,\
                            title, R='auto', inf=inf, ep=ep, \
                            sigmaind=sigmaind,kappaind=kappaind, top=top, option='reg', \
                                            figtitle=r'E:\流形研究\20250317代码整理\temp.png')
        err = max(abs((sigmaind1 - sigmaind)/sigmaind),abs((kappaind1-kappaind)/kappaind))
        sigmaind,kappaind = sigmaind1,kappaind1

    #alphaind=0.830655455735911
    print(sigmaind,kappaind)
    R1=cover_to_rad(cp_lam,cid_s,cid_k,\
                            title, R=R, inf=inf, ep=ep, \
                            sigmaind=sigmaind,kappaind=kappaind, top=top, option='', \
                 figtitle=r'E:\流形研究\20250317代码整理\temp.png',tiv=tiv,\
                    titletiv=titletiv)
    return R1
def weimat_to_dich(file,inf,title,k=2,exset=[],postitle=r'E:\流形研究\20241214基于分区的模型\pos_cid1.npy'):
    #将交互矩阵的每一行转化为排序，记录minf（在其他函数中也对应inf）交互以上，各排序节点对应的交互量，然后存储为cen_rank
    #file是交互矩阵的csv文件，第一行是节点编号的list，其后每行是矩阵的行
    #minf用于判断一条边的交互量是否不在考虑范围内
    #k对应探索返回二分中的rgk，表示提取A点之后的第几名，k=2对应A的下一名
    #exset是用于排除一部分需要剪枝的节点

    cid_cen={}#用于记录以每个节点为中心，其他节点按距离排列的列表，以及距离字典
    with open(file,'r') as f:
        rd=csv.reader(f)
        nodes=next(rd)
        nodes=eval(nodes[0])
        set0=set(nodes)-set(exset)
        count=0
        for row in rd:
            cid=nodes[count]
            count += 1
            if cid in set0:
                rank={}
                temp=eval(row[0])
                if len(temp)!=len(nodes):
                    print('err, num of nodes != mat width')
                for i in range(len(temp)):
                    cid1=nodes[i]

                    if cid1!=cid and temp[i]<inf and cid1 in set0:#第二个条件确保了cid1和cid是相连的，因此二分中不与O相连的部分确实是A的邻居不与O相连
                        rank[cid1]=temp[i]
                if rank!={}:
                    ranks=sorted(rank.items(),key=lambda x:x[1])
                    cid_cen[cid]=(rank,ranks)
    dist_dist1={}#二分的分析结果，OA和OB
    pos_cid1=[]#记录B在A邻域中的位置，按与O的距离排序，理想情况下应该在中间
    dist_pair = {}#用以记录O和B的对，与dist_dist1一一对应，以便分析条件2的实证，即inbox的B有更小的地理距离
    for cid in cid_cen:#cid对应O
        rank,ranks=cid_cen[cid]
        for j in range(len(ranks)):
            cid1,dist=ranks[j]#cid1对应A
            if cid1 in cid_cen:
                rank1,ranks1=cid_cen[cid1]
                temp1={}
                for cid2 in rank1:#cid2对应B
                    if rank.get(cid2,inf)>rank[cid1] and cid2!=cid and cid2!=cid1:#当OB不相连时，赋值为maxd，也即inf
                        temp1[cid2]=rank.get(cid2,inf)
                stemp1=sorted(temp1.items(),key=lambda x:x[1])
                pos_cid1.append(len(stemp1)/len(rank1))
                if len(stemp1)>=k-1:
                    dist1 = stemp1[k - 2][1]
                    if dist not in dist_dist1:
                        dist_dist1[dist] = []
                        dist_pair[dist] = []
                    dist_dist1[dist].append(dist1)
                    dist_pair[dist].append((cid,cid1, stemp1[k - 2][0]))
                '''
                else:#是否把cid1邻域太小的情况考虑进去有待取舍，之前的分析结果差异不大。保留邻域太小的影响，相当于说扩张在这个小邻域处达到极限。不保留是为了让k不同时可比。
                    dist1=maxd
                '''
    np.save(postitle,pos_cid1)
    with open(title, 'w', newline='') as f:
        wt = csv.writer(f)
        for k in dist_dist1:
            if len(dist_dist1[k]) != len(dist_pair[k]):
                print('err len')
            wt.writerow([k, dist_dist1[k], dist_pair[k]])

    return
def plot_dich(file,title,inf=100):
    x,y=[],[]
    alphatag=500
    '''
    这里需要改二分图中，纵轴过长导致二分两部分相距很远，而且对角线过于贴近x轴的问题
    或许可以考虑做成截断的y轴，即标注横线在inf，从这里到对角线那部分的最高点之间是截掉的y轴
    '''
    y1,y2=[],[]#y1是inbox的y，表示截断下的部分，y2是截断上的部分
    with open(file,'r') as f:
        rd=csv.reader(f)
        for row in rd:
            x0=float(row[0])
            temp=eval(row[1])
            for i in temp:
                x.append(x0)
                y.append(i)
                flag=(inf-i)/(inf-x0)
                if flag<0.5:
                    y2.append(i)
                else:
                    y1.append(i)
    print(len(x),len(y))
    y1.sort()
    y2.sort()
    xlist=x.copy()
    xlist.sort()
    ycut0=0
    ycut1=int(y1[-1]/5+1)*5
    ycut2=int(y2[0]-1)
    ycut3=int(y2[-1]+1)
    xcut0=0
    xcut1=int(xlist[-1]/5+1)*5
    #plt.figure()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': \
                                                                                       [ycut3-ycut2,ycut1-ycut0]})

    # 上面的轴显示高的部分
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.set_xticks([])
    ax1.set_xlim(xcut0, xcut1)
    ax1.scatter(x, y, color=(0.22,0.29,0.51,max(0.01,min(1,alphatag/len(x)))))
    ax1.set_ylim(ycut2, ycut3)  # 上部分 Y 范围

    # 下面的轴显示低的部分
    ax2.scatter(x, y, color=(0.22,0.29,0.51,max(0.01,min(1,alphatag/len(x)))))
    ax2.set_ylim(ycut0, ycut1)  # 下部分 Y 范围
    ax2.set_xlim(xcut0,xcut1)
    ax2.set_xticks(range(xcut0,xcut1,5))

    # 去除上下x轴之间的刻度标签
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labelbottom=False)

    # 添加锯齿标记，表示断裂
    d = .5  # 锯齿大小
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)  # 上轴底部
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)  # 下轴顶部

    # 添加标签
    ax1.set_ylabel('$OB$')
    ax2.set_xlabel("$OA$")
    # plt.plot(x,y,'o',markersize=3,color=(1,0,0,min(1,len(x)/3000)))
    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    plt.savefig(title,dpi=300)
    plt.show()
    return

def dich_hist_geo(file,histtitle,title,c_xy,inf,dichthre,x0max=None,\
                  dist1title='20250606viz/ita_dich/dist1.npy',dist2title='20250606viz/ita_dich/dist2.npy'):
    #file是二分分析的结果，第一列为OA距离，第二列为该OA距离下OB距离的list，第三列是O和B的节点对的list
    #title是输出文件名
    #c_xy是节点空间位置的字典
    #dichthre是二元组，b1和b2是判断二分两个峰边界的阈值；或设置为auto，自动根据kmeans计算
    #x0max是最大OA距离，可设置为与inf相等
    # x0max = 0
    # with open(file, 'r') as f:
    #     rd = csv.reader(f)
    #     for row in rd:
    #         data0 = float(row[0])
    #         if inf > data0 > x0max:
    #             x0max = data0
    #
    #         temp = eval(row[1])
    #         for i in range(len(temp)):
    #             x1 = temp[i]
    #             if inf > x1 > x0max:
    #                 x0max = x1
    # print('new inf', x0max)
    # inf = x0max

    if x0max==None:
        x0max=inf
    if not os.path.exists(dist1title):
        print('making geo dich data')
        data=[]
        dist1, dist2 = [], []
        with open(file, 'r') as f:
            rd = csv.reader(f)
            for row in rd:
                data0 = float(row[0])
                if data0 <= x0max:
                    temp = eval(row[1])
                    pairs = eval(row[2])
                    for i in range(len(temp)):
                        x1 = temp[i]
                        x = (inf - x1) / (inf - data0)
                        data.append(x)
        if dichthre!='auto':
            b1,b2=dichthre
        else:
            kmeans = KMeans(n_clusters=2, random_state=0)
            data = np.array(data)
            kmeans.fit(data.reshape(-1, 1))

            # 输出两个簇的中心
            cluster_centers = kmeans.cluster_centers_
            print("簇的中心:", cluster_centers)

            # 获取分界点，可以假设是两个簇中心的中点
            boundary_point = np.mean(cluster_centers)  # 注意，x<这个数时才是盒子外
            print("分界点:", boundary_point)
            b1, b2 =boundary_point,boundary_point
        with open(file, 'r') as f:
            rd = csv.reader(f)
            for row in rd:
                data0 = float(row[0])
                if data0 <= x0max:
                    temp = eval(row[1])
                    pairs = eval(row[2])
                    for i in range(len(temp)):
                        x1 = temp[i]
                        x = (inf - x1) / (inf - data0)
                        cp = pairs[i]
                        cid0,cid2, cid1 = cp
                        if cid0 in c_xy and cid1 in c_xy:
                            x0, y0 = c_xy[cid0]
                            x1, y1 = c_xy[cid1]
                            d = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
                            if x < b1:
                                dist2.append(d)
                            elif x > b2:
                                dist1.append(d)
        print('dist ready',len(dist1),len(dist2))
        plt.hist(data)
        plt.savefig(histtitle, dpi=150)
        plt.show()
        ax = plt.subplot()
        np.save(dist1title,np.array(dist1))
        np.save(dist2title, np.array(dist2))
    else:
        print('reading geo dich data')
    # ax.set_yscale('log')
    '''
    前面主要是计算两组距离，分别对应于二分的两个类，这些计算也可以之后做个中间结果输出，加快可视化调整的运行。从这里开始是可视化。
    '''
    dist1=np.load(dist1title)
    dist2=np.load(dist2title)
    plt.figure()
    plt.xlabel('$OB_g$')
    plt.ylabel('Probability density')
    bins=20
    plt.hist(dist1, color=(0.98,0.71,0.17), alpha=0.5, bins=bins, density=True, label='$OB_g^1$')
    plt.hist(dist2, color=(0.22,0.29,0.51), alpha=0.5, bins=bins, density=True, label='$OB_g^2$')
    plt.legend()
    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.2,top=0.9,right=0.9)
    plt.savefig(title, dpi=300)
    plt.show()
    t_stat, p_value = ttest_ind(dist1, dist2, alternative='less')

    print(f"T statistic: {t_stat}")
    print(f"P value: {p_value}")
    return

def read_xy(file):
    c_xy={}
    with open(file,'r') as f:
        rd=csv.reader(f)
        header=next(rd)
        nodes=eval(header[0])
        count=0
        for row in rd:
            c_xy[nodes[count]]=eval(row[1])
            count+=1
    return nodes,c_xy
def auto_dich(matfile,inf,dir0,c_xy,dichthre=(0.5,0.5),k=2):
    print('start')
    minf=inf
    dichtitle=r'{dir0}/dich.csv'.format(dir0=dir0)
    postitle=r'{dir0}/pos.npy'.format(dir0=dir0)
    if not os.path.exists(dichtitle):
        print('making dich')
        weimat_to_dich(matfile, minf, dichtitle, k, exset=[], postitle=postitle)
        print('dich analysis ready')
    else:
        print('dich already made')
    dichplt=r'{dir0}/dich.png'.format(dir0=dir0)
    plot_dich(dichtitle, dichplt,inf)

    histplt=r'{dir0}/hist.png'.format(dir0=dir0)
    geoplt=r'{dir0}/geo.png'.format(dir0=dir0)
    dist1title=r'{dir0}/dist1.npy'.format(dir0=dir0)
    dist2title = r'{dir0}/dist2.npy'.format(dir0=dir0)
    dich_hist_geo(dichtitle,histplt,geoplt,c_xy,inf,dichthre,x0max=None,dist1title=dist1title,dist2title=dist2title)
    return

if __name__ == '__main__':
    xmin, xmax, ymin, ymax = 113.67561783007596, 114.60880792079337, \
        22.28129833936937, 22.852485545898546  # 深圳最大最小经纬度
    r = 6371 * 1000
    ymid = (ymin + ymax) / 2
    r1 = r * cos(ymid / 180 * pi)
    scale = 150
    xgap = scale / r1 / pi * 180
    ygap = scale / r / pi * 180
    xnum = int((xmax - xmin) / xgap) + 1
    xnumls = int(xnum / 6) + 1
    ynumls = int(424 / 6) + 1

    minpop = 10

