# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:28:21 2020

@author: lq
"""
import numpy as np
import networkx as nx
from numpy.random.mtrand import RandomState as rdm
import random
random.seed(10)

def fruchterman_reingold_init(
   G, A, k=None, pos=None, fixed=None,k1=2, iterations=200, threshold=1e-4, dim=2, seed=None
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    nnodes, _ = A.shape
    # 位置初始化
    if pos is None:
        seed = rdm(10)
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        pos = pos.astype(A.dtype)
    # 初始化k
    if k is None:
        k = np.sqrt(k1 / nnodes)
    # 初始化t
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    dt = t / float(iterations + 1)
    # 初始化 距离表  
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # print('pos', pos)
    for iteration in range(iterations):
        # 计算每个点之间的距离
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  
        distance = np.linalg.norm(delta, axis=-1)  # 求范数 默认二范数 可以求出不同点之间的距离 
       
        # 制大小 最小为0.01    out：可以指定输出矩阵的对象，shape与a相同
        np.clip(distance, 0.01, None, out=distance) # 限
        
        # 计算x,y方向上的累计位移
        a = (k * k / distance ** 2  - A * distance / k)
        displacement = np.einsum(
            "ijk,ij->ik", delta, (k * k / distance ** 2  - A * distance / k)        # 在不同维度上产生的合力 = 1 / distance (k * k / distance ** 2  - A * distance / k), "ijk,ij->ik" 点乘再累加
        )
        # 更新
        length = np.linalg.norm(displacement, axis=-1)            
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # 模拟降温
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    pos = dict(zip(G, pos))
    return pos

# 基于多约束的力引导布局
def fruchterman_reingold(
   G, A, pre_neighbors, time_evolution,k2=1, k=None, pos=None, fixed=None, iterations=200, threshold=1e-4, dim=2, seed=None, 
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    nnodes, _ = A.shape
    # 位置初始化
    if pos is None:
        seed = rdm()
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        pos = pos.astype(A.dtype)
    # 初始化k
    if k is None:
        k = np.sqrt(k2 / nnodes)
    # 初始化t
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    dt = t / float(iterations + 1)
    # 初始化 距离表  
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    
    ## 基于多约束的力引导算法
    # 确定节点的布局能量等级
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  
    distance = np.linalg.norm(delta, axis=-1) 
    np.clip(distance, 0.01, None, out=distance)
    displacement = np.einsum(
        "ijk,ij->ik", delta, (k * k / distance ** 2  - A * distance / k)        # 在不同维度上产生的合力 = 1 / distance (k * k / distance  - A * distance**2 / k), "ijk,ij->ik" 点乘再累加
    )
    length = np.linalg.norm(displacement, axis=-1) 
    avg = np.average(length)
    id_node = {}
    node_id = {}
    for idx, n in enumerate(list(G.nodes)):
        id_node[idx] = n
        node_id[n] = idx
    a = 0.5
    b = 0.2
    lvl = [(abs(i-avg) / avg ) for i in length]
    l0 = []  
    l1 = []
    l2 = []
    for idx,i in enumerate(lvl):
        if i>=0.5:
              l2.append(id_node[idx])
        elif i<0.2:
            l0.append(id_node[idx])
        else:
            l1.append(id_node[idx]) 
    
    fixed = l0
    # 更新时间参数 
    now_neighbors = {n:list(G.neighbors(n)) for n in  G.nodes}
    bt = 1
    for n in now_neighbors:
        if n not in pre_neighbors:
            time_evolution[n] = 1
        elif len(now_neighbors[n]) > 0:
            n_pre_neighbors = pre_neighbors[n]
            n_now_neighbors = now_neighbors[n]
            s_old = sum([time_evolution[i] for i in n_pre_neighbors])
            s_new = sum([time_evolution[i] for i in (set(n_pre_neighbors) & set(n_now_neighbors))])
            time_evolution[n] = bt * time_evolution[n] * (s_new / s_old) + 1 if not s_old else 1
        else:
            time_evolution[n] = bt * time_evolution[n] + 1
    to_del = set(pre_neighbors) - set(now_neighbors)
    # 时间演化参数表删除已经消失的节点
    for n in to_del:
        del time_evolution[n]   
      
    ## 迭代计算
    for iteration in range(iterations):
        # 计算每个点之间的距离
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  
        distance = np.linalg.norm(delta, axis=-1)  # 求范数 默认二范数 可以求出不同点之间的距离 
       
        # 制大小 最小为0.01    out：可以指定输出矩阵的对象，shape与a相同
        np.clip(distance, 0.01, None, out=distance) # 限
        
        # 计算x,y方向上的累计位移
        a = (k * k / distance ** 2  - A * distance / k)
        displacement = np.einsum(
            "ijk,ij->ik", delta, (k * k / distance ** 2  - A * distance / k)        # 在不同维度上产生的合力 = 1 / distance (k * k / distance  - A * distance**2 / k), "ijk,ij->ik" 点乘再累加
        )
        ## 更新
        # 计算合位移
        length = np.linalg.norm(displacement, axis=-1)       
        # 对l1中的元素施加约束
        

        length[[node_id[i] for i in l1]] = [np.exp(-time_evolution[i])*length[node_id[i]] for i in l1]
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[[node_id[n] for n in fixed ]] = 0.0
        pos += delta_pos
        ## 模拟降温
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    pre_neighbors = now_neighbors
    pos = dict(zip(G, pos))
    return pos

# 基于新增连接的节点布局微调
def preLayout_edgeAdd(G, pos, G1, a=1):
    print('preLayout_edgeAdd',len(G),len(G1))
    pos_ = pos.copy()
    dl = 1 / len(G)
    G_edges_set = set(G.edges)
    G1_edges_set = set(G1.edges(G.nodes)) 
    new_edges = G1_edges_set - G_edges_set
    G_nodes = G.nodes
    new_edges_ = []
    for edge in new_edges:
        if (edge[0]  in G_nodes) and (edge[1] in G_nodes):
            new_edges_.append(edge)
    for edge in new_edges_:
        v1, v2 = edge
        v1_pos = pos_[v1]
        v2_pos = pos_[v2]
        v1_degree = G.degree(v1)
        v2_degree = G.degree(v2)
        delta = np.sqrt(sum((v1_pos-v2_pos)**2))
        v1_mov = (a * (1 - dl/delta )) * (v1_degree / (v1_degree + v2_degree)) * (v2_pos - v1_pos ) + v1_pos
        v2_mov = (a * (1 - dl/delta )) * (v2_degree / (v1_degree + v2_degree)) * (v1_pos - v2_pos ) + v2_pos
        pos[v1] = v1_mov
        pos[v2] = v2_mov
    return pos_  

# 随机扰动
def get_random_disturb():
    return random.random()

# 新增节点预布局
def preLayout_nodeAdd(G, pos, G1,layout_size = 1,e1 = 0.05, e2 = 0.1, a = 1 ):
    pos_ = pos.copy()
    dl = 1 / len(G1)
    e3 = layout_size
    G_nodes_set = set(G.nodes)
    G1_nodes_set = set(G1.nodes)
    new_nodes = G1_nodes_set - G_nodes_set 
    print('new_nodes',new_nodes)
    # 与原网络节点有连接的边 假设一个对于与此的中间网络
    e = list(G1.edges(G_nodes_set))
    G_ = nx.Graph()
    G_.add_edges_from(e)
    s = set()
    for i in e:
        s.add(i[0])
        s.add(i[1])
    # 与原网络节点无连接的新增节点    
    d_equal_0 = G1_nodes_set - s
    # 去掉原网络的节点
    [s.remove(i) if i in s else None for i in G_nodes_set] 
    # 与原网络节点连接数为2的新增节点
    d_more_2 = []
    # 与原网络节点连接数为1的新增节点
    d_equal_1 = []
    [d_more_2.append(i) if G_.degree(i)>=2 else d_equal_1.append(i) for i in s]
    # 确定初始位置
    for n in d_more_2:
        n_neighbors = list(G_.neighbors(n))
        n_neighbors_len = len(n_neighbors)
        pos_[n] = 1 / n_neighbors_len * sum([pos[neighbor] if neighbor in G_nodes_set else None for neighbor in n_neighbors]) + e1*get_random_disturb()
         
    for n in d_equal_1:
        n_neighbor = list(G_.neighbors(n))[0]
        n_neighbor_pos = pos[n_neighbor]
        pos_[n] = n_neighbor_pos + dl + e2 * get_random_disturb()
    
    for n in d_equal_0:
        pos_[n] = np.array([e3 * get_random_disturb(), e3 * get_random_disturb()])
    
    # 迭代调整位置    
    N = 5
    for i in range(N):
        for n in new_nodes:   
            n_neighbors = list(G1.neighbors(n))
            n_neighbors_len = len(n_neighbors)
            if n_neighbors_len >= 2:
                m1 = 0.05 * get_random_disturb()
                n_pos = 1 / n_neighbors_len * sum([pos_[neighbor] for neighbor in n_neighbors]) + m1
                n_neighbor_pos = sum([pos_[neighbor] for neighbor in n_neighbors])
            elif n_neighbors_len == 1:
                n_neighbor = list(n_neighbors)[0]
                n_pos =  pos_[n_neighbor] + dl + 0.1 * get_random_disturb()
            else:             
                n_pos = np.array([layout_size * get_random_disturb(), layout_size * get_random_disturb()])
            pos_[n] = n_pos
    return pos_

# 单层布局
def multipFR(G1,pos,G2,time_evolution=None):
    # 基于新增连接的节点布局微调
    pos1 = preLayout_edgeAdd(G1, pos, G2, a=1)
    # 新增节点预布局
    for i in set(G1.nodes) - set(G2.nodes):
        del pos1[i]
    
    pos2 = preLayout_nodeAdd(G1,pos1,G2)
    # 基于多约束的力引导布局 
    pos3 = np.array(list(pos2.values()))
    pre_neighbors = {n:list(G1.neighbors(n)) for n in G1.nodes}
    if not time_evolution:
        time_evolution = {n:1 for n in G1.nodes}
    A2 = nx.to_numpy_array(G2, weight='weight')
    pos4 = fruchterman_reingold(G2, A2, pos=pos3, pre_neighbors=pre_neighbors,k2=1, time_evolution=time_evolution )
    # pos4 =  nx.spring_layout(G2,pos=pos3, pre_neighbors=pre_neighbors,k2=5, time_evolution=time_evolution )
    return pos4, time_evolution

# 动态网络布局
def multipleNetLayout(gs):
    G1 = gs[0]
    A1 = nx.to_numpy_array(G1, weight='weight')
    pos = fruchterman_reingold_init(G1,A1,k1=3)
    time_evolution = None
    poss = []
    poss.append(pos)
    for idx, g in enumerate(gs):
        if idx+1 < len(gs):
            pre = g
            print(idx+1)
            cur = gs[idx+1]
            pos,time_evolution = multipFR(pre,pos,cur,time_evolution=time_evolution)
            poss.append(pos)
    G3 = gs[len(gs)-1]
    nx.draw(G3, pos=pos,node_size =5,width=0.3) 
    return poss

#%% 完整数据测试
import pandas  as pd
import os
import json
import networkx as nx

def read_json_file(file_path):
    with open(file_path,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
        return json_data
    
data1 = read_json_file('./data/data1.json')
data2 = read_json_file('./data/data2.json')

nodes1 = [i['id'] for i in data1['nodes']]
nodes2 = [i['id'] for i in data2['nodes']]
nodes2.sort()

links1 = [(i['source'],i['target']) for i in data1['links']]
links2 = [(i['source'],i['target']) for i in data2['links']]

G1 = nx.Graph()
G2 = nx.Graph()

G1.add_nodes_from(nodes1)
G2.add_nodes_from(nodes2)

G1.add_edges_from(links1)
G2.add_edges_from(links2)

gs = [G1,G2]
poss = multipleNetLayout(gs)
