# -*- coding = utf-8 -*-
# @Time : 2022/11/28 20:39
# @Author : zackf
# @File : TS_单代号.py
# @Software : PyCharm
import random
import math
import time

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
def get_data(path):
    '''
    获取数据
    :param path: 数据所在文件的路径
    :return:
    '''
    # 存储城市间的位置
    city_positions = []
    data_file = open(path)
    # 将每条数据进行分割，形成列表
    data_content = data_file.read().split("\n")
    data_file.close()
    for info in data_content:
        # 被分割的数据中有三个数值，分别为城市编号，x坐标，y坐标，不需要城市编号故从下标1开始读，读取数值之后将其转为整数类型
        city_positions.append([float(x) for x in info.split()[1:]])
    return city_positions

def matrix(input_city_list):  # 城市距离计算
    '''
    # 根据坐标计算城市间的距离
    :param input_city_list: 城市坐标
    :return: 城市距离二维列表
    '''
    input_city_list = np.array(input_city_list)
    matrix_init = np.zeros([city_num, city_num])
    for i in range(city_num):
        for j in range(city_num):
            d = input_city_list[i, :]-input_city_list[j, :]
            matrix_init[i][j] = np.sqrt(np.dot(d, d))
    return matrix_init


# 计算每一条路径的距离
def calculate_all_paths(distances, distance_matrix):  # 计算路径距离
    '''
    所有蚂蚁走完后计算每个蚂蚁走过的路径总距离
    :param distances: 一个二维列表，每一行代表一条路径
    :param distance_matrix: 城市间距离矩阵
    :return:
    '''
    distances_list = []
    for seq in distances:
        distance = 0
        length = len(seq)
        for j in range(length-1):
            # 距离 = 上一次走完的距离+这一次走过的距离
            distance += distance_matrix[seq[j]][seq[j+1]]
        distance += distance_matrix[seq[length-1]][seq[0]]
        # 一条完整路径（一只蚂蚁走过的路径）计算完，将其插入记录距离的列表中
        distances_list.append(distance)
    return distances_list


# 期望矩阵计算
def calculate_e(distances):  # 计算路径距离
    '''
    :param distances: 一个二维列表，每一行代表一条路径
    :return: 城市间路径信息素浓度矩阵
    '''
    # 初始化矩阵
    init_matrix = [[0 for col in range(city_num)] for row in range(city_num)]
    for i in range(len(distances)):
        for j in range(len(distances)):
            # 计算初始化的每个城市间信息素浓度初始值，初始值为1/城市距离
            if distances[i][j] != 0:
                init_matrix[i][j] = 1/distances[i][j]
    return init_matrix


if __name__ == "__main__":
    file_path = "22_ACO.xlsx"
    result = []

    start_time = time.time()
    alpha = 1 # 信息素因子， 即选择新路径时残留信息素重要程度
    beta = 1 # 启发函数因子
    info_co = 0.2  # 信息残留系数
    Q = 1  # 常数，信息素增加强度系数
    iter_num = 1000  # 迭代次数
    min_distance = 0
    min_path = []
    # 读取数据坐标
    positions = get_data("ulysses22.tsp")
    # 城市数量
    city_num = len(positions)
    # 蚂蚁数量
    ant_num = int(city_num*(city_num-1)/2)
    # 城市距离矩阵
    distance_matrix = matrix(positions)
    # 初始化信息浓度
    pheromone_matrix = [[1 for col in range(city_num)] for row in range(city_num)]
    # 初始化路径矩阵
    path_matrix = [[0 for i in range(city_num)] for j in range(ant_num)]
    # 信息浓度期望矩阵
    e_matrix = calculate_e(distance_matrix)
    for i in range(iter_num):
        for ant in range(ant_num):
            # 可选城市，初始化为所有城市
            cities = list(range(city_num))
            # 选定起点城市
            visit = random.randint(0, city_num-1)
            cities.remove(visit)
            path_matrix[ant][0] = visit
            for j in range(1,city_num):
                # 轮盘赌法选择下一个城市
                trans_list = []
                trans_sum = 0

                for k in range(len(cities)):
                    # 计算信息浓度
                    trans_sum += math.pow(pheromone_matrix[visit][cities[k]], alpha)*math.pow(e_matrix[visit][cities[k]], beta)
                    trans_list.append(trans_sum)
                # 产生随机数，数值为0到信息浓度直接
                rand = random.uniform(0, trans_sum)
                # 选择第一个浓度>随机数的城市
                for t in range(len(trans_list)):
                    if rand <= trans_list[t]:
                        next_city = cities[t]
                        break
                path_matrix[ant][j] = next_city
                # 从可选城市列表中移除被选城市
                cities.remove(next_city)
                visit = next_city
        # 完成此次循环后计算距离
        distance_list = calculate_all_paths(path_matrix, distance_matrix)
        # 更新最小距离和最好路径
        if i == 0:
            min_distance = min(distance_list)
            min_path = path_matrix[distance_list.index(min_distance)].copy()
        else:
            if min(distance_list) < min_distance:
                min_distance = min(distance_list)
                min_path = path_matrix[distance_list.index(min_distance)].copy()

        # 信息素矩阵更新
        pheromone_change_matrix = [[0 for i in range(city_num)] for j in range(city_num)]
        # 计算信息素改变值
        for ant in range(ant_num):
            for j in range(city_num-1):
                pheromone_change_matrix[path_matrix[ant][j]][path_matrix[ant][j+1]] += Q / distance_matrix[path_matrix[ant][j]][path_matrix[ant][j+1]]
            pheromone_change_matrix[path_matrix[ant][city_num-1]][path_matrix[ant][0]] += Q / distance_matrix[path_matrix[ant][city_num-1]][path_matrix[ant][0]]
        # 根据公式更新信息素浓度
        for r1 in range(city_num):
            for r2 in range(city_num):
                pheromone_matrix[r1][r2] = (1-info_co)*pheromone_matrix[r1][r2]+pheromone_change_matrix[r1][r2]
        end_time = time.time()
        result.append([0, i, end_time - start_time, min_distance])
            # print(t)
    df = pd.DataFrame(result, columns=None)
    df.to_excel(file_path)