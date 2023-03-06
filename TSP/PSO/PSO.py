# -*- coding = utf-8 -*-
# @Time : 2022/11/28 10:25
# @Author : zackf
# @File : example.py
# @Software : PyCharm
import random
import math
import time
import pandas as pd
import numpy as np


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


# 解的距离计算
def calculate_distance(seq):  # 计算路径距离
    '''
        所有蚂蚁走完后计算每个蚂蚁走过的路径总距离
        :param distances: 一个二维列表，每一行代表一条路径
        :param distance_matrix: 城市间距离矩阵
        :return:
        '''
    distances1 = 0
    length = len(seq)
    for t in range(length-1):
        distances1 += distance_matrix[seq[t]][seq[t+1]]
    distances1 += distance_matrix[seq[length-1]][seq[0]]
    return distances1


# 获得新解
def crossover(path, p_seq, g_seq, w, c1, c2):
    '''
    :param path: parent1及父代1
    :param p_seq: 个体最优路径
    :param g_seq: 全局最优路径
    :param w: 惯性因子
    :param c1: 个体认知因子
    :param c2: 社会认知因子
    :return: 更新后的位置
    '''
    new_path = [0 for _ in range(len(path))]
    parent1 = path
    # 选择父代2
    r = random.random()
    if r <= w:
        # 倒序后的parent1
        parent2 = [path[i] for i in range(len(path)-1, -1, -1)]
    elif r <= w+c1:
        parent2 = p_seq
    else:
        parent2 = g_seq

    # 将parent1片段给新路径
    start_pos = random.randint(0,len(parent1)-2)
    end_pos = random.randint(start_pos, len(parent1)-1)
    new_path[start_pos:end_pos+1] = parent1[start_pos: end_pos+1].copy()

    # 将parent2的片段给新路径剩余位置
    list1 = list(range(0, start_pos))
    list2 = list(range(end_pos + 1, len(parent2)))
    list_index = list1 + list2
    j = -1
    for i in list_index:
        for j in range(0, len(parent2)):
            if parent2[j] not in new_path:
                new_path[i] = parent2[j]
                break

    return new_path


# 初始化种群
def init_group(city_num1, group_size):
    '''

    :param city_num1: 城市个数
    :return: 初始化的种群
    '''
    old_group = []
    for i in range(group_size):
        base_group = [i for i in range(city_num1)]
        random.shuffle(base_group)
        old_group.append(base_group)
    return old_group


if __name__ == "__main__":
    file_path = "22_PSO"
    result = []

    start_time = time.time()
    positions = get_data("ulysses22.tsp")
    city_num = len(positions)  # 城市数量
    iter_num = 1000  # 迭代次数
    size = city_num*(city_num-1)  # 种群数量
    w = 0.2  # 惯性概率
    c1 = 0.4  # 自我认知因子
    c2 = 0.4  # 社会认知因子
    pBest, pSeq = 0, []  # 当前最优值，当前最优解路径
    gBest, gSeq = 0, []  # 全局最优值，全局最优解路径

    # 城市距离矩阵
    distance_matrix = matrix(positions)
    path_list = init_group(city_num, size)
    distances = [calculate_distance(path) for path in path_list]
    pBest = gBest = min(distances)
    pSeq = gSeq = path_list[distances.index(gBest)]
    for i in range(iter_num):
        for j in range(len(path_list)):
            path_list[j] = crossover(path_list[j], pSeq, gSeq, w, c1, c2)
            distances[j] = calculate_distance(path_list[j])
        pBest = min(distances)
        pSeq = path_list[distances.index(pBest)]
        if pBest<gBest:
            gBest = pBest
            gSeq = pSeq
        end_time = time.time()
        result.append([0, i, end_time - start_time, gBest])
    df = pd.DataFrame(result, columns=None)
    df.to_excel(file_path)
