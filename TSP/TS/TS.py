# -*- coding = utf-8 -*-
# @Time : 2022/11/30 14:49
# @Author : zackf
# @File : TS_单代号.py
# @Software : PyCharm
import copy
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import time
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
    matrix_init = [[0 for cols in range(city_num)] for rows in range(city_num)]
    for i in range(city_num):
        for j in range(city_num):
            if i != j:
                d = input_city_list[i, :]-input_city_list[j, :]
                matrix_init[i][j] = np.sqrt(np.dot(d, d))
            else:
                matrix_init[i][j] = sys.maxsize
    return matrix_init


# 贪心算法选取初始值
def greedy():
    # 实例化一个距离矩阵，便于贪心算法使用时改变距离，不再取出已走过的城市
    dis = [[0 for col in range(city_num)] for raw in range(city_num)]
    for i in range(city_num):
        for j in range(city_num):
            dis[i][j] = distance_matrix[i][j]
    visited_cities = []
    current_city = random.randint(0, city_num-1)
    visited_cities.append(current_city)
    # 每次选取距离最近的城市，并将此次的城市设置为后续不可取（将其他城市到此城市距离设置为很大）
    for i in range(1,city_num):
        for j in range(city_num):
            dis[j][current_city] = sys.maxsize
        min_value = min(dis[current_city])
        current_city = dis[current_city].index(min_value)
        visited_cities.append(current_city)

    return visited_cities


def calculate_distance(seq):  # 计算路径距离
    '''
       :param seq: 路径
       :return: 路径长度
       '''
    distance = 0
    length = len(seq)
    for t in range(length-1):
        distance += distance_matrix[seq[t]][seq[t+1]]
    distance += distance_matrix[seq[length-1]][seq[0]]
    return distance


def initialize_parameters():
    '''
    初始化参数
    :return:
    '''
    global best_route # 最优路径
    global best_distance # 最优距离
    global current_tabu_num # 当前禁忌表长度
    global current_route # 当前路径
    global current_distance # 当前路径距离
    global tabu_list #禁忌表，存放禁忌路径
    # 初始路径
    current_route = greedy()
    for i in current_route:
        best_route.append(i)
    current_distance = calculate_distance(current_route)
    best_distance = current_distance

    tabu_list.clear()
    tabu_time.clear()
    # 当前禁忌表长度
    current_tabu_num = 0


def exchange(index1, index2, arr):
    '''
    交换位置
    :param index1: 位置1的索引
    :param index2: 位置2的索引
    :param arr: 序列（即一个可行解）
    :return: 交换后的序列
    '''
    current_list = copy.copy(arr)
    current_list[index1], current_list[index2] = current_list[index2],current_list[index1]
    return current_list


# 求邻域与候选解
def get_candidate(neighbor):
    global best_route # 最优路径
    global best_distance # 最优距离
    global current_tabu_num # 当前禁忌表中数量
    global current_distance # 当前距离
    global current_route #当前路径
    global tabu_list #禁忌表
    # 交换位置
    exchange_positions = []
    temp = 0
    # 随机选取200个邻域
    while True:
        current = random.sample(range(0, city_num), 2)

        exchange_positions.append(current)
        candidate[temp] = exchange(current[0],current[1],current_route)
        if candidate[temp] not in tabu_list:
            candidate_length[temp] = calculate_distance(candidate[temp])
            temp += 1
        if temp >= neighbor:
            break

    # 邻域最优解
    candidate_best = min(candidate_length)
    best_candidate_index = candidate_length.index(candidate_best)

    current_distance = candidate_best
    current_route = copy.copy(candidate[best_candidate_index])
    # 如果此次最优解优于之前的最优解，则更新最优解
    if current_distance < best_distance:
        best_route = copy.copy(current_route)
        best_distance = current_distance
    # 将邻域最优解加入禁忌表

    # 如果最优解不在禁忌表中，则将最优解放到禁忌表中
    if candidate[best_candidate_index] not in tabu_list:
        tabu_list.append(candidate[best_candidate_index])
        tabu_time.append(tabu_limit)
        current_tabu_num += 1
    else:
        sorted_length = sorted(candidate_length)
        # 如果最优解在禁忌表中，则将邻域中不在禁忌表且最好的解放入禁忌表中作为当前解
        for length in sorted_length[2:]:

            seq = candidate[length]
            if length>len(sorted_length)/3:
                current_route = candidate[2]
                break
            if seq not in tabu_list:
                current_route = seq
                tabu_list.append(current_route)
                tabu_time.append(tabu_limit)

                break


# 更新禁忌表和禁忌周期表
def update_tabu_infor():
    global current_tabu_num
    global tabu_list
    global tabu_time

    del_num = 0
    temp = [0 for col in range(city_num)]
    tabu_time = [x-1 for x in tabu_time]
    # 达到期限则释放
    for i in range(current_tabu_num):
        if tabu_time[i] == 0:
            del_num += 1
            tabu_list[i] = temp
    current_tabu_num -= del_num
    # 从记录特赦时间的表中移除特赦时间为0的位置
    while 0 in tabu_time:
        tabu_time.remove(0)
    # 同上，移除释放的元素
    while temp in tabu_list:
        tabu_list.remove(temp)


if __name__ == "__main__":
    file_path = "22_TS.xlsx"
    result = []
    start_time = time.time()
    # 读取数据
    city_list = get_data("ulysses22.tsp")
    city_num = len(city_list)
    # 城市距离矩阵
    distance_matrix = matrix(city_list)
    # 禁忌表
    tabu_list = []
    tabu_time = []
    # 当前禁忌表对象数量
    current_tabu_num = 0
    # 禁忌表长度
    tabu_limit = 300
    # 选取候选邻域点个数
    neighbor_num = city_num * (city_num - 1)
    # 候选集
    candidate = [[0 for cols in range(city_num)] for rows in range(neighbor_num)]  # 200行，每行记录一条路径
    candidate_length = [0 for _ in range(len(candidate))]  # 记录候选集每一条路径长度
    # 最佳路径最佳距离及该次迭代的最佳路径与最佳距离
    best_route = []
    best_distance = 0
    current_route = []
    current_distance = 0
    initialize_parameters()
    # 邻域解个数
    neigbor_num = city_num*(city_num-1)
    for rt in range(1000):
        # 获取邻域解
        get_candidate(neigbor_num)
        # 更新禁忌表
        update_tabu_infor()
        end_time = time.time()
        # 第几次实验，第几次外循环结束，耗时多久，此时刻最好距离
        result.append([0, rt, end_time-start_time, best_distance])
    df = pd.DataFrame(result, columns=None)
    df.to_excel(file_path)








