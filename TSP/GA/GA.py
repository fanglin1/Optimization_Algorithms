# -*- coding = utf-8 -*-
# @Time : 2022/11/26 20:09
# @Author : zackf
# @File : TS_单代号.py
# @Software : PyCharm
import copy
import math
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# 初始化种群
def init_group(city_num1, individual_num):
    '''
    :param city_num1: 城市数量,即条数据的维度
    :param individual_num: 个体数量
    :return: 产生的种群
    '''
    old_group = []
    for i in range(individual_num):
        # 产生一个按序排序的列表
        base_group = [i for i in range(city_num1)]
        # 打乱列表
        random.shuffle(base_group)
        old_group.append(base_group)
    return old_group


# 距离计算
def calculate_distance(seq):  # 计算路径距离
    '''
    :param seq: 染色体序列
    :return: 路径长度
    '''
    distances1 = 0
    length = len(seq)
    for t in range(length-1):
        distances1 += distance_matrix[seq[t]][seq[t+1]]
    distances1 += distance_matrix[seq[length-1]][seq[0]]
    return distances1

# 交叉
def crossing(seq_list1):
    '''
    :param seq_list: 种群内已有的所有个体
    :return: 子代
    '''
    # 子代数量 = 种群所需数量-已有个体数量
    seq_list = copy.deepcopy(seq_list1)
    child_num = individual_num-len(seq_list)
    children = []
    add_list = []
    while len(children) < child_num:
        # 生成两条染色体的索引
        mother_index = random.randint(0, len(seq_list)-2)
        father_index = random.randint(mother_index+1, len(seq_list)-1)
        mo_genes = []
        fa_genes = []
        # 选择两条染色体
        for gene in seq_list[mother_index]:
            mo_genes.append(gene)
        for gene in seq_list[father_index]:
            fa_genes.append(gene)


        add_list.append(mo_genes)
        add_list.append(fa_genes)
        # 交叉的开始和结束位置
        start_index = random.randint(0, len(mo_genes)-2)
        end_index = random.randint(start_index, len(mo_genes))
        pos1_dic = {value: idx for idx,value in enumerate(mo_genes)}
        pos2_dic = {value: idx for idx,value in enumerate(fa_genes)}
        # 进行交叉操作
        for j in range(start_index,  end_index):
            value1, value2 = mo_genes[j], fa_genes[j]
            pos1, pos2 = pos1_dic[value2], pos2_dic[value1]
            # 互换
            mo_genes[j], mo_genes[pos1] = mo_genes[pos1], mo_genes[j]
            fa_genes[j], fa_genes[pos2] = fa_genes[pos2], fa_genes[j]
            # 更新位置
            pos1_dic[value1], pos1_dic[value2] = pos1, j
            pos2_dic[value1], pos2_dic[value2] = j, pos2
            if j == end_index-1:
                children.append(mo_genes)
                children.append(fa_genes)
    # if children == add_list:
    #     print("相同")
    return children


# 变异
def mutation(children):
    '''
    :param children: 交叉生成的子代
    :return: 变异后的子代
    '''

    mutated_children=[]
    for i in range(len(children)):
        # 随机生成一个数，小于变异概率则进行变异
        if random.random() < mutate_prob:
            # 选择变异开始和结束位置
            u = random.randint(0, len(children[i]) - 2)
            v = random.randint(u + 1, len(children[i]) - 1)

            child_x = children[i][u:v]
            # 颠倒
            child_x.reverse()
            child = children[i][0:u] + child_x + children[i][v:]
            mutated_children.append(child)
        else:
            mutated_children.append(children[i])

    # if mutated_children == children:
    #     print("变异失效")
    return mutated_children


def select(population):
    graded = []
    for seq in population:
        graded.append(calculate_distance(seq))

    group_min_distance = min(graded)
    index = graded.index(group_min_distance)
    min_city_list = population[index]

    # 计算适应度
    fit_value = []  # 存储每个个体的适应度
    for i in range(len(graded)):
        fit_value.append(1 / graded[i] ** 20)
    # 适应度总和
    total_fit = 0
    for i in range(len(fit_value)):
        total_fit += fit_value[i]

    # 计算每个适应度占适应度总和的比例
    newfit_value = []  # 储存每个个体轮盘选择的概率
    for i in range(len(fit_value)):
        newfit_value.append(fit_value[i] / total_fit)

    # 计算累计概率
    t = 0
    for i in range(len(newfit_value)):
        t = t + newfit_value[i]
        newfit_value[i] = t

    # 生成随机数序列用于选择和比较
    ms = []  # 随机数序列
    for i in range(len(population)):
        ms.append(random.random())
    ms.sort()

    # 轮盘赌选择法
    i = 0
    j = 0
    parents = []
    while i < len(population):
        # 选择--累积概率大于随机概率
        if (ms[i] < newfit_value[j]):
            if population[j] not in parents:
                parents.append(population[j])
            i = i + 1
        # 不选择--累积概率小于随机概率
        else:
            j = j + 1
    return parents,group_min_distance,min_city_list


if __name__ == "__main__":
    file_path = "22_GA.xlsx"
    result = []

    start_time = time.time()
    iter_num = 1000  # 迭代次数
    mutate_prob = 0.5  # 变异概率
    # 读取数据坐标
    positions = get_data("ulysses22.tsp")
    # 城市数量
    city_num = len(positions)
    individual_num =  city_num*(city_num-1)# 种群数
    # 距离矩阵
    distance_matrix = matrix(positions)
    # 初始化总群
    old_group = init_group(city_num, individual_num)
    best_list = []
    min_distance = 0
    min_distance_list = []
    # print("初始种群", old_group)
    for i in range(iter_num):
        # print("原始种群数目",len(old_group))
        children_group = crossing(old_group)
        # print("新生成数目", len(children_group))
        mutated_group = mutation(children_group)
        # print("新生成数目", len(mutated_group))
        total_group = old_group + mutated_group

        old_group, group_min_distance, min_city_list = select(total_group)
        # print("筛选后种群数", len(old_group))
        # print(old_group)
        if i == 0:
            min_distance = group_min_distance
            best_list = min_city_list
        else:
            if min_distance > group_min_distance:
                min_distance = group_min_distance
                best_list = min_city_list
        min_distance_list.append(min_distance)
        end_time = time.time()
        result.append([0, i, end_time - start_time, min_distance])
    df = pd.DataFrame(result, columns=None)
    df.to_excel(file_path)
            # print("第", i, "次循环最小路径为", best_list)















