import time
from math import *
import random
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


def get_city_distances(lists):
    '''
    根据坐标计算城市间的距离
    :param lists: 各城市的坐标
    :return: 城市间的距离
    '''
    # 初始化距离矩阵
    matrix = [[0 for col in range(city_num)] for row in range(city_num)]
    for i in range(city_num):
        for j in range(city_num):
            # 计算距离
            matrix[i][j] = sqrt((lists[i][0]-lists[j][0])**2+(lists[i][1]-lists[j][1])**2)
    return matrix


def calculate_distance(seq, matrix):  # 计算路径距离
    '''
    根据行走的路径计算路径的距离长度
    :param seq: 路径列表
    :param matrix: 距离矩阵
    :return: 距离
    '''
    distance = 0
    length = len(seq)
    for t in range(length-1):
        distance += matrix[seq[t]][seq[t+1]]
    distance += matrix[seq[length-1]][seq[0]]
    return distance




def generate_new_path(old_path1):
    '''
    以三种方式，产生新的解，每种方式的概率为1/3
    :param old_path1: 旧路径
    :return: 新路径
    '''
    r = random.random()
    length = len(old_path1)
    base_p = 0.333333
    # 方式1：交换两点
    if r < base_p:
        path = []
        s = []
        # 产生两点的位置
        while len(s) < 3:
            r1 = floor(random.random() * length)
            if r1 not in s:
                s.append(r1)
        s.sort()
        # 第一片段
        for t in range(s[0]):
            path.append(old_path1[t])
        # 插入第二个位置的元素
        path.append(old_path1[s[1]])
        # 第二片段
        for t in range(s[0]+1, s[1]):
            path.append(old_path1[t])

        # 插入第一个位置的元素
        path.append(old_path1[s[0]])
        # 第三片段
        for t in range(s[1]+1,length):
            path.append(old_path1[t])

        return path
    # 方式2：三点交换，思想和过程与上相同
    elif r < 2*base_p:
        s = []
        while len(s) < 3:
            r1 = floor(random.random()*length)
            if r1 not in s:
                s.append(r1)
        s.sort()
        path = []
        for t in range(0,s[0]):
            path.append(old_path1[t])
        for t in range(s[1]+1,s[2]+1):
            path.append(old_path1[t])
        for t in range(s[0],s[1]+1):
            path.append(old_path1[t])
        for t in range(s[2]+1,length):
            path.append(old_path1[t])

        return path
    # 方式3：颠倒两点
    else:
        s = []
        path = []
        while len(s) < 2:
            r1 = floor(random.random() * length)
            if r1 not in s:
                s.append(r1)
        s.sort()
        for t in range(0, s[0]):
            path.append(old_path1[t])
        temp = old_path1[s[0]: s[1]+1]
        temp.reverse()
        for el in temp:
            path.append(el)
        for t in range(s[1]+1,length):
            path.append(old_path1[t])

        return path


# 模拟退火主体
if __name__ == "__main__":
    file_path = "22_SA.xlsx"
    result = []

    start_time = time.time()
    T0 = 1000  # 初始温度
    T = T0  # 第一次迭代温度
    maxGen = 1000  # 最大迭代次数

    al = 0.96  # 温度衰减系数
    # 读取数据坐标
    positions = get_data("ulysses22.tsp")
    # 城市数量
    city_num = len(positions)
    lk =  (city_num-1)*city_num# 每个温度下的迭代次数
    # 获得城市间的距离
    distance_matrix = get_city_distances(positions)
    # 随机初始化原始路径
    old_path = list(range(city_num))
    random.shuffle(old_path)
    # 初始化距离，并将其赋值给最小路径距离
    old_distance = calculate_distance(old_path, distance_matrix)
    min_path = old_path
    min_distance = old_distance

    min_list = []
    # 进行迭代退火，寻找最优解
    for outer_iter in range(maxGen):
        for inner_iter in range(lk):
            # 邻解（新解）
            new_path = generate_new_path(old_path)
            # 新解的长度
            new_distance = calculate_distance(new_path, distance_matrix)
            # 如果新解优于旧解则接受
            if new_distance < old_distance:
                old_path = new_path
                old_distance = new_distance
            # 新解差于旧解则以一定的概率接受，且概率会随着外层迭代次数增加而降低
            else:
                p = exp(-(new_distance-old_distance)/T)
                r = random.random()
                if r < p:
                    old_path = new_path
                    old_distance = new_distance
            # 更新最优解
            if old_distance < min_distance:
                min_distance = old_distance
                min_path = old_path
        # 内层循环结束后降温
        T = al*T
        min_list.append(min_distance)
        end_time = time.time()
        result.append([0, outer_iter, end_time - start_time, min_distance])
    df = pd.DataFrame(result, columns=None)
    df.to_excel(file_path)
    # plt.plot(min_list)
    # plt.show()
    #
    # min_path.append(min_path[0])
    # # 输出最小路径和最小距离
    # print(min_path)
    # print(min_distance)
    # # 绘图
    # plot_x=[]
    # plot_y=[]
    # for city_index in min_path:
    #     plot_x.append(positions[city_index][0])
    #     plot_y.append(positions[city_index][1])
    #
    # print(len(plot_x))
    # plt.plot(plot_x,plot_y)
    # plt.show()
    # plt.plot(list(range(len(min_list))),min_list)
    # plt.show()


