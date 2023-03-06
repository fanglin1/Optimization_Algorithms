# -*- coding = utf-8 -*-
# @Time : 2022/12/4 19:50
# @Author : zackf
# @File : ACO_双代号网络.py
# @Software : PyCharm
import math
import random
import time as ti
import pandas as pd


class Node:
    def __init__(self, its_id=0):
        self.its_id = its_id  # 结点id
        self.pre_node = []  # 前结点
        self.next_node = []  # 后续结点


def createNodes(nodeNum):
    '''
    产生结点对象
    :param nodeNum: 结点数目
    :return:
    '''
    node_list = []
    for i in range(nodeNum):
        node = Node(i)
        node_list.append(node)
    return node_list


def get_data(path):
    '''
    读取数据
    :param path: 数据文件路径
    :return: 结点数量，活动数量，结点对象，资源，活动信息，活动头尾结点
    '''
    datafile = open("data.txt")
    data_content = datafile.read().split("\n")
    datafile.close()
    general_info = data_content[0].split("%*%")

    node_num = int(general_info[0])  # 结点数
    activity_num = int(general_info[1])  # 活动数
    resource = [int(general_info[2]), int(general_info[3])]  # 2类资源的数量
    node_list = createNodes(node_num)
    # data:存储工期及资源消耗
    # activities:存储活动
    data = [[0 for i in range(4)] for j in range(activity_num)]
    activities = [[0 for i in range(2)] for j in range(activity_num)]
    data_info = data_content[2:]
    for datas in data_info:
        data_detail = datas.split("/")
        data1 = data_detail[0].split(" ")

        i = int(data1[0])
        data[i][0] = int(data1[1])  #工期
        data[i][1] = int(data1[2])  #资源1
        data[i][2] = int(data1[3])  #资源2

        data2 = data_detail[1].split("-")
        head_node = int(data2[0])
        tail_node = int(data2[1])
        node_list[head_node].next_node.append(tail_node)  # 设置结点的紧后结点
        node_list[tail_node].pre_node.append(head_node)  # 设置结点的紧前结点
        activities[i][0] = head_node
        activities[i][1] = tail_node
    return node_num, activity_num, node_list, resource, data, activities


# 初始化启发项矩阵
def init_e(activity_num):
    '''
    初始化启发项矩阵
    :param activity_num: 活动数量
    :return:
    '''
    e_matrix = [0 for i in range(activity_num)]
    for i in range(activity_num):
        e_matrix[i] = (data[i][1]+data[i][2]+data[i][3])/(resource[0]+resource[1])
    return e_matrix


# 信息素矩阵
def init_pheromone(activity_num1):
    '''
    初始化信息素矩阵
    :param activity_num1: 活动数量
    :return: 初始化后的信息素矩阵
    '''
    pheromone_matrix = [1/activity_num1 for _ in range(activity_num1)]
    return pheromone_matrix


# 参数初始化
def init_parameters():
    cur_time = 0 # 当前时间
    res_time = [] #当前时间下以排班但还未结束的活动

    events_finished = [] #前序活动均已完成的世界
    events_finished.append(0)

    activity_waited = [] #等待被选的活动
    activity_waited.append(0)

    activity_finished = []
    R1 = resource[0]
    R2 = resource[1]

    return cur_time, res_time, events_finished, activity_waited, R1, R2, activity_finished


# 时间约束检验
def pre_time_check(activity):
    '''
    检查被选择活动是否前序活动均已完成，若未全部完成，则释放活动，更新资源和实际
    :param activity: 被选择活动
    :return:
    '''
    global cur_time
    # 活动头结点
    head_node = activities[activity][0]
    should_finish_list = []

    for i in range(len(activities)):
        # print(activities[i][1])
        if activities[i][1]==head_node:
            # 还未完成的活动列表
            should_finish_list.append(i)
    pre_max_time = -1
    # res全局变量，表示已排班但当前还未完成的活动
    for res in res_time:
        if res[1] in should_finish_list:
            if res[0] > pre_max_time:
                pre_max_time=res[0]
    if pre_max_time>=0:
        # 更新时间，释放资源
        cur_time = pre_max_time
        time_ahead()

# 资源检验
def resource_check(activity):
    '''
    检查资源是否足够，当剩余资源不足以支持被选活动消耗时，释放正在排班的活动
    :param activity: 被选择的活动
    :return:
    '''
    global R1, R2, cur_time
    while True:
        # 资源足够，安排活动并更新资源剩余量
        if data[activity][1] <= R1 and data[activity][2] <= R2:
            R1 -= data[activity][1]
            R2 -= data[activity][2]
            break
        # 资源不足，释放活动，直至资源足够
        elif res_time:
            cur_time = min(i[0] for i in res_time)
            time_ahead()
        else:
            print(activity, R1, R2)


def time_ahead():
    '''
    释放资源，并更新还在排班中的活动
    :return:
    '''
    global cur_time
    global R1, R2, res_time
    # remove_element在time时间下应该被完成的活动
    remove_element = []
    for res in res_time:
        if res[0] <= cur_time:
            # 释放资源
            R1 += data[res[1]][1]
            R2 += data[res[1]][2]
            remove_element.append(res)
    # 更新已排班但还未到结束时间的活动
    res_time = [res for res in res_time if res not in remove_element]

def check_finish(finish_node_list):
    '''
    检查所有活动是否均已完成
    :param finish_node_list:
    :return:
    '''
    for i in range(node_num-2):
        if i not in finish_node_list:
            return False
    if node_num - 2 in finish_node_list:
        return False
    return True


# 更新活动列表以及已完成活动
def update_list(activity):
    '''

    :param activity: 被选中的活动
    :return:
    '''
    global cur_time
    # 尾结点
    next_node = activities[activity][1]
    activity_waited.remove(activity)

    if len(activity_waited) == 0 and check_finish(events_finished):
        events_finished.append(node_num-1)
        activity_waited.append(activity_num-1)
    elif next_node != node_num-1 and next_node not in events_finished:
        # 如果尾结点的前序活动只有activity一个，则将尾结点更新为已完成结点
        if len(node_list[next_node].pre_node) == 1:
            events_finished.append(next_node)
        else:
            t = []
            for i in range(len(activity_finished)):
                # 获取已被选择的活动的头结点和尾结点
                t.append(activities[activity_finished[i][0]])
            num = 0
            for i in node_list[next_node].pre_node:
                if [i, next_node] not in t:
                    num += 1
            ''' 如果只有一个活动不在该结点的前序活动中，即本次的活动，则代表其他前序活动均已完成
            ，本次活动被选择后该结点全部活动均完成,故将其放入已完成事件中'''
            if num == 1:
                events_finished.append(next_node)
        # 更新下次可选择的活动
        for node_id in node_list[next_node].next_node:
            node = node_list[node_id]
            is_finished = True
            for i in node.pre_node:
                if i not in events_finished:
                    is_finished = False
            if is_finished:
                for i in node.pre_node:
                    activity_waited.append(activities.index([i, node.its_id]))
    res_time.append([cur_time+data[activity][0], activity])
    activity_finished.append([activity, cur_time+data[activity][0]])



# 进行下一个活动
def step_over(activity):
    global R1, R2, cur_time
    # 时序检查
    pre_time_check(activity)
    # 资源检查
    resource_check(activity)
    # 活动状态更新
    update_list(activity)



# 蚁群算法主程序
if __name__ == "__main__":
    file_path = "D:\研\资料\datasets_result\RCPSP_data_ACO.xlsx"
    result_lis = []
    for t in range(5):
        start_time = ti.time()
        # 数据获取
        node_num, activity_num, node_list, resource, data, activities = get_data("data.txt")
        e_matrix = init_e(activity_num)
        # 种群大小
        ant_num = (activity_num-1)*activity_num
        alpha = 0.1
        beta = 1
        rho = 0.4
        Q = 1
        iter_num = 500
        # 信息素矩阵
        pheromone_matrix = init_pheromone(activity_num)
        # 最佳时间
        best_time = 0
        # 最佳序列
        best_list = []
        count = 0
        for i in range(iter_num):
            activity_list_result = []  # 记录各蚂蚁的走的解
            for ant in range(ant_num):
                cur_time, res_time, events_finished, activity_waited, R1, R2, activity_finished = init_parameters()
                for j in range(activity_num): # 轮盘法选择下一个活动
                    if len(activity_waited) == 1:
                        # 如果只有一个等待选择的活动，则直接选
                        step_over(activity_waited[0])

                    else:
                        # 轮盘法选择
                        trans_list = []
                        trans_sum = 0
                        for t in range(len(activity_waited)):
                            trans_sum += math.pow(pheromone_matrix[activity_waited[t]], alpha)*\
                                         math.pow(e_matrix[activity_waited[t]],beta)
                            trans_list.append(trans_sum)
                        rand = random.uniform(0, trans_sum)
                        for k in range(len(trans_list)):
                            if trans_list[k] >= rand:
                                # 下一个活动
                                next_activity = activity_waited[k]
                                break

                        step_over(next_activity)

                activity_list_result.append(activity_finished)  # 保存该蚂蚁的结果
            # 更新最优解
            for result in activity_list_result:
                if best_time == 0:
                    best_time = result[-1][1]
                    best_list = result
                else:
                    if best_time > result[-1][1]:
                        best_time = result[-1][1]
                        best_list = result
            # 信息素矩阵更新
            for result in activity_list_result:
                time = result[-1][1]
                for activity_info in result:
                    activity_id = activity_info[0]
                    pheromone_matrix[activity_id] = (1-rho)*pheromone_matrix[activity_id] + Q/time
            end_time = ti.time()
            result_lis.append([t, i, end_time-start_time, best_time])
    df = pd.DataFrame(result_lis, columns=None)
    df.to_excel(file_path)


























